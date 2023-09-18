import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import torch
from torch import optim
from torch.utils import tensorboard
from tqdm import tqdm

from comic_ocr.models.localization import calculate_high_level_metrics
from comic_ocr.models.localization import localization_dataset
from comic_ocr.models.localization import localization_model
from comic_ocr.utils import files

logger = logging.getLogger(__name__)

HistoricalMetrics = Dict[str, List[float]]
UpdateCallback = Callable[[int, HistoricalMetrics, HistoricalMetrics], None]


def callback_to_save_model(model, model_path) -> UpdateCallback:
    def save(steps, training_metrics, validation_metrics):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.to('cpu')
        torch.save(model, model_path)

    return save


def callback_to_save_model_on_increasing_metric(model, model_path, metric_name) -> UpdateCallback:
    save_model_callback = callback_to_save_model(model, model_path)
    metric_value = []

    def save(steps, training_metrics, validation_metrics):
        if not metric_value or metric_value[0] < validation_metrics[metric_name][-1]:
            save_model_callback(steps, training_metrics, validation_metrics)
            metric_value.clear()
            metric_value.append(validation_metrics[metric_name][-1])

    return save


def compute_loss_for_each_sample(
        model: localization_model.LocalizationModel,
        dataset: localization_dataset.LocalizationDataset,
        loss_criterion_for_char: Optional[Callable] = None,
        loss_criterion_for_line: Optional[Callable] = None,
        device: torch.device = torch.device('cpu')):
    if isinstance(dataset, localization_dataset.LocalizationDatasetWithAugmentation):
        dataset = dataset.without_augmentation()

    loader = dataset.loader(batch_size=1, shuffle=False)
    results = []

    for i, batch in enumerate(loader):
        model.eval()
        loss = model.compute_loss(
            batch,
            loss_criterion_for_char=loss_criterion_for_char,
            loss_criterion_for_line=loss_criterion_for_line,
            device=device)
        results += [loss]
    return results


def train(
        model_name: str,
        model: localization_model.LocalizationModel,
        train_dataset: localization_dataset.LocalizationDataset,
        train_epoch_count: int = 25,
        train_device: Optional[torch.device] = None,
        validate_dataset: Optional[localization_dataset.LocalizationDataset] = None,
        validate_device: Optional[torch.device] = None,
        update_callback: Optional[UpdateCallback] = None,
        update_every_n: Optional[int] = 20,
        update_validate_sample_size: Optional[int] = None,
        batch_size=10,
        optimizer=None,
        lr_scheduler=None,
        loss_criterion_for_char: Optional[Callable] = None,
        loss_criterion_for_line: Optional[Callable] = None,
        tqdm_enable=True,
        tensorboard_log_enable=True,
        tensorboard_log_dir=None
) -> Tuple[HistoricalMetrics, HistoricalMetrics]:
    logger.info(f'Training {train_epoch_count} epochs, on {len(train_dataset)} samples ' +
                f'({len(validate_dataset)} validation samples)' if validate_dataset else '')

    writer_validate = None
    writer_train = None
    if tensorboard_log_enable:
        if tensorboard_log_dir is None:
            tensorboard_log_dir = files.get_path_project_dir(f'data/logs/{model_name}')
        logger.info(f'Writing tensorboard logs at {tensorboard_log_dir}')
        writer_validate = tensorboard.SummaryWriter(log_dir=f'{tensorboard_log_dir}/validate')
        writer_train = tensorboard.SummaryWriter(log_dir=f'{tensorboard_log_dir}/train')

    train_steps = train_epoch_count * ((len(train_dataset) + batch_size - 1) // batch_size)
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = lr_scheduler if lr_scheduler else optim.lr_scheduler.StepLR(optimizer,
                                                                               step_size=int(max(train_steps * 0.8, 1)),
                                                                               gamma=0.1)
    validation_metrics: HistoricalMetrics = defaultdict(list)
    training_metrics: HistoricalMetrics = defaultdict(list)
    step_counter = 0
    for i_epoch in range(train_epoch_count):
        with tqdm(total=len(train_dataset), disable=(not tqdm_enable)) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")
            training_dataloader = train_dataset.loader(batch_size=batch_size, shuffle=True)
            for batch in training_dataloader:
                model = model.train()
                step_counter += 1

                # Update the model
                optimizer.zero_grad()
                loss = model.compute_loss(
                    batch,
                    loss_criterion_for_char=loss_criterion_for_char,
                    loss_criterion_for_line=loss_criterion_for_line,
                    device=train_device)

                # Step loss / optimizer / lr_scheduler
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                current_batch_loss = loss.item()
                current_batch_size = batch['input'].size(0)

                if update_every_n and step_counter % update_every_n == 0:
                    model = model.eval()
                    # Update stats on training_dataset and write to tensorbaord
                    _validate_model(training_metrics, model, train_dataset,
                                    batch_size=batch_size,
                                    sample_size_limit=batch_size * 2,
                                    loss_criterion_for_char=loss_criterion_for_char,
                                    loss_criterion_for_line=loss_criterion_for_line,
                                    device=validate_device)
                    if writer_train:
                        writer_train.add_scalar('lr', lr_scheduler.get_last_lr()[-1], step_counter)
                        for k in training_metrics:
                            writer_train.add_scalar(k, training_metrics[k][-1], step_counter)

                    # Update stats on validate_dataset and write to tensorbaord
                    if validate_dataset:
                        _validate_model(validation_metrics, model, validate_dataset, batch_size=batch_size,
                                        sample_size_limit=update_validate_sample_size,
                                        loss_criterion_for_char=loss_criterion_for_char,
                                        loss_criterion_for_line=loss_criterion_for_line,
                                        device=validate_device)
                        if writer_validate:
                            writer_validate.add_scalar('lr', lr_scheduler.get_last_lr()[-1], step_counter)
                            for k in validation_metrics:
                                writer_validate.add_scalar(k, validation_metrics[k][-1], step_counter)

                    if update_callback:
                        update_callback(step_counter, training_metrics, validation_metrics)

                tepoch.update(current_batch_size)
                tepoch.set_postfix(
                    current_batch_loss=current_batch_loss)

    return training_metrics, validation_metrics


def _validate_model(metrics, model, dataset, batch_size, sample_size_limit=None,
                    loss_criterion_for_char: Optional[Callable] = None,
                    loss_criterion_for_line: Optional[Callable] = None,
                    device: Optional[torch.device] = None):
    device = device if device else torch.device('cpu')
    model = model.to(device)
    loss = _calculate_avg_loss(model, dataset, batch_size=batch_size, sample_size_limit=sample_size_limit,
                               loss_criterion_for_char=loss_criterion_for_char,
                               loss_criterion_for_line=loss_criterion_for_line,
                               device=device)
    metrics['loss'].append(loss)
    if len(dataset) > 0 and len(dataset.get_line_locations(0)) > 0:
        new_metrics = calculate_high_level_metrics(model, dataset, sample_size_limit=sample_size_limit, device=device)
        for k in new_metrics:
            metrics[k].append(new_metrics[k])


def _calculate_avg_loss(
        model: localization_model.LocalizationModel,
        dataset: localization_dataset.LocalizationDataset,
        batch_size: int,
        sample_size_limit: Optional[int] = None,
        loss_criterion_for_char: Optional[Callable] = None,
        loss_criterion_for_line: Optional[Callable] = None,
        device: Optional[torch.device] = None) -> float:
    sample_size_limit = sample_size_limit if sample_size_limit else len(dataset)
    dataset = dataset if sample_size_limit >= len(dataset) else dataset.shuffle().subset(to_idx=sample_size_limit)

    total_loss = 0
    with torch.no_grad():
        valid_dataloader = dataset.loader(batch_size=batch_size, num_workers=0)
        for i_batch, batch in enumerate(valid_dataloader):
            loss = model.compute_loss(batch, device=device, loss_criterion_for_char=loss_criterion_for_char,
                                      loss_criterion_for_line=loss_criterion_for_line)
            total_loss += loss.item()

    return total_loss / sample_size_limit


def _run_example():
    from comic_ocr.models.localization.conv_unet import conv_unet

    model_name = 'test_training_localization'
    model_path = files.get_path_project_dir(f'data/output/models/{model_name}.bin')
    if os.path.exists(model_path):
        print('Loading an existing model...')
        model = torch.load(model_path)
    else:
        print('Creating a new model...')
        model = conv_unet.BaselineConvUnet()

    dataset = localization_dataset.LocalizationDataset.load_line_annotated_manga_dataset(
        files.get_path_project_dir('data/manga_line_annotated'))

    save_model = callback_to_save_model_on_increasing_metric(model, model_path, 'line_level_precision')

    def update(epoch, training_losses, validation_metrics):
        print('Update')
        save_model(epoch, training_losses, validation_metrics)

    validation_dataset = dataset.subset(to_idx=2)
    training_dataset = dataset.subset(from_idx=2, to_idx=10)
    train(model_name, model,
          train_dataset=training_dataset,
          validate_dataset=validation_dataset,
          update_callback=update,
          batch_size=3,
          update_every_n=1,
          train_epoch_count=1)


if __name__ == '__main__':
    _run_example()
