import logging
import os
import collections
from typing import Optional

import torch
from torch.utils import tensorboard
from tqdm import tqdm

from comic_ocr.models import train_helpers
from comic_ocr.models.recognition import recognition_dataset, recognition_model, calculate_high_level_metrics
from comic_ocr.utils import files

logger = logging.getLogger(__name__)


def train(
        model_name: str,
        model: recognition_model.RecognitionModel,
        train_dataset: recognition_dataset.RecognitionDataset,
        train_epoch_count: int = 25,
        train_device: Optional[torch.device] = None,
        validate_dataset: Optional[recognition_dataset.RecognitionDataset] = None,
        validate_device: Optional[torch.device] = None,
        update_callback: Optional[train_helpers.UpdateCallback] = None,
        update_every_n: Optional[int] = 20,
        update_validate_sample_size: Optional[int] = None,
        batch_size=50,
        optimizer=None,
        lr_scheduler=None,
        tqdm_enable=True,
        tensorboard_log_enable=True,
        tensorboard_log_dir=None
):
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

    # hack: try different optimizer
    optimizer = optimizer if optimizer else torch.optim.SGD(
        model.parameters(), lr=0.02, nesterov=True, weight_decay=1e-5, momentum=0.9)
    lr_scheduler = lr_scheduler if lr_scheduler else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=5)

    validation_metrics: train_helpers.HistoricalMetrics = collections.defaultdict(list)
    training_metrics: train_helpers.HistoricalMetrics = collections.defaultdict(list)
    step_counter = 0
    for i_epoch in range(train_epoch_count):
        with tqdm(total=len(train_dataset), disable=(not tqdm_enable)) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")
            training_dataloader = train_dataset.loader(batch_size=batch_size, shuffle=True)
            for batch in training_dataloader:
                model = model.train()
                step_counter += 1

                # Compute loss
                optimizer.zero_grad()
                loss = model.compute_loss(batch)

                # Step loss / optimizer / lr_scheduler
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # TODO: try tuning this later
                optimizer.step()
                lr_scheduler.step(loss)

                current_batch_loss = loss.item()
                current_batch_size = batch['image'].shape[0]
                tepoch.update(current_batch_size)
                tepoch.set_postfix(
                    current_batch_loss=current_batch_loss)

                if update_every_n and step_counter % update_every_n == 0:
                    model = model.eval()

                    # Update stats on training_dataset (and write to tensorbaord)
                    _validate_model(training_metrics, model, train_dataset,
                                    batch_size=batch_size,
                                    sample_size_limit=batch_size * 2,
                                    device=validate_device)
                    if writer_train:
                        # TODO: Make this work on ReduceLROnPlateau
                        # writer_train.add_scalar('lr', lr_scheduler.get_last_lr()[-1], step_counter)
                        for k in training_metrics:
                            writer_train.add_scalar(k, training_metrics[k][-1], step_counter)

                    if validate_dataset:
                        # Update stats on validate_dataset (and write to tensorbaord)
                        _validate_model(validation_metrics, model, validate_dataset, batch_size=batch_size,
                                        sample_size_limit=update_validate_sample_size,
                                        device=validate_device)
                        if writer_validate:
                            # TODO: Make this work on ReduceLROnPlateau
                            # writer_validate.add_scalar('lr', lr_scheduler.get_last_lr()[-1], step_counter)
                            for k in validation_metrics:
                                writer_validate.add_scalar(k, validation_metrics[k][-1], step_counter)

                    if update_callback:
                        update_callback(step_counter, training_metrics, validation_metrics)

    return training_metrics, validation_metrics


def _validate_model(metrics, model, dataset, batch_size,
                    sample_size_limit: Optional[int] = None,
                    device: Optional[torch.device] = None):
    device = device if device else torch.device('cpu')
    model = model.to(device).eval()
    loss = _calculate_avg_loss(model, dataset, batch_size=batch_size, sample_size_limit=sample_size_limit,
                               device=device)
    metrics['loss'].append(loss)
    if len(dataset) > 0:
        new_metrics = calculate_high_level_metrics(model, dataset, sample_size_limit=sample_size_limit, device=device)
        for k in new_metrics:
            metrics[k].append(new_metrics[k])


def _calculate_avg_loss(
        model: recognition_model.RecognitionModel,
        dataset: recognition_dataset.RecognitionDataset,
        batch_size: int,
        sample_size_limit: Optional[int] = None,
        device: Optional[torch.device] = None) -> float:
    sample_size_limit = sample_size_limit if sample_size_limit else len(dataset)
    dataset = dataset if sample_size_limit >= len(dataset) else dataset.shuffle().subset(to_idx=sample_size_limit)

    total_loss = 0
    with torch.no_grad():
        valid_dataloader = dataset.loader(batch_size=batch_size, num_workers=0)
        for i_batch, batch in enumerate(valid_dataloader):
            loss = model.compute_loss(batch, device=device)
            total_loss += loss.item()

    return total_loss / sample_size_limit


if __name__ == '__main__':
    from comic_ocr.models.recognition.crnn.crnn import CRNN
    from comic_ocr.models.recognition.trocr.trocr import TrOCR

    path_dataset = files.get_path_project_dir('example/manga_generated')
    dataset = recognition_dataset.RecognitionDataset.load_generated_dataset(path_dataset)
    print(f'Loaded generated manga dataset of size {len(dataset)}...')

    model = CRNN.create_small_model()
    # path_output_model = get_path_project_dir('data/output/models/recognition.bin')
    # if os.path.exists(path_output_model):
    #     print('Loading an existing model...')
    #     model = torch.load(path_output_model)

    validation_dataset = dataset.subset(to_idx=5)
    training_dataset = dataset.subset(from_idx=5)


    def update(epoch, training_losses, validation_metrics):
        print('Update')


    train('test', model,
          train_dataset=training_dataset,
          validate_dataset=validation_dataset,
          update_callback=update,
          train_epoch_count=5)
