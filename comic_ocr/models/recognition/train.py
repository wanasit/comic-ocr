import logging
import os
from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset
from comic_ocr.models.recognition.recognition_model import RecognitionModel
from comic_ocr.utils import get_path_project_dir

logger = logging.getLogger(__name__)


def train(
        model: RecognitionModel,
        training_dataset: RecognitionDataset,
        validation_dataset: Optional[RecognitionDataset] = None,
        validation_every_n: Optional[int] = 200,
        epoch_count: int = 1,
        epoch_callback: Optional[Callable] = None,
        update_callback: Optional[Callable] = None,
        update_every_n: Optional[int] = 200,
        tqdm_disable=False,
        batch_size=50,
        optimizer=None,
        scheduler=None
):
    # hack: try different optimizer
    optimizer = optimizer if optimizer else torch.optim.SGD(
        model.parameters(), lr=0.02, nesterov=True, weight_decay=1e-5, momentum=0.9)
    scheduler = scheduler if scheduler else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=5)

    training_losses = []
    validation_losses = []

    logger.info(f'Training {epoch_count} epochs, on {len(training_dataset)} samples ' +
                f'({len(validation_dataset)} validation samples)' if validation_dataset else '')

    for i_epoch in range(epoch_count):

        # currently, we can't have batch training because of the padding
        # todo: support different batch size when it's possible
        training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0) \
            if validation_dataset else None

        with tqdm(total=len(training_dataset), disable=tqdm_disable) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")

            batch_loss = 0.0
            for i_batch, batch in enumerate(training_dataloader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()

                # hack: try tuning this later
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                # Todo: remove this when we can set the real batch_size in dataloader
                batch_loss += loss.item()
                tepoch.update(1)

                if tepoch.n % batch_size == 0:
                    batch_loss_avg = batch_loss / batch_size
                    batch_loss = 0

                    tepoch.set_postfix(training_batch_loss=batch_loss_avg)
                    training_losses.append(batch_loss_avg)

                if validation_every_n >= 0 and tepoch.n % validation_every_n == 0:
                    if valid_dataloader:
                        validation_losses.append(_calculate_validation_loss(model, valid_dataloader))
                        scheduler.step(validation_losses[-1])

                if update_every_n >= 0 and tepoch.n % update_every_n == 0:
                    if update_callback:
                        update_callback(tepoch.n, training_losses, validation_losses)

            if valid_dataloader:
                validation_losses.append(_calculate_validation_loss(model, valid_dataloader))
                scheduler.step(validation_losses[-1])

            if epoch_callback:
                epoch_callback(i_epoch, training_losses, validation_losses)

    return (training_losses, validation_losses) if validation_dataset else (training_losses, None)


def _calculate_validation_loss(model, valid_dataloader):
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for i_batch, batch in enumerate(valid_dataloader):
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            total_count += batch['input'].size(0)

    return total_loss / total_count


if __name__ == '__main__':
    from comic_ocr.models.recognition.crnn.crnn import CRNN
    from comic_ocr.models.recognition.trocr.trocr import TrOCR

    path_dataset = get_path_project_dir('data/output/generate_manga_dataset')
    path_output_model = get_path_project_dir('data/output/models/recognition.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = CRNN()

    dataset = RecognitionDataset.load_generated_dataset(path_dataset)
    print(f'Loaded generated manga dataset of size {len(dataset)}...')

    validation_dataset = dataset.subset(to_idx=100)
    training_dataset = dataset.subset(from_idx=100, to_idx=200)

    training_dataset.get_line_image(0).show()
    training_dataset.get_line_image(1).show()
    validation_dataset.get_line_image(0).show()
    validation_dataset.get_line_image(1).show()

    train(model,
          training_dataset=training_dataset,
          validation_dataset=validation_dataset,
          epoch_callback=lambda: torch.save(model, path_output_model),
          epoch_count=10)
