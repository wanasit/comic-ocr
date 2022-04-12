import logging
import os
from typing import Optional, Callable

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from manga_ocr.models.localization.localization_dataset import LocalizationDataset
from manga_ocr.models.localization.localization_model import LocalizationModel


logger = logging.getLogger(__name__)


def train(
        model: LocalizationModel,
        training_dataset: LocalizationDataset,
        validation_dataset: Optional[LocalizationDataset] = None,
        epoch_count: int = 1,
        epoch_callback: Optional[Callable] = None,
        tqdm_disable=False,
        batch_size=10,
        optimizer=None
):
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    training_losses = []
    validation_losses = []
    logger.info(f'Training {epoch_count} epochs, on {len(training_dataset)} samples ' +
                f'({len(validation_dataset)} validation samples)' if validation_dataset else '')

    for i_epoch in range(epoch_count):
        with tqdm(total=len(training_dataset), disable=tqdm_disable) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")

            training_loss = 0.0
            for i_batch, batch in enumerate(training_dataloader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()

                optimizer.step()

                batch_loss = loss.item()
                current_batch_size = batch['image'].size(0)

                tepoch.set_postfix(training_batch_loss=batch_loss)
                tepoch.update(current_batch_size)
                training_loss += batch_loss * current_batch_size

            training_loss = training_loss / len(training_dataset)
            training_losses.append(training_loss)

            if validation_dataset:
                validation_loss = 0.0
                valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                for i_batch, batch in enumerate(valid_dataloader):
                    current_batch_size = batch['image'].size(0)
                    logger.debug(f'> Validating with {current_batch_size} samples')
                    with torch.no_grad():
                        loss = model.compute_loss(batch)
                        validation_loss += loss.item() * current_batch_size
                validation_losses.append(validation_loss / len(validation_dataset))
            else:
                validation_loss = None

            tepoch.set_postfix(training_loss=training_loss, validation_loss=validation_loss)
            logger.info(f'> Finished training with training_loss={training_loss}, validation_loss={validation_loss}')

            if epoch_callback:
                epoch_callback()

    return (training_losses, validation_losses) if validation_dataset else (training_losses, None)


if __name__ == '__main__':
    from manga_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from manga_ocr.utils import get_path_project_dir

    path_dataset = get_path_project_dir('data/output/generate_manga_dataset')
    path_output_model = get_path_project_dir('data/output/models/localization.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    dataset = LocalizationDataset.load_generated_manga_dataset(path_dataset, image_size=model.image_size)
    print(f'Loaded generated manga dataset of size {len(dataset)}...')

    validation_dataset = dataset.subset(to_dix=100)
    training_dataset = dataset.subset(from_idx=100)
    train(model,
          training_dataset=training_dataset,
          validation_dataset=validation_dataset,
          epoch_callback=lambda: torch.save(model, path_output_model),
          epoch_count=10)
