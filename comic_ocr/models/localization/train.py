import logging
import os
from typing import Optional, Callable

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from comic_ocr.models.localization.localization_dataset import LocalizationDataset
from comic_ocr.models.localization.localization_model import LocalizationModel
from comic_ocr.typing import Size
from comic_ocr.utils.pytorch_model import calculate_validation_loss

logger = logging.getLogger(__name__)


def train(
        model: LocalizationModel,
        train_dataset: LocalizationDataset,
        validate_dataset: Optional[LocalizationDataset] = None,
        validate_every_n: Optional[int] = 200,
        update_callback: Optional[Callable] = None,
        update_every_n: Optional[int] = 200,
        epoch_count: int = 1,
        epoch_callback: Optional[Callable] = None,
        tqdm_disable=False,
        batch_size=10,
        optimizer=None
):
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)

    logger.info(f'Training {epoch_count} epochs, on {len(train_dataset)} samples ' +
                f'({len(validate_dataset)} validation samples)' if validate_dataset else '')

    all_train_losses = []
    all_validate_losses = []

    update_counter = 0
    validate_counter = 0

    for i_epoch in range(epoch_count):
        with tqdm(total=len(train_dataset), disable=tqdm_disable) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")

            epoch_training_total_loss = 0
            epoch_training_total_count = 0
            training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for i_batch, batch in enumerate(training_dataloader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()

                optimizer.step()

                current_batch_loss = loss.item()
                current_batch_size = batch['input'].size(0)

                all_train_losses.append(current_batch_loss)
                epoch_training_total_loss += current_batch_loss
                epoch_training_total_count += current_batch_size

                validate_counter += current_batch_size
                if validate_dataset and validate_every_n and validate_counter >= validate_every_n:
                    validate_counter = 0
                    validation_loss = calculate_validation_loss(model, validate_dataset, batch_size=batch_size)
                    all_validate_losses.append(validation_loss)

                update_counter += current_batch_size
                if update_callback and update_counter >= update_every_n:
                    update_counter = 0
                    update_callback(tepoch.n, all_train_losses, all_validate_losses)

                tepoch.update(current_batch_size)
                tepoch.set_postfix(
                    current_batch_loss=current_batch_loss,
                    validation_loss=all_validate_losses[-1] if all_validate_losses else 0)

            training_loss = epoch_training_total_loss / epoch_training_total_count
            validation_loss = all_validate_losses[-1] if all_validate_losses else 0
            tepoch.set_postfix(train_loss=training_loss, validate_loss=validation_loss)
            logger.info(f'> Finished training with training_loss={training_loss}, validation_loss={validation_loss}')

            if epoch_callback:
                epoch_callback(i_epoch, all_train_losses, all_validate_losses)

    return (all_train_losses, all_validate_losses) if validate_dataset else (all_train_losses, None)


if __name__ == '__main__':
    from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from comic_ocr.utils import get_path_project_dir

    path_output_model = get_path_project_dir('data/output/models/localization.bin')
    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    path_dataset = get_path_project_dir('data/manga_line_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(path_dataset, image_size=model.preferred_image_size)
    print(f'Loaded dataset of size {len(dataset)}...')

    #dataset.get_image(0).show()
    #dataset.get_mask_line(0).show()

    validation_dataset = dataset.subset(to_idx=2)
    training_dataset = dataset.subset(from_idx=2)
    train(model,
          train_dataset=training_dataset,
          validate_dataset=validation_dataset,
          update_callback=lambda: torch.save(model, path_output_model),
          update_every_n=10,
          epoch_count=10)
