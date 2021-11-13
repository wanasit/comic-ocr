import logging
import os
from typing import List, Optional, Callable

import torch
from PIL.Image import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from manga_ocr.dataset.generated_manga import DEFAULT_LINE_ALPHA, DEFAULT_CHAR_ALPHA
from manga_ocr.text.localization import divine_rect_into_overlapping_tiles
from manga_ocr.text.localization.localization_model import image_to_input_tensor, image_mask_to_output_tensor, \
    LocalizationModel
from manga_ocr.typing import Size
from manga_ocr.utils import load_images

logger = logging.getLogger(__name__)


class GeneratedMangaDataset(Dataset):
    def __init__(self,
                 images: List[Image],
                 image_masks: List[Image]
                 ):
        self.images = images
        self.image_masks = image_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': image_to_input_tensor(self.images[idx]),
            'mask_line': image_mask_to_output_tensor(self.image_masks[idx], DEFAULT_LINE_ALPHA - 0.1),
            'mask_char': image_mask_to_output_tensor(self.image_masks[idx], DEFAULT_CHAR_ALPHA - 0.1)
        }

    @staticmethod
    def load(directory, image_size: Size = Size.of(768, 768)):
        raw_images = load_images(os.path.join(directory, 'image/*.jpg'))
        raw_image_masks = load_images(os.path.join(directory, 'image_mask/*.jpg'))

        assert len(raw_images) == len(raw_image_masks)
        images, image_masks = GeneratedMangaDataset._split_or_pad_images_into_size(raw_images, raw_image_masks,
                                                                                   image_size)

        return GeneratedMangaDataset(images=images, image_masks=image_masks)

    @staticmethod
    def _split_or_pad_images_into_size(raw_images, raw_image_masks, image_size: Size = Size.of(768, 768)):
        output_images = []
        output_raw_image_masks = []

        tile_overlap_x = image_size.width // 4
        tile_overlap_y = image_size.width // 4
        for i in range(len(raw_images)):
            image = raw_images[i]
            image_mask = raw_image_masks[i]

            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=image_size, min_overlap_x=tile_overlap_x, min_overlap_y=tile_overlap_y):
                output_images.append(image.crop(tile))
                output_raw_image_masks.append(image_mask.crop(tile))

        return output_images, output_raw_image_masks


def train(
        model: LocalizationModel,
        train_dataset: GeneratedMangaDataset,
        validation_dataset: Optional[GeneratedMangaDataset] = None,
        epoch_count: int = 1,
        epoch_callback: Optional[Callable] = None,
        tqdm_disable=False,
        batch_size=10,
        optimizer=None
):
    optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    training_losses = []
    validation_losses = []
    logger.info(f'Training {epoch_count} epochs, on {len(train_dataset)} samples ' +
                f'({len(validation_dataset)} validation samples)' if validation_dataset else '')

    for i_epoch in range(epoch_count):
        with tqdm(total=len(train_dataset), disable=tqdm_disable) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")

            training_loss = 0.0
            for i_batch, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()

                optimizer.step()

                batch_loss = loss.item()
                batch_size = batch['image'].size(0)

                tepoch.set_postfix(training_batch_loss=batch_loss)
                tepoch.update(batch_size)
                training_loss += batch_loss * batch_size

            training_loss = training_loss / len(train_dataset)
            training_losses.append(training_loss)

            if validation_dataset:
                validation_loss = 0.0
                valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                for i_batch, batch in enumerate(valid_dataloader):
                    batch_size = batch['image'].size(0)
                    logger.debug(f'> Validating with {batch_size} samples')
                    with torch.no_grad():
                        loss = model.compute_loss(batch)
                        validation_loss += loss.item() * batch_size
                validation_losses.append(validation_loss / len(validation_dataset))
            else:
                validation_loss = None

            tepoch.set_postfix(training_loss=training_loss, validation_loss=validation_loss)
            logger.info(f'> Finished training with training_loss={training_loss}, validation_loss={validation_loss}')

            if epoch_callback:
                epoch_callback()

    return (training_losses, validation_losses) if validation_dataset else (training_losses, None)


if __name__ == '__main__':
    module_path = os.path.dirname(__file__)

    generator_input_directory = module_path + '/../../../data'
    generator_output_directory = module_path + '/../../../data/output'

    generator = GeneratedMangaDataset.load(generator_input_directory)

    input_images = load_images(module_path + "/../../out/generate/input/*.jpg")
    output_images = load_images(module_path + "/../../out/generate/output/*.jpg")

    model = ConvUnet()
    input = model.image_to_input(input_images[0]).unsqueeze(0)
    output_char, output_mask = model(input)
