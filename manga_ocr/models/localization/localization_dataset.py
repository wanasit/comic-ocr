import logging
import os
from typing import List, Optional, Callable

import torch
from PIL.Image import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from manga_ocr.dataset.generated_manga import DEFAULT_LINE_ALPHA, DEFAULT_CHAR_ALPHA
from manga_ocr.models.localization import divine_rect_into_overlapping_tiles
from manga_ocr.models.localization.localization_model import image_mask_to_output_tensor, image_to_input_tensor
from manga_ocr.typing import Size
from manga_ocr.utils import load_images


class LocalizationDataset(Dataset):
    def __init__(self, images: List[Image], image_masks: List[Image]):
        assert len(images) == len(image_masks)
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

    def subset(self, from_idx: Optional[int] = None, to_dix: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_dix if to_dix is not None else len(self.images)
        return LocalizationDataset(self.images[from_idx:to_dix], self.image_masks[from_idx:to_dix])

    @staticmethod
    def load_generated_manga_dataset(directory, image_size: Size = Size.of(500, 500)):
        raw_images = load_images(os.path.join(directory, 'image/*.jpg'))
        raw_image_masks = load_images(os.path.join(directory, 'image_mask/*.jpg'))

        assert len(raw_images) > 0
        assert len(raw_images) == len(raw_image_masks)
        images, image_masks = LocalizationDataset._split_or_pad_images_into_size(raw_images, raw_image_masks,
                                                                                 image_size)
        return LocalizationDataset(images=images, image_masks=image_masks)

    @staticmethod
    def _split_or_pad_images_into_size(raw_images, raw_image_masks, image_size: Size = Size.of(500, 500)):
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
