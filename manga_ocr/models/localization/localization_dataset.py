from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from manga_ocr.dataset import annotated_manga
from manga_ocr.dataset import generated_manga
from manga_ocr.models.localization.localization_utils import divine_rect_into_overlapping_tiles, \
    output_tensor_to_image_mask
from manga_ocr.models.localization.localization_utils import image_mask_to_output_tensor, image_to_input_tensor
from manga_ocr.typing import Size


class LocalizationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 images: List[Image.Image],
                 output_masks_char: Optional[List[torch.Tensor]] = None,
                 output_masks_line: Optional[List[torch.Tensor]] = None,
                 output_masks_paragraph: Optional[List[torch.Tensor]] = None,
                 ):
        assert len(images)
        assert output_masks_char is None or len(output_masks_char) == len(images)
        assert output_masks_line is None or len(output_masks_line) == len(images)
        assert output_masks_paragraph is None or len(output_masks_paragraph) == len(images)

        self.images = images
        self.output_masks_char = output_masks_char
        self.output_masks_line = output_masks_line
        self.output_masks_paragraph = output_masks_paragraph

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        output = {
            'input': image_to_input_tensor(self.images[idx]),
        }

        if self.output_masks_char:
            output['output_mask_char'] = self.output_masks_char[idx]

        if self.output_masks_line:
            output['output_mask_line'] = self.output_masks_line[idx]

        if self.output_masks_paragraph:
            output['output_mask_paragraph'] = self.output_masks_paragraph[idx]

        return output

    def get_image(self, idx):
        return self.images[idx]

    def get_mask_char(self, idx):
        return output_tensor_to_image_mask(self.output_masks_char[idx]) if self.output_masks_char else None

    def get_mask_line(self, idx):
        return output_tensor_to_image_mask(self.output_masks_line[idx]) if self.output_masks_line else None

    def get_mask_paragraph(self, idx):
        return output_tensor_to_image_mask(self.output_masks_paragraph[idx]) if self.output_masks_paragraph else None


    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_idx if to_idx is not None else len(self.images)
        return LocalizationDataset(
            self.images[from_idx:to_dix],
            self.output_masks_char[from_idx:to_dix] if self.output_masks_char else None,
            self.output_masks_line[from_idx:to_dix] if self.output_masks_line else None,
            self.output_masks_paragraph[from_idx:to_dix] if self.output_masks_paragraph else None
        )

    @staticmethod
    def load_generated_manga_dataset(directory, image_size: Size = Size.of(500, 500)):
        images, _, image_masks = generated_manga.load_dataset(directory)
        assert len(images) > 0

        images, image_masks = LocalizationDataset._split_or_pad_images_into_size(images, image_masks, image_size)

        output_masks_char = []
        output_masks_line = []
        output_masks_paragraph = []

        for mask in image_masks:
            output_masks_char.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_CHAR_ALPHA - 0.1))
            output_masks_line.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_LINE_ALPHA - 0.1))
            output_masks_paragraph.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_RECT_ALPHA - 0.1))

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph)

    @staticmethod
    def load_line_annotated_manga_dataset(directory, image_size: Size = Size.of(500, 500)):
        original_images, annotations = annotated_manga.load_line_annotated_dataset(
            directory, include_empty_text=True)

        images = []
        output_masks_line = []

        for image, lines in zip(original_images, annotations):
            mask_image = torch.zeros(size=(image.height, image.width))
            for l in lines:
                mask_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = 1.0

            tile_overlap_x = image.width // 4
            tile_overlap_y = image.width // 4
            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=image_size, min_overlap_x=tile_overlap_x, min_overlap_y=tile_overlap_y):

                images.append(image.crop(tile))
                output_masks_line.append(mask_image[tile.top:tile.bottom, tile.left:tile.right])

        return LocalizationDataset(images=images, output_masks_line=output_masks_line)

    @staticmethod
    def _split_or_pad_images_into_size(
            original_images,
            original_image_masks,
            output_image_size: Size = Size.of(500, 500)):
        output_images = []
        output_raw_image_masks = []

        tile_overlap_x = output_image_size.width // 4
        tile_overlap_y = output_image_size.width // 4

        for image, image_mask in zip(original_images, original_image_masks):

            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=output_image_size, min_overlap_x=tile_overlap_x,
                    min_overlap_y=tile_overlap_y):
                output_images.append(image.crop(tile))
                output_raw_image_masks.append(image_mask.crop(tile))

        return output_images, output_raw_image_masks
