from __future__ import annotations

from random import Random
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from comic_ocr.dataset import annotated_manga
from comic_ocr.dataset import generated_manga
from comic_ocr.models.localization.localization_utils import divine_rect_into_overlapping_tiles
from comic_ocr.models.localization.localization_utils import image_mask_to_output_tensor
from comic_ocr.models.localization.localization_utils import image_to_input_tensor
from comic_ocr.models.localization.localization_utils import output_tensor_to_image_mask
from comic_ocr.typing import Size, Rectangle


class LocalizationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 images: List[Image.Image],
                 output_masks_char: Optional[List[torch.Tensor]] = None,
                 output_masks_line: Optional[List[torch.Tensor]] = None,
                 output_masks_paragraph: Optional[List[torch.Tensor]] = None,
                 output_locations_lines: Optional[List[List[Rectangle]]] = None,
                 ):
        assert len(images)
        assert output_masks_char is None or len(output_masks_char) == len(images)
        assert output_masks_line is None or len(output_masks_line) == len(images)
        assert output_masks_paragraph is None or len(output_masks_paragraph) == len(images)
        assert output_locations_lines is None or len(output_locations_lines) == len(images)

        self.images = images
        self.output_masks_char = output_masks_char
        self.output_masks_line = output_masks_line
        self.output_masks_paragraph = output_masks_paragraph
        self.output_locations_lines = output_locations_lines

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

    def get_image_size(self) -> Size:
        return Size(self.images[0].size)

    def get_image(self, idx) -> Image.Image:
        return self.images[idx]

    def get_mask_char(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self.output_masks_char[idx]) if self.output_masks_char else None

    def get_mask_line(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self.output_masks_line[idx]) if self.output_masks_line else None

    def get_mask_paragraph(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self.output_masks_paragraph[idx]) if self.output_masks_paragraph else None

    def get_line_locations(self, idx) -> List[Rectangle]:
        return self.output_locations_lines[idx] if self.output_locations_lines else None

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_idx if to_idx is not None else len(self.images)
        return LocalizationDataset(
            self.images[from_idx:to_dix],
            self.output_masks_char[from_idx:to_dix] if self.output_masks_char else None,
            self.output_masks_line[from_idx:to_dix] if self.output_masks_line else None,
            self.output_masks_paragraph[from_idx:to_dix] if self.output_masks_paragraph else None,
            self.output_locations_lines[from_idx:to_dix] if self.output_locations_lines else None
        )

    def shuffle(self, random_seed: any = '') -> LocalizationDataset:
        indexes = list(range(len(self.images)))
        random = Random(random_seed)
        random.shuffle(indexes)

        images = [self.images[i] for i in indexes]
        output_masks_char = [self.output_masks_char[i] for i in indexes] \
            if self.output_masks_char else None
        output_masks_line = [self.output_masks_line[i] for i in indexes] \
            if self.output_masks_line else None
        output_masks_paragraph = [self.output_masks_paragraph[i] for i in indexes] \
            if self.output_masks_paragraph else None
        output_location_lines = [self.output_locations_lines[i] for i in indexes] \
            if self.output_locations_lines else None

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_locations_lines=output_location_lines)

    @staticmethod
    def merge(dataset_a: LocalizationDataset, dataset_b: LocalizationDataset):
        assert dataset_a.get_image_size() == dataset_b.get_image_size(), \
            'Can only merge dataset with the same image size. TODO: add this later'

        images = dataset_a.images + dataset_b.images
        output_masks_char = dataset_a.output_masks_char + dataset_b.output_masks_char \
            if dataset_a.output_masks_char and dataset_b.output_masks_char else None
        output_masks_line = dataset_a.output_masks_line + dataset_b.output_masks_line \
            if dataset_a.output_masks_line and dataset_b.output_masks_line else None
        output_masks_paragraph = dataset_a.output_masks_paragraph + dataset_b.output_masks_paragraph \
            if dataset_a.output_masks_paragraph and dataset_b.output_masks_paragraph else None
        output_locations_lines = dataset_a.output_locations_lines + dataset_b.output_locations_lines \
            if dataset_a.output_locations_lines and dataset_b.output_locations_lines else None

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_locations_lines=output_locations_lines)

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
        output_masks_char = []
        output_masks_line = []
        output_locations_lines = []

        for image, lines in zip(original_images, annotations):
            mask_image = torch.zeros(size=(image.height, image.width))
            mask_char_image = torch.zeros(size=(image.height, image.width))
            original_image = pil_to_tensor(image.convert('L'))[0] / 255
            for l in lines:
                mask_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = 1.0
                mask_char_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = \
                    1 - original_image[l.location.top: l.location.bottom, l.location.left:l.location.right]

            tile_overlap_x = image.width // 4
            tile_overlap_y = image.width // 4
            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=image_size, min_overlap_x=tile_overlap_x, min_overlap_y=tile_overlap_y):
                images.append(image.crop(tile))
                output_masks_line.append(mask_image[tile.top:tile.bottom, tile.left:tile.right])
                output_masks_char.append(mask_char_image[tile.top:tile.bottom, tile.left:tile.right])

                location_lines = []
                for l in lines:
                    new_location = Rectangle.intersect_bounding_rect((l.location, tile))
                    if new_location:
                        new_location = new_location.move(-tile.left, -tile.top)
                        location_lines.append(new_location)

                output_locations_lines.append(location_lines)

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_locations_lines=output_locations_lines
        )

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
