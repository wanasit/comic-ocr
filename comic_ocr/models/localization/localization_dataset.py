"""The dataset (Pytorch) for training and evaluating localization model.
"""
from __future__ import annotations

import random
from random import Random
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from comic_ocr.dataset import annotated_manga
from comic_ocr.dataset import generated_manga
from comic_ocr.models.localization.localization_utils import image_mask_to_output_tensor
from comic_ocr.models.localization.localization_utils import image_to_input_tensor
from comic_ocr.models.localization.localization_utils import output_tensor_to_image_mask
from comic_ocr.types import Size, Rectangle, Point, SizeLike


class LocalizationDataset(torch.utils.data.Dataset):
    """A dataset for training and evaluating `LocalizationModel`.

    Each dataset entry consists of an input image and its binary marks for text characters, lines, and paragraphs (each
    of the mark is optional).
    """

    def __init__(self,
                 r: Random,
                 batch_image_size: Size,
                 images: List[Image.Image],
                 output_masks_char: Optional[List[torch.Tensor]] = None,
                 output_masks_line: Optional[List[torch.Tensor]] = None,
                 output_masks_paragraph: Optional[List[torch.Tensor]] = None,
                 output_line_locations: Optional[List[List[Rectangle]]] = None,
                 ):
        assert len(images) >= 0
        assert output_masks_char is None or len(output_masks_char) == len(images)
        assert output_masks_line is None or len(output_masks_line) == len(images)
        assert output_masks_paragraph is None or len(output_masks_paragraph) == len(images)
        assert output_line_locations is None or len(output_line_locations) == len(images)

        self.r = r
        self.batch_image_size = batch_image_size
        self.images = images
        self.output_masks_char = output_masks_char
        self.output_masks_line = output_masks_line
        self.output_masks_paragraph = output_masks_paragraph
        self.output_line_locations = output_line_locations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        output = {}
        image = self.images[idx]

        # Pad the image. This padding information also need to be applied to masks
        padded_size, paste_location = _get_padded_size_and_paste_location(image, self.batch_image_size)
        image = _pad_image(image, padded_size, paste_location)

        # Crop the image. This cropping information also need to be applied to masks
        crop_rect = _get_random_crop(self.r, image, self.batch_image_size)
        output['input'] = image_to_input_tensor(image.crop(crop_rect))

        if self.output_masks_char:
            mask = self.output_masks_char[idx]
            mask = _pad_mask(mask, padded_size, paste_location)
            output['output_mask_char'] = mask[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]

        if self.output_masks_line:
            mask = self.output_masks_line[idx]
            mask = _pad_mask(mask, padded_size, paste_location)
            output['output_mask_line'] = mask[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]

        if self.output_masks_paragraph:
            mask = self.output_masks_paragraph[idx]
            mask = _pad_mask(mask, padded_size, paste_location)
            output['output_mask_paragraph'] = mask[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]

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
        return self.output_line_locations[idx] if self.output_line_locations else None

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_idx if to_idx is not None else len(self.images)
        return LocalizationDataset(
            self.r,
            self.batch_image_size,
            self.images[from_idx:to_dix],
            self.output_masks_char[from_idx:to_dix] if self.output_masks_char else None,
            self.output_masks_line[from_idx:to_dix] if self.output_masks_line else None,
            self.output_masks_paragraph[from_idx:to_dix] if self.output_masks_paragraph else None,
            self.output_line_locations[from_idx:to_dix] if self.output_line_locations else None
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
        output_location_lines = [self.output_line_locations[i] for i in indexes] \
            if self.output_line_locations else None

        return LocalizationDataset(
            r=self.r,
            batch_image_size=self.batch_image_size,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_location_lines)

    def with_batch_image_size(self, batch_image_size: SizeLike):
        return LocalizationDataset(
            r=self.r,
            batch_image_size=Size(batch_image_size),
            images=self.images,
            output_masks_char=self.output_masks_char,
            output_masks_line=self.output_masks_line,
            output_masks_paragraph=self.output_masks_paragraph,
            output_line_locations=self.output_line_locations)

    @staticmethod
    def merge(dataset_a: LocalizationDataset, dataset_b: LocalizationDataset):
        r = Random(dataset_a.r.random() * dataset_b.r.random())
        batch_image_size = Size.of(
            min(dataset_b.batch_image_size.width, dataset_a.batch_image_size.width),
            min(dataset_b.batch_image_size.height, dataset_a.batch_image_size.height)
        )
        images = dataset_a.images + dataset_b.images
        output_masks_char = dataset_a.output_masks_char + dataset_b.output_masks_char \
            if dataset_a.output_masks_char and dataset_b.output_masks_char else None
        output_masks_line = dataset_a.output_masks_line + dataset_b.output_masks_line \
            if dataset_a.output_masks_line and dataset_b.output_masks_line else None
        output_masks_paragraph = dataset_a.output_masks_paragraph + dataset_b.output_masks_paragraph \
            if dataset_a.output_masks_paragraph and dataset_b.output_masks_paragraph else None
        output_locations_lines = dataset_a.output_line_locations + dataset_b.output_line_locations \
            if dataset_a.output_line_locations and dataset_b.output_line_locations else None

        return LocalizationDataset(
            r,
            batch_image_size,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_locations_lines)

    @staticmethod
    def load_generated_manga_dataset(
            directory,
            random_seed: str = "",
            batch_image_size: SizeLike = Size.of(500, 500)):
        images, image_texts, image_masks = generated_manga.load_dataset(directory)
        assert len(images) > 0

        output_masks_char = []
        output_masks_line = []
        output_masks_paragraph = []

        for mask in image_masks:
            output_masks_char.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_CHAR_ALPHA - 0.1))
            output_masks_line.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_LINE_ALPHA - 0.1))
            output_masks_paragraph.append(image_mask_to_output_tensor(mask, generated_manga.DEFAULT_RECT_ALPHA - 0.1))

        output_line_locations = []
        for lines in image_texts:
            output_line_locations.append([l.location for l in lines])

        return LocalizationDataset(
            r=Random(random_seed),
            batch_image_size=Size(batch_image_size),
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_line_locations)

    @staticmethod
    def load_line_annotated_manga_dataset(directory,
                                          random_seed: str = "",
                                          batch_image_size: SizeLike = Size.of(500, 500)):
        batch_image_size = Size(batch_image_size)
        original_images, annotations = annotated_manga.load_line_annotated_dataset(
            directory, include_empty_text=True)

        images = []
        output_masks_char = []
        output_masks_line = []
        output_locations_lines = []

        for image, lines in zip(original_images, annotations):
            if image.width < batch_image_size.width or image.height < batch_image_size.height:
                padded_size = Size.of(
                    max(image.width, batch_image_size.width),
                    max(image.height, batch_image_size.height))
                padded_image = Image.new('RGB', padded_size, (255, 255, 255))
                padded_image.paste(image, (0, 0))
                image = padded_image

            location_lines = []
            mask_image = torch.zeros(size=(image.height, image.width))
            mask_char_image = torch.zeros(size=(image.height, image.width))
            original_binary_image = pil_to_tensor(image.convert('L'))[0] / 255

            for l in lines:
                location_lines.append(l.location)
                mask_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = 1.0
                # This assumes the text is darker color on the whiter background
                line_image = original_binary_image[l.location.top: l.location.bottom, l.location.left:l.location.right]
                avg_pixel_value = (line_image.sum() / l.location.width / l.location.height)
                line_image = line_image < avg_pixel_value
                line_image = line_image.float()
                mask_char_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = line_image

            images.append(image)
            output_locations_lines.append(location_lines)
            output_masks_line.append(mask_image)
            output_masks_char.append(mask_char_image)

        return LocalizationDataset(
            r=Random(random_seed),
            batch_image_size=batch_image_size,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_line_locations=output_locations_lines
        )


def _get_padded_size_and_paste_location(image, target_size: Size) -> Tuple[Size, Point]:
    if image.width < target_size.width or image.height < target_size.height:
        padded_size = Size.of(
            max(image.width, target_size.width),
            max(image.height, target_size.height))
        x = (padded_size.width - image.width) // 2
        y = (padded_size.height - image.height) // 2
        return padded_size, Point.of(x, y)
    return Size.of(image.width, image.height), Point.of(0, 0)


def _pad_image(image, padded_size, paste_location, padding_color=(0, 0, 0)):
    if image.width == padded_size.width and image.height == padded_size.height:
        return image
    padded_image = Image.new('RGB', padded_size, padding_color)
    padded_image.paste(image, paste_location)
    return padded_image


def _pad_mask(mask, padded_size, paste_location):
    height, width = mask.shape
    if width == padded_size.width and height == padded_size.height:
        return mask
    padding_mask = torch.zeros(size=(padded_size.height, padded_size.width))
    padding_mask[paste_location.y:paste_location.y + height, paste_location.x: paste_location.x + width] = mask
    return padding_mask


def _get_random_crop(r: Random, image_size: Size, target_size: Size) -> Rectangle:
    assert target_size.width <= image_size.width
    assert target_size.height <= image_size.height
    x = r.randint(0, image_size.width - target_size.width)
    y = r.randint(0, image_size.height - target_size.height)

    return Rectangle.of_xywh(x, y, target_size.width, target_size.height)
