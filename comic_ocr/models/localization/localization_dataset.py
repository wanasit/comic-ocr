"""The dataset (Pytorch) for training and evaluating localization model.
"""
from __future__ import annotations

import random
from random import Random
from typing import List, Optional, Tuple, Sequence

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
    """A torch dataset for evaluating a localization model.

    Each dataset entry consists of an input image and its binary marks for text characters, lines, and paragraphs (each
    of the mark is optional). Because each image may have different size, the dataset should be load with batch size 1.
    training.

    For training a localization model, consider using `LocalizationDatasetWithAugmentation` instead.
    """

    def __init__(self,
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

        self._images = images
        self._output_masks_char = output_masks_char
        self._output_masks_line = output_masks_line
        self._output_masks_paragraph = output_masks_paragraph
        self._output_line_locations = output_line_locations

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        output = dict()
        output['input'] = image_to_input_tensor(self._images[idx])
        if self._output_masks_char:
            output['output_mask_char'] = self._output_masks_char[idx]
        if self._output_masks_line:
            output['output_mask_line'] = self._output_masks_line[idx]
        if self._output_masks_paragraph:
            output['output_mask_paragraph'] = self._output_masks_paragraph[idx]
        return output

    def loader(self, **kwargs):
        kwargs.pop('batch_size', None)
        kwargs.pop('num_workers', None)
        return torch.utils.data.DataLoader(self, batch_size=1, num_workers=0, **kwargs)

    def get_image(self, idx) -> Image.Image:
        return self._images[idx]

    def get_mask_char(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self._output_masks_char[idx]) if self._output_masks_char else None

    def get_mask_line(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self._output_masks_line[idx]) if self._output_masks_line else None

    def get_mask_paragraph(self, idx) -> Image.Image:
        return output_tensor_to_image_mask(self._output_masks_paragraph[idx]) if self._output_masks_paragraph else None

    def get_line_locations(self, idx) -> List[Rectangle]:
        return self._output_line_locations[idx] if self._output_line_locations else None

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None) -> LocalizationDataset:
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_idx if to_idx is not None else len(self._images)
        return LocalizationDataset(
            self._images[from_idx:to_dix],
            self._output_masks_char[from_idx:to_dix] if self._output_masks_char else None,
            self._output_masks_line[from_idx:to_dix] if self._output_masks_line else None,
            self._output_masks_paragraph[from_idx:to_dix] if self._output_masks_paragraph else None,
            self._output_line_locations[from_idx:to_dix] if self._output_line_locations else None
        )

    def shuffle(self, random_seed: any = '') -> LocalizationDataset:
        indexes = list(range(len(self._images)))
        r = Random(random_seed)
        r.shuffle(indexes)

        images = [self._images[i] for i in indexes]
        output_masks_char = [self._output_masks_char[i] for i in indexes] \
            if self._output_masks_char else None
        output_masks_line = [self._output_masks_line[i] for i in indexes] \
            if self._output_masks_line else None
        output_masks_paragraph = [self._output_masks_paragraph[i] for i in indexes] \
            if self._output_masks_paragraph else None
        output_location_lines = [self._output_line_locations[i] for i in indexes] \
            if self._output_line_locations else None

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_location_lines)

    @staticmethod
    def merge(dataset_a: LocalizationDataset, dataset_b: LocalizationDataset):
        images = dataset_a._images + dataset_b._images
        output_masks_char = dataset_a._output_masks_char + dataset_b._output_masks_char \
            if dataset_a._output_masks_char and dataset_b._output_masks_char else None
        output_masks_line = dataset_a._output_masks_line + dataset_b._output_masks_line \
            if dataset_a._output_masks_line and dataset_b._output_masks_line else None
        output_masks_paragraph = dataset_a._output_masks_paragraph + dataset_b._output_masks_paragraph \
            if dataset_a._output_masks_paragraph and dataset_b._output_masks_paragraph else None
        output_locations_lines = dataset_a._output_line_locations + dataset_b._output_line_locations \
            if dataset_a._output_line_locations and dataset_b._output_line_locations else None

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_locations_lines)

    @staticmethod
    def load_generated_manga_dataset(
            directory,
            min_image_size: SizeLike = Size.of(500, 500)):
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
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_line_locations=output_line_locations)

    @staticmethod
    def load_line_annotated_manga_dataset(directory,
                                          min_image_size: SizeLike = Size.of(500, 500)):
        min_image_size = Size(min_image_size)
        original_images, annotations = annotated_manga.load_line_annotated_dataset(
            directory, include_empty_text=True)

        images = []
        output_masks_char = []
        output_masks_line = []
        output_locations_lines = []

        for image, lines in zip(original_images, annotations):
            if image.width < min_image_size.width or image.height < min_image_size.height:
                padded_size = Size.of(
                    max(image.width, min_image_size.width),
                    max(image.height, min_image_size.height))
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
                line_image: torch.Tensor = original_binary_image[l.location.top: l.location.bottom,
                                           l.location.left:l.location.right]
                avg_pixel_value = (line_image.sum() / l.location.width / l.location.height)
                line_image = line_image < avg_pixel_value
                line_image = line_image.float()
                mask_char_image[l.location.top: l.location.bottom, l.location.left:l.location.right] = line_image

            images.append(image)
            output_locations_lines.append(location_lines)
            output_masks_line.append(mask_image)
            output_masks_char.append(mask_char_image)

        return LocalizationDataset(
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_line_locations=output_locations_lines
        )


class LocalizationDatasetWithAugmentation(LocalizationDataset):
    """A torch dataset for training a localization model.

    This dataset contains images and masks (similar to normal localization dataset), but it randomly crops the images
    and masks to the given `batch_image_size`.
    """

    def __init__(self,
                 r: Random,
                 batch_image_size: SizeLike,
                 choices_padding_width: Sequence[int],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r = r
        self._batch_image_size: Size = Size(batch_image_size)
        self._choices_padding_width = tuple(choices_padding_width)
        self._image_tensors = [image_to_input_tensor(i) for i in self._images]

    def __getitem__(self, idx):
        output = {}

        # Crop the image.
        padding_width = self._r.choice(self._choices_padding_width)
        crop_rect = _get_random_crop(self._r, self._images[idx], self._batch_image_size)
        crop_rect = crop_rect.expand(-padding_width)
        output['input'] = _apply_crop_to_image(self._image_tensors[idx], crop_rect, padded_size=self._batch_image_size)

        # Apply the same cropping to masks
        if self._output_masks_char:
            output['output_mask_char'] = _apply_crop_to_mask(
                self._output_masks_char[idx], crop_rect, padded_size=self._batch_image_size)
        if self._output_masks_line:
            output['output_mask_line'] = _apply_crop_to_mask(
                self._output_masks_line[idx], crop_rect, padded_size=self._batch_image_size)
        if self._output_masks_paragraph:
            output['output_mask_paragraph'] = _apply_crop_to_mask(
                self._output_masks_paragraph[idx], crop_rect, padded_size=self._batch_image_size)
        return output

    def loader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def subset(self, from_idx: Optional[int] = None,
               to_idx: Optional[int] = None) -> LocalizationDatasetWithAugmentation:
        subset = super().subset(from_idx, to_idx)
        return LocalizationDatasetWithAugmentation.of_dataset(
            subset, batch_image_size=self._batch_image_size, choices_padding_width=self._choices_padding_width)

    def shuffle(self, random_seed: any = '') -> LocalizationDatasetWithAugmentation:
        shuffled = super().shuffle(random_seed)
        return LocalizationDatasetWithAugmentation.of_dataset(
            shuffled, batch_image_size=self._batch_image_size, choices_padding_width=self._choices_padding_width)

    def with_batch_image_size(self, batch_image_size: SizeLike) -> LocalizationDatasetWithAugmentation:
        return LocalizationDatasetWithAugmentation.of_dataset(
            self, batch_image_size=batch_image_size, choices_padding_width=self._choices_padding_width)

    def with_choices_padding_width(self, choices_padding_width: Sequence[int]) -> LocalizationDatasetWithAugmentation:
        return LocalizationDatasetWithAugmentation.of_dataset(
            self, batch_image_size=self._batch_image_size, choices_padding_width=choices_padding_width)

    def without_augmentation(self) -> LocalizationDataset:
        return LocalizationDataset(images=self._images,
                                   output_masks_char=self._output_masks_char,
                                   output_masks_line=self._output_masks_line,
                                   output_line_locations=self._output_line_locations)

    @staticmethod
    def of_dataset(dataset: LocalizationDataset,
                   r: Optional[Random] = None,
                   batch_image_size: Optional[SizeLike] = None,
                   choices_padding_width: Optional[Sequence[int]] = None) -> LocalizationDatasetWithAugmentation:
        r = r if r is not None else Random()
        batch_image_size = Size(batch_image_size) if batch_image_size is not None else Size.of(500, 500)
        choices_padding_width = choices_padding_width if choices_padding_width is not None else [0]
        return LocalizationDatasetWithAugmentation(
            r=r,
            batch_image_size=batch_image_size,
            choices_padding_width=choices_padding_width,
            images=dataset._images,
            output_masks_char=dataset._output_masks_char,
            output_masks_line=dataset._output_masks_line,
            output_masks_paragraph=dataset._output_masks_paragraph,
            output_line_locations=dataset._output_line_locations
        )

    @staticmethod
    def merge(
            dataset_a: LocalizationDatasetWithAugmentation,
            dataset_b: LocalizationDatasetWithAugmentation
    ) -> LocalizationDatasetWithAugmentation:
        r = Random(dataset_a._r.random() * dataset_b._r.random())
        batch_image_size = Size.of(
            min(dataset_b._batch_image_size.width, dataset_a._batch_image_size.width),
            min(dataset_b._batch_image_size.height, dataset_a._batch_image_size.height)
        )
        choices_padding_width = []
        choices_padding_width += dataset_a._choices_padding_width
        choices_padding_width += dataset_b._choices_padding_width
        merged_dataset = LocalizationDataset.merge(dataset_a, dataset_b)
        return LocalizationDatasetWithAugmentation.of_dataset(
            merged_dataset, r=r, batch_image_size=batch_image_size, choices_padding_width=choices_padding_width)

    @staticmethod
    def load_generated_manga_dataset(
            directory,
            random_seed: str = "",
            batch_image_size: SizeLike = Size.of(500, 500),
            **kwargs
    ) -> LocalizationDatasetWithAugmentation:
        dataset = LocalizationDataset.load_generated_manga_dataset(directory, min_image_size=batch_image_size)
        return LocalizationDatasetWithAugmentation.of_dataset(
            dataset, r=Random(random_seed), batch_image_size=Size(batch_image_size), **kwargs)

    @staticmethod
    def load_line_annotated_manga_dataset(
            directory,
            random_seed: str = "",
            batch_image_size: SizeLike = Size.of(500, 500),
            **kwargs
    ) -> LocalizationDatasetWithAugmentation:
        dataset = LocalizationDataset.load_line_annotated_manga_dataset(directory, min_image_size=batch_image_size)
        return LocalizationDatasetWithAugmentation.of_dataset(
            dataset, r=Random(random_seed), batch_image_size=batch_image_size, **kwargs)


def _apply_crop_to_mask(mask: torch.Tensor, crop_rect: Rectangle, padded_size: Size) -> torch.Tensor:
    assert len(mask.shape) == 2
    if crop_rect.width == padded_size.width and crop_rect.height == padded_size.height:
        return mask[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]

    padding_image = torch.zeros(size=(padded_size.height, padded_size.width))
    y_offset = (padded_size.height - crop_rect.height) // 2
    x_offset = (padded_size.width - crop_rect.width) // 2
    padding_image[y_offset:y_offset + crop_rect.height, x_offset:x_offset + crop_rect.width] = \
        mask[crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]
    return padding_image


def _apply_crop_to_image(image: torch.Tensor, crop_rect: Rectangle, padded_size: Size) -> torch.Tensor:
    assert len(image.shape) == 3
    if crop_rect.width == padded_size.width and crop_rect.height == padded_size.height:
        return image[:, crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]

    padding_image = torch.zeros(size=(image.shape[0], padded_size.height, padded_size.width))
    y_offset = (padded_size.height - crop_rect.height) // 2
    x_offset = (padded_size.width - crop_rect.width) // 2
    padding_image[:, y_offset:y_offset + crop_rect.height, x_offset:x_offset + crop_rect.width] = \
        image[:, crop_rect.top:crop_rect.bottom, crop_rect.left:crop_rect.right]
    return padding_image


def _get_random_crop(r: Random, image_size: Size | Image, target_size: Size) -> Rectangle:
    assert target_size.width <= image_size.width
    assert target_size.height <= image_size.height
    x = r.randint(0, image_size.width - target_size.width)
    y = r.randint(0, image_size.height - target_size.height)

    return Rectangle.of_xywh(x, y, target_size.width, target_size.height)
