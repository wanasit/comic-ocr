from __future__ import annotations

from random import Random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from comic_ocr.dataset import annotated_manga
from comic_ocr.dataset import generated_manga
from comic_ocr.models.localization.localization_model import LocalizationModel, TransformImageToTensor
from comic_ocr.models.localization.localization_utils import divine_rect_into_overlapping_tiles
from comic_ocr.models.localization.localization_utils import output_mark_image_to_tensor
from comic_ocr.models.localization.localization_utils import image_to_tensor
from comic_ocr.models.localization.localization_utils import output_tensor_to_image_mask
from comic_ocr.typing import Size, Rectangle


class LocalizationDataset(torch.utils.data.Dataset):
    """A dataset for training/testing localization model.

    Example:
        >>> model: LocalizationModel = ...
        >>> dataset = LocalizationDataset.load_line_annotated_dataset(model, ...)
        >>> dataloader = torch.utils.data.DataLoader(dataset, ...)
        >>> for i_batch, batch in enumerate(dataloader):
        >>>     loss = model.compute_loss(batch)
        >>>     ...
    """

    def __init__(self,
                 model: LocalizationModel,
                 images: List[Image.Image],
                 output_masks_char: Optional[List[Image.Image]] = None,
                 output_masks_line: Optional[List[Image.Image]] = None,
                 output_masks_paragraph: Optional[List[Image.Image]] = None,
                 output_locations_lines: Optional[List[List[Rectangle]]] = None,
                 ):
        assert len(images)
        assert output_masks_char is None or len(output_masks_char) == len(images)
        assert output_masks_line is None or len(output_masks_line) == len(images)
        assert output_masks_paragraph is None or len(output_masks_paragraph) == len(images)
        assert output_locations_lines is None or len(output_locations_lines) == len(images)

        self.model = model
        self.images = images
        self.output_masks_char = output_masks_char
        self.output_masks_line = output_masks_line
        self.output_masks_paragraph = output_masks_paragraph
        self.output_locations_lines = output_locations_lines

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        output = {
            'input': self.model.image_to_tensor(self.images[idx]),
        }
        if self.output_masks_char:
            output['output_mask_char'] = self.model.output_mask_to_tensor(self.output_masks_char[idx])
        if self.output_masks_line:
            output['output_mask_line'] = self.model.output_mask_to_tensor(self.output_masks_line[idx])
        if self.output_masks_paragraph:
            output['output_mask_paragraph'] = self.model.output_mask_to_tensor(self.output_masks_paragraph[idx])
        return output

    def get_image_size(self) -> Size:
        return Size(self.images[0].size)

    def get_image(self, idx) -> Image.Image:
        return self.images[idx]

    def get_mask_char(self, idx) -> Image.Image:
        return self.output_masks_char[idx] if self.output_masks_char else None

    def get_mask_line(self, idx) -> Image.Image:
        return self.output_masks_line[idx] if self.output_masks_line else None

    def get_mask_paragraph(self, idx) -> Image.Image:
        return self.output_masks_paragraph[idx] if self.output_masks_paragraph else None

    def get_line_locations(self, idx) -> List[Rectangle]:
        return self.output_locations_lines[idx] if self.output_locations_lines else None

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_idx if to_idx is not None else len(self.images)
        return LocalizationDataset(
            model=self.model,
            images=self.images[from_idx:to_dix],
            output_masks_char=self.output_masks_char[from_idx:to_dix] if self.output_masks_char else None,
            output_masks_line=self.output_masks_line[from_idx:to_dix] if self.output_masks_line else None,
            output_masks_paragraph=self.output_masks_paragraph[
                                   from_idx:to_dix] if self.output_masks_paragraph else None,
            output_locations_lines=self.output_locations_lines[from_idx:to_dix] if self.output_locations_lines else None
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
            model=self.model,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_locations_lines=output_location_lines)

    @staticmethod
    def merge(dataset_a: LocalizationDataset, dataset_b: LocalizationDataset):
        assert dataset_a.model == dataset_b.model, \
            'Can only merge dataset for the same model. TODO: add this later'

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
            model=dataset_a.model,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph,
            output_locations_lines=output_locations_lines)

    @staticmethod
    def load_generated_dataset(model: LocalizationModel, directory):
        loaded_images, _, loaded_image_masks = generated_manga.load_dataset(directory)
        assert len(loaded_images) > 0

        images = []
        output_masks_char = []
        output_masks_line = []
        output_masks_paragraph = []

        target_image_size = model.preferred_image_size
        for image, image_mask in zip(loaded_images, loaded_image_masks):
            assert image.size == image_mask.size
            image_mask = image_mask.convert('L')

            # If the image is too small, pad the image with white background
            if image.width < target_image_size.width or image.height < target_image_size.height:
                padded_size = Size.of(
                    max(image.width, target_image_size.width),
                    max(image.height, target_image_size.height))
                padded_image = Image.new('RGB', padded_size, (255, 255, 255))
                padded_image.paste(image, (0, 0))
                image = padded_image
                padded_image_mask = Image.new('RGB', padded_size, (0, 0, 0))
                padded_image_mask.paste(image, (0, 0))
                image_mask = padded_image_mask

            tile_overlap_x = target_image_size.width // 4
            tile_overlap_y = target_image_size.width // 4
            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=target_image_size, min_overlap_x=tile_overlap_x,
                    min_overlap_y=tile_overlap_y):
                images.append(image.crop(tile))

                mask_char_tensor = output_mark_image_to_tensor(image_mask.crop(tile),
                                                               generated_manga.DEFAULT_CHAR_ALPHA - 0.1)
                mask_line_tensor = output_mark_image_to_tensor(image_mask.crop(tile),
                                                               generated_manga.DEFAULT_LINE_ALPHA - 0.1)
                mask_paragraph_tensor = output_mark_image_to_tensor(image_mask.crop(tile),
                                                                    generated_manga.DEFAULT_RECT_ALPHA - 0.1)
                output_masks_char.append(Image.fromarray(np.uint8(mask_char_tensor.numpy() * 255), 'L'))
                output_masks_line.append(Image.fromarray(np.uint8(mask_line_tensor.numpy() * 255), 'L'))
                output_masks_paragraph.append(Image.fromarray(np.uint8(mask_paragraph_tensor.numpy() * 255), 'L'))

        return LocalizationDataset(
            model=model,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_masks_paragraph=output_masks_paragraph
        )

    @staticmethod
    def load_line_annotated_dataset(model: LocalizationModel, directory):
        original_images, annotations = annotated_manga.load_line_annotated_dataset(
            directory, include_empty_text=True)

        images = []
        output_masks_char = []
        output_masks_line = []
        output_locations_lines = []

        target_image_size = model.preferred_image_size
        for image, lines in zip(original_images, annotations):

            # If the image is too small, pad the image with white background
            if image.width < target_image_size.width or image.height < target_image_size.height:
                padded_size = Size.of(
                    max(image.width, target_image_size.width),
                    max(image.height, target_image_size.height))
                padded_image = Image.new('RGB', padded_size, (255, 255, 255))
                padded_image.paste(image, (0, 0))
                image = padded_image

            # Create output masks (similar size as the image), and populate the masks via annotated lines.
            # For each annotated line's location,
            # - In output mask for line, fill the location with 1's
            # - In output mask for characters, fill the location with (1 - original black/white pixel value)'s
            # NOTE: We assume the characters are darker (0's) on the white background (1's).
            # TODO: Apply normalization. Remove noise or blurry background.
            mask_line_tensor = torch.zeros(size=(image.height, image.width))
            mask_char_tensor = torch.zeros(size=(image.height, image.width))
            black_white_original = pil_to_tensor(image.convert('L'))[0] / 255
            for l in lines:
                mask_line_tensor[l.location.top: l.location.bottom, l.location.left:l.location.right] = 1.0
                mask_char_tensor[l.location.top: l.location.bottom, l.location.left:l.location.right] = \
                    1 - black_white_original[l.location.top: l.location.bottom, l.location.left:l.location.right]
            mask_line_image = output_tensor_to_image_mask(mask_line_tensor)
            mask_char_image = output_tensor_to_image_mask(mask_char_tensor)

            # Cut the image into tiles of the target size (if it's too large)
            tile_overlap_x = target_image_size.width // 4
            tile_overlap_y = target_image_size.height // 4
            for tile in divine_rect_into_overlapping_tiles(
                    Size(image.size), tile_size=target_image_size, min_overlap_x=tile_overlap_x,
                    min_overlap_y=tile_overlap_y):
                images.append(image.crop(tile))
                output_masks_line.append(mask_line_image.crop(tile))
                output_masks_char.append(mask_char_image.crop(tile))

                location_lines = []
                for l in lines:
                    new_location = Rectangle.intersect_bounding_rect((l.location, tile))
                    if new_location:
                        new_location = new_location.move(-tile.left, -tile.top)
                        location_lines.append(new_location)

                output_locations_lines.append(location_lines)

        return LocalizationDataset(
            model=model,
            images=images,
            output_masks_char=output_masks_char,
            output_masks_line=output_masks_line,
            output_locations_lines=output_locations_lines
        )


if __name__ == '__main__':
    from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from comic_ocr.utils.files import get_path_project_dir
    from torch.utils.data import DataLoader

    model = ConvUnet()

    path = get_path_project_dir('example/manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_dataset(model, path)
    dataset_loader = DataLoader(dataset)
    loss = model.compute_loss(next(iter(dataset_loader)))

    path = get_path_project_dir('example/manga_generated')
    dataset = LocalizationDataset.load_generated_dataset(model, path)
    dataset_loader = DataLoader(dataset)
    loss = model.compute_loss(next(iter(dataset_loader)))
