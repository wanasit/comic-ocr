"""A module for solving localization problem = locating text inside image

"""

import math
from typing import Union, Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from comic_ocr.models.transforms import AddGaussianNoise

from comic_ocr.types import Rectangle, Size

TRANSFORM_TO_TENSOR = transforms.PILToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()
TRANSFORM_ADD_NOISE = AddGaussianNoise()


def image_to_input_tensor(
        image: Image.Image
) -> torch.Tensor:
    input = TRANSFORM_TO_TENSOR(image).float() / 255
    return input


def image_mask_to_output_tensor(
        image: Image.Image,
        threshold_min: float = 0.5,
        threshold_max: float = 1.0
) -> torch.Tensor:
    output = image_to_input_tensor(image)
    output = TRANSFORM_TO_GRAY_SCALE(output)[0]
    output = ((threshold_max >= output) & (output > threshold_min)).float()
    return output


def output_tensor_to_image_mask(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    array = tensor_or_array
    if isinstance(array, torch.Tensor):
        array = tensor_or_array.numpy()

    if len(array.shape) == 3:
        array = array.mean(0)

    return Image.fromarray(np.uint8(array * 255), 'L').convert('RGB')


def align_line_horizontal(block_a: Rectangle, block_b: Rectangle, x_min_margin=10, y_margin=2):
    if block_b.left < block_a.left:
        block_a, block_b = block_b, block_a

    x_margin = max(block_a.height, block_b.height, x_min_margin)
    return (block_a.top - y_margin) <= block_b.center.y <= (block_a.bottom + y_margin) and \
           (abs(block_a.left - block_b.left) - block_a.width) < x_margin




def divine_rect_into_overlapping_tiles(
        rect: Union[Rectangle, Size],
        tile_size: Size,
        min_overlap_x: int,
        min_overlap_y: int
):
    """Divide the rectangle or size into smaller tiles of the target size.

    This function overlaps the tile contents. This duplication is expected because we want to minimize information loss
    when train the model on divided images. The minimum overlap can be controlled by parameter.

    Args:
        rect (Rectangle, Size) the input rectangle to be divided
        tile_size (Size)
        min_overlap_x (int)
        min_overlap_y (int)

    Returns:
        An iterator of tile rectangle
    """
    tile_count_x = math.ceil((rect.width - min_overlap_x) / (tile_size.width - min_overlap_x))
    overlap_size_x = math.ceil(
        (tile_count_x * tile_size.width - rect.width) / (tile_count_x - 1)) if tile_count_x > 1 else 0

    tile_count_y = math.ceil((rect.height - min_overlap_y) / (tile_size.height - min_overlap_y))
    overlap_size_y = math.ceil(
        (tile_count_y * tile_size.height - rect.height) / (tile_count_y - 1)) if tile_count_y > 1 else 0

    offset_x = rect[0] if len(rect) == 4 else 0
    offset_y = rect[1] if len(rect) == 4 else 0

    for i in range(tile_count_y):
        for j in range(tile_count_x):
            y = offset_y + i * (tile_size.height - overlap_size_y)
            x = offset_x + j * (tile_size.width - overlap_size_x)
            yield Rectangle.of_size(tile_size, at=(x, y))
