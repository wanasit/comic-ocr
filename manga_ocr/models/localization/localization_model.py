import os
from typing import Callable, Tuple, List, Optional

import numpy as np
import torch
from PIL.Image import Image

from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from manga_ocr.dataset.generated_manga import DEFAULT_CHAR_ALPHA, DEFAULT_LINE_ALPHA
from manga_ocr.models.localization import divine_rect_into_overlapping_tiles
from manga_ocr.models.transforms import AddGaussianNoise
from manga_ocr.typing import Size
from manga_ocr.utils import load_images

TRANSFORM_TO_TENSOR = transforms.ToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()
TRANSFORM_ADD_NOISE = AddGaussianNoise()


def image_to_input_tensor(image):
    input = TRANSFORM_TO_TENSOR(image)
    return input


def image_mask_to_output_tensor(image, threshold: float = 0.5):
    output = TRANSFORM_TO_TENSOR(image)
    output = TRANSFORM_TO_GRAY_SCALE(output)
    output = (output > threshold).float()
    return output


class LocalizationModel(nn.Module):
    """
    (-1, input_size[0], input_size[1])
    """

    __call__: Callable[..., Tuple[Tensor, Tensor]]

    def __init__(self):
        super().__init__()

    @property
    def image_size(self) -> Size:
        return Size.of(750, 750)

    def compute_loss(self, dataset_batch, criterion=nn.BCEWithLogitsLoss(), char_pred_weight=0.5,
                     line_pre_weight=0.5) -> Tensor:
        input = dataset_batch['image']
        mask_line = dataset_batch['mask_line'].float()
        mask_character = dataset_batch['mask_char'].float()

        output_char, output_line = self(input)
        loss = criterion(output_char, mask_character) * char_pred_weight + \
               criterion(output_line, mask_line) * line_pre_weight

        return loss
