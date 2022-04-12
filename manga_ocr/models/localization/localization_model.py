import os
from typing import Callable, Tuple, List, Optional

import numpy as np
import torch
from PIL import Image

from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from manga_ocr.dataset.generated_manga import DEFAULT_CHAR_ALPHA, DEFAULT_LINE_ALPHA
from manga_ocr.models.localization import divine_rect_into_overlapping_tiles
from manga_ocr.models.transforms import AddGaussianNoise
from manga_ocr.typing import Size

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

def output_tensor_to_image_mask(tensor):
    return Image.fromarray(np.uint8(tensor.numpy()[0] * 255), 'L')


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

    def create_image_mark_lines(self, image) -> Image.Image:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            _, output = self(input_tensor)
            output = torch.sigmoid(output[0])
        return output_tensor_to_image_mask(output)


if __name__ == '__main__':
    from manga_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from manga_ocr.utils import get_path_project_dir
    from manga_ocr.utils import load_image

    path_output_model = get_path_project_dir('data/output/models/localization.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    example = load_image(get_path_project_dir('example/manga_generated/image/0000.jpg'))
    example_mask_lines = load_image(get_path_project_dir('example/manga_generated/image_mask/0000.jpg'))
    located_mask_lines = model.create_image_mark_lines(example)

    example.show()
    example_mask_lines.show()
    located_mask_lines.show()