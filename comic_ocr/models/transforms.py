from typing import Callable, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

ImageToTensorTransformFunc = Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor]
TensorTransformFunc = Callable[[torch.Tensor], torch.Tensor]

# Re-export torchvision.transforms
Compose = transforms.Compose
ColorJitter = transforms.ColorJitter

TRANSFORM_TO_TENSOR = transforms.PILToTensor()



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.15):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor, min=0, max=1.0)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def image_to_tensor(
        image: Union[Image.Image, torch.Tensor]
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = TRANSFORM_TO_TENSOR(image).float() / 255

    assert image.shape[0] == 3
    return image
