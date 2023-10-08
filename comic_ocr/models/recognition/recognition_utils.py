import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from comic_ocr.models.transforms import AddGaussianNoise

from comic_ocr.types import Rectangle, Size

TRANSFORM_TO_TENSOR = transforms.PILToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()


