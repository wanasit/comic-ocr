"""The module provides abstract/shared functionalities for ML model for localization
"""

import os
from typing import Callable, Tuple, List, Union

import torch
from PIL import Image
from torch import nn, Tensor
from comic_ocr.models import transforms
from comic_ocr.models.localization import localization_open_cv as cv
from comic_ocr.models.localization import localization_utils as utils
from comic_ocr.typing import Size, Rectangle, SizeLike

DEFAULT_LOSS_CRITERION_CHAR = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2]))
DEFAULT_LOSS_CRITERION_LINE = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
TRANSFORM_TO_TENSOR = transforms.PILToTensor()
TRANSFORM_ADD_NOISE = transforms.AddGaussianNoise()

TransformImageToTensor = Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor]


class LocalizationModel(nn.Module):
    """An abstract for localization Module for text localization

    The ML model (PyTorch) classifies each input image's pixel if it's part of a character, or a line boundary. The
    model output those probabilities as 'mask images' in training dataset format. Namely, given the input source image,
    the model is trained to output three types of mask images.

    The module also applies other non-ML image-processing techniques on top of the mask images to output the locations.

    Example:
        >>> localization_model: LocalizationModel = ....
        >>> loss = localization_model.compute_loss(batch) # Training
        >>> paragraph_rectangles = localization_model.locate_lines(image) # Evaluation

    Shape:
        - Input input_image (Tensor [C x H x W])
        - Output Tuple(output_mask_character, output_mask_line)
            - output_mask_character (Tensor [H x W]) the unnormalized probability that the pixel is character/text
            - output_mask_line (Tensor [H x W]) the unnormalized probability that the pixel is inside line rect boundary
    """
    __call__: Callable[..., Tuple[Tensor, Tensor]]

    def __init__(self, image_size: SizeLike = Size.of(500, 500), training_mode=True):
        super().__init__()
        self.image_size = Size(image_size)
        self.training_mode = training_mode

    @property
    def preferred_image_size(self) -> Size:
        return self.image_size

    def train(self, mode: bool = True) -> nn.Module:
        self.training_mode = mode
        return super().train(mode)

    def compute_loss(
            self,
            dataset_batch,
            loss_criterion_for_char=DEFAULT_LOSS_CRITERION_CHAR,
            loss_criterion_for_line=DEFAULT_LOSS_CRITERION_LINE
    ) -> Tensor:
        """Computes loss for a given LocalizationDataset's batch
        """
        input_tensor = dataset_batch['input']
        output_tensor_char, output_tensor_line = self(input_tensor)

        loss = torch.zeros(1)
        if 'output_mask_char' in dataset_batch:
            expected_output = dataset_batch['output_mask_char'].float()
            loss += loss_criterion_for_char(output_tensor_char, expected_output)

        if 'output_mask_line' in dataset_batch:
            expected_output = dataset_batch['output_mask_line'].float()
            loss += loss_criterion_for_line(output_tensor_line, expected_output)

        return loss

    def locate_paragraphs(self, image, debugging=False) -> List[Tuple[Rectangle, List[Rectangle]]]:
        output, _ = self._create_output_mask_tensors(image)
        return cv.locate_paragraphs_in_character_mask(output, debugging=debugging)

    def locate_lines(self, image, debugging=False) -> List[Rectangle]:
        output, _ = self._create_output_mask_tensors(image)
        return cv.locate_lines_in_character_mask(output,  debugging=debugging)

    def image_to_tensor(self, image) -> Tensor:
        image_tensor = TRANSFORM_TO_TENSOR(image).float() / 255
        # if self.training_mode:
        #     image_tensor = TRANSFORM_ADD_NOISE(image_tensor)
        return image_tensor

    def output_mask_to_tensor(self, image) -> Tensor:
        return utils.output_mark_image_to_tensor(image, threshold_min=0.5)

    def _create_output_mark_images(self, image) -> Tuple[Image.Image, Image.Image]:
        output_char, output_line = self._create_output_mask_tensors(image)

        return utils.output_tensor_to_image_mask(output_char), \
               utils.output_tensor_to_image_mask(output_line)

    def _create_output_mask_tensors(self, image) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            input_tensor = self.image_to_tensor(image).unsqueeze(0)
            output_char, output_line = self(input_tensor)

        return torch.sigmoid(output_char[0]), torch.sigmoid(output_line[0])


if __name__ == '__main__':
    from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from comic_ocr.utils import image_with_annotations, concatenated_images
    from comic_ocr.utils.files import load_image, get_path_project_dir

    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    path_output_model = get_path_project_dir('data/output/models/localization.bin')
    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)

        output_mask_char, output_mask_line = model._create_output_mark_images(example)
        paragraphs = model.locate_paragraphs(example)
        concatenated_images([
            example,
            output_mask_char,
            output_mask_line,
            image_with_annotations(example, [rect for rect, line_rects in paragraphs]),
        ], num_col=4).show()

    print('Creating a new model...')
    model = ConvUnet()
