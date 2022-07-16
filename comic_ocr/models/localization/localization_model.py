"""The module provides abstract/shared functionalities for ML model for localization
"""

import os
from typing import Callable, Tuple, List

import torch
from PIL import Image
from torch import nn, Tensor

from comic_ocr.models.localization import localization_open_cv as cv
from comic_ocr.models.localization.localization_utils import image_to_input_tensor, output_tensor_to_image_mask
from comic_ocr.typing import Size, Rectangle

DEFAULT_LOSS_CRITERION_CHAR = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2]))
DEFAULT_LOSS_CRITERION_LINE = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))


class LocalizationModel(nn.Module):
    """An abstract for localization Module for text localization

    The ML model (Pytorch) classifies each input image's pixel if it's part of a character, a line boundary, or a
    paragraph boundary. The model output those probabilities as 'mask images' in training dataset format. Namely, given
    the input source image, the model is trained to output three types of mask images.

    Args:
        input_image (Tensor [C x H x W])

    Returns probability maps (un-normalized):
        output_mask_character (Tensor [H x W]) the probability that the pixel is character/text
        output_mask_line (Tensor [H x W]) the probability that the pixel is inside line rect boundary

    The module also applies other non-ML image-processing techniques on top of the mask images to output the locations.

    Example:
        model = ....
        paragraph_rectangles = model.locate_lines(image)
    """

    __call__: Callable[..., Tuple[Tensor, Tensor]]

    def __init__(self):
        super().__init__()

    @property
    def preferred_image_size(self) -> Size:
        return Size.of(500, 500)

    def compute_loss(
            self,
            dataset_batch,
            loss_criterion_for_char=DEFAULT_LOSS_CRITERION_CHAR,
            loss_criterion_for_line=DEFAULT_LOSS_CRITERION_LINE
    ) -> Tensor:
        input = dataset_batch['input']
        output_char, output_line = self(input)

        loss = torch.zeros(1)
        if 'output_mask_char' in dataset_batch:
            output = dataset_batch['output_mask_char'].float()
            loss += loss_criterion_for_char(output_char, output)

        if 'output_mask_line' in dataset_batch:
            output = dataset_batch['output_mask_line'].float()
            loss += loss_criterion_for_line(output_line, output)

        return loss

    def locate_paragraphs(self, image) -> List[Tuple[Rectangle, List[Rectangle]]]:
        output, _ = self._create_output_mask(image)
        return cv.locate_paragraphs_in_character_mask(output)

    def locate_lines(self, image) -> List[Rectangle]:
        output, _ = self._create_output_mask(image)
        return cv.locate_lines_in_character_mask(output)

    def create_output_marks(self, image) -> Tuple[Image.Image, Image.Image]:
        output_char, output_line = self._create_output_mask(image)

        return output_tensor_to_image_mask(output_char), output_tensor_to_image_mask(output_line)

    def _create_output_mask(self, image) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            output_char, output_line = self(input_tensor)

        return torch.sigmoid(output_char[0]), torch.sigmoid(output_line[0])


if __name__ == '__main__':
    from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from comic_ocr.utils import image_with_annotations, concatenated_images
    from comic_ocr.utils.files import load_image, get_path_project_dir

    path_output_model = get_path_project_dir('data/output/models/localization.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    # example = load_image(get_path_project_dir('example/manga_generated/image/0001.jpg'))
    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    # example = load_image(get_path_project_dir('data/manga_line_annotated/u_01.jpg'))

    # paragraphs = model.locate_paragraphs(example)
    #
    # paragraph_locations = [rect for rect, _ in paragraphs]
    # line_locations = [l for _, lines in paragraphs for l in lines]
    line_locations = model.locate_lines(example)
    output_mask_char, output_mask_line = model.create_output_marks(example)

    concatenated_images([
        example,
        output_mask_char,
        output_mask_line,
        image_with_annotations(example, line_locations),
    ], num_col=4).show()
