"""An abstraction and shared functionalities for localization models.
"""

import os
from typing import Callable, Tuple, List, Any

import torch
from PIL import Image
from torch import nn

from comic_ocr.models.localization import localization_open_cv as cv
from comic_ocr.models.localization.localization_utils import image_to_input_tensor, output_tensor_to_image_mask
from comic_ocr.types import Size, Rectangle


def WeightedBCEWithLogitsLoss(weight: float, pos_weight: float):
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
    return lambda y_pred, y: bce_loss(y_pred, y) * weight


DEFAULT_LOSS_CRITERION_CHAR = WeightedBCEWithLogitsLoss(pos_weight=0.5, weight=2.0)
DEFAULT_LOSS_CRITERION_LINE = WeightedBCEWithLogitsLoss(pos_weight=0.5, weight=1.0)


class LocalizationModel(nn.Module):
    """An abstract for localization models (nn.Module).

    A localization model extends this abstraction should classify if input image's pixel a character or a line (Semantic
    Segmentation problem). Specifically, given an image as input, the model should output upto probability masks
    (similar size to input) for each pixel is a part of a character or a line.

    For training, this abstract class implements `compute_loss()` that computes the model binary classification loss on
    a LocalizationDataset's batch (images and labelled marks).

    For evaluation or serving, this abstract class implements `locate_paragraphs()` (and `locate_lines()`) that takes
    image input, computes the probability via the model, and uses heuristic OpenCV techniques to identify the lines.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Computes character and line probability marks for the input image.

        Args:
            x (torch.Tensor[?, C=3, H, W]): the input image

        Returns:
            output_mask_character (torch.Tensor[?, H, W]): The predict probability mask for characters
            output_mask_line (torch.Tensor[?, H, W]): The predict probability mask for lines
            other_output: Other output used by the implementation classes.
        """
        raise NotImplementedError

    @property
    def preferred_image_size(self) -> Size:
        return Size.of(500, 500)

    def compute_loss(
            self,
            dataset_batch,
            loss_criterion_for_char=DEFAULT_LOSS_CRITERION_CHAR,
            loss_criterion_for_line=DEFAULT_LOSS_CRITERION_LINE
    ) -> torch.Tensor:
        input = dataset_batch['input']
        output_char, output_line, _ = self(input)

        loss = torch.zeros(1)
        if 'output_mask_char' in dataset_batch:
            output = dataset_batch['output_mask_char'].float()
            loss += loss_criterion_for_char(output_char, output)

        if 'output_mask_line' in dataset_batch:
            output = dataset_batch['output_mask_line'].float()
            loss += loss_criterion_for_line(output_line, output)

        return loss

    def locate_paragraphs(self, image, threshold=0.5) -> List[Tuple[Rectangle, List[Rectangle]]]:
        output, _ = self._create_output_mask(image)
        return cv.locate_paragraphs_in_character_mask(output, input_threshold=threshold)

    def locate_lines(self, image, threshold=0.5) -> List[Rectangle]:
        output, _ = self._create_output_mask(image)
        return cv.locate_lines_in_character_mask(output, input_threshold=threshold)

    def create_output_marks(self, image) -> Tuple[Image.Image, Image.Image]:
        output_char, output_line = self._create_output_mask(image)

        return output_tensor_to_image_mask(output_char), output_tensor_to_image_mask(output_line)

    def _create_output_mask(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            output_char, output_line, _ = self(input_tensor)

        return torch.sigmoid(output_char[0]), torch.sigmoid(output_line[0])


class BasicLocalizationModel(LocalizationModel):
    """A basic implementation for the LocalizationModel to be used for testing.

    The model transform the input into the two outputs by two conv2d layers.
    """

    def __init__(self, kernel_size=1, stride=1):
        super(BasicLocalizationModel, self).__init__()
        self.output_conv_char = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride)
        self.output_conv_line = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        y_char = self.output_conv_char(x)
        y_line = self.output_conv_line(x)
        return y_char[:, 0, :], y_line[:, 0, :], None

    def reset_parameters(self):
        self.output_conv_char.reset_parameters()
        self.output_conv_line.reset_parameters()


if __name__ == '__main__':
    from comic_ocr.utils import image_with_annotations, concatenated_images
    from comic_ocr.utils.files import load_image, get_path_project_dir

    path_output_model = get_path_project_dir('data/output/models/localization_base.bin')
    # path_output_model = ''

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = BasicLocalizationModel()

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
