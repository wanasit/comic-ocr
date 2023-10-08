"""An abstraction and shared functionalities for localization models.
"""

import os
from typing import Callable, Tuple, List, Any, Optional

import torch
from PIL import Image
from torch import nn

from comic_ocr.models.localization import localization_open_cv as cv
from comic_ocr.models.localization.localization_utils import image_to_input_tensor, output_tensor_to_image_mask
from comic_ocr.types import Size, Rectangle


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=1.0, pos_weight=1.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.weight = weight

    def forward(self, y_pred, y):
        return self.bce_loss(y_pred, y) * self.weight


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=1.0, smooth=1, apply_sigmoid_to_y_pred=True):
        super().__init__()
        self.smooth = smooth
        self.weight = weight
        self.apply_sigmoid_to_y_pred = apply_sigmoid_to_y_pred

    def forward(self, y_pred, y):
        if self.apply_sigmoid_to_y_pred:
            y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y.sum() + self.smooth)
        return (1 - dice) * self.weight


def default_lost_criterion_char():
    return WeightedBCEWithLogitsLoss(weight=1.0)


def default_lost_criterion_line():
    return WeightedDiceLoss(weight=0.5)


class LocalizationModel(nn.Module):
    """An abstraction for localization models (nn.Module).

    A localization model extends this abstraction should classify if input image's pixel a character or a line (Semantic
    Segmentation problem). Specifically, given an image as input, the model should output upto probability masks
    (similar size to input) for each pixel is a part of a character or a line.

    For training, this abstract class implements `compute_loss()` that computes the model binary classification loss on
    a LocalizationDataset's batch (images and labelled marks).

    For evaluation or serving, this abstract class implements `locate_paragraphs()` (and `locate_lines()`) that takes
    image input, computes the probability via the model, and uses heuristic OpenCV techniques to identify the lines.
    """

    def __init__(self, preferred_image_size: Optional[Size] = None):
        super().__init__()
        self._preferred_image_size = preferred_image_size

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
        if self._preferred_image_size is not None:
            return self._preferred_image_size
        return Size.of(500, 500)

    def compute_loss(
            self,
            dataset_batch,
            loss_criterion_for_char: Optional[Callable] = None,
            loss_criterion_for_line: Optional[Callable] = None,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:

        if loss_criterion_for_char is None:
            loss_criterion_for_char = default_lost_criterion_char()

        if loss_criterion_for_line is None:
            loss_criterion_for_line = default_lost_criterion_line()

        input_tensor = dataset_batch['input']
        if device is not None:
            self.to(device)
            input_tensor = input_tensor.to(device)

        output_char, output_line, _ = self(input_tensor)
        loss = torch.zeros(1, device=device)
        if 'output_mask_char' in dataset_batch:
            output = dataset_batch['output_mask_char'].float()
            if device is not None:
                output = output.to(device)
                loss_criterion_for_char = loss_criterion_for_char.to(device) if \
                    isinstance(loss_criterion_for_char, nn.Module) else loss_criterion_for_char
            loss += loss_criterion_for_char(output_char, output)

        if 'output_mask_line' in dataset_batch:
            output = dataset_batch['output_mask_line'].float()
            if device is not None:
                output = output.to(device)
                loss_criterion_for_line = loss_criterion_for_line.to(device) if \
                    isinstance(loss_criterion_for_line, nn.Module) else loss_criterion_for_line
            loss += loss_criterion_for_line(output_line, output)

        return loss

    def locate_paragraphs(
            self,
            image: Image.Image,
            threshold: float = 0.5,
            device: Optional[torch.device] = None
    ) -> List[Tuple[Rectangle, List[Rectangle]]]:
        output, _ = self._create_prob_output_masks(image, device=device)
        return cv.locate_paragraphs_in_character_mask(output, input_threshold=threshold)

    def locate_lines(
            self,
            image: Image.Image,
            threshold: float = 0.5,
            device: Optional[torch.device] = None
    ) -> List[Rectangle]:
        output, _ = self._create_prob_output_masks(image, device=device)
        return cv.locate_lines_in_character_mask(output, input_threshold=threshold)

    def create_output_marks(self, image, device: Optional[torch.device] = None) -> Tuple[Image.Image, Image.Image]:
        output_char, output_line = self._create_prob_output_masks(image, device=device)
        return output_tensor_to_image_mask(output_char), output_tensor_to_image_mask(output_line)

    def _create_prob_output_masks(
            self,
            image: Image.Image,
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device if device is not None else torch.device('cpu')
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            self.to(device)
            input_tensor = input_tensor.to(device)
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
