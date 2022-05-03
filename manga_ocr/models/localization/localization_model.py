import os
from typing import Callable, Tuple, Union, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor

from manga_ocr.models.localization.localization_utils import image_to_input_tensor, output_tensor_to_image_mask
from manga_ocr.typing import Size, Rectangle


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
            criterion=nn.BCEWithLogitsLoss(),
            weight_char_prediction=0.5,
            weight_line_prediction=0.5,
    ) -> Tensor:
        input = dataset_batch['input']
        output_char, output_line = self(input)

        loss = torch.zeros(1)
        if 'output_mask_char' in dataset_batch:
            output = dataset_batch['output_mask_char'].float()
            loss += criterion(output_char, output) * weight_char_prediction

        if 'output_mask_line' in dataset_batch:
            output = dataset_batch['output_mask_line'].float()
            loss += criterion(output_line, output) * weight_line_prediction

        return loss

    def locate_paragraphs(self, image) -> List[Tuple[Rectangle, List[Rectangle]]]:
        lines = self.locate_lines(image)
        return group_lines_into_paragraphs(lines)

    def locate_lines(self, image) -> List[Rectangle]:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            _, output = self(input_tensor)
            output = torch.sigmoid(output[0])

        return locate_lines_in_image_mask(output)

    def create_output_marks(self, image) -> Tuple[Image.Image, Image.Image]:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            output_char, output_line = self(input_tensor)

            output_char = torch.sigmoid(output_char[0])
            output_line = torch.sigmoid(output_line[0])

        return output_tensor_to_image_mask(output_char), \
               output_tensor_to_image_mask(output_line)


def locate_lines_in_image_mask(
        output_tensor: Union[np.ndarray, Tensor],
        output_threshold: float = 0.95,
        line_output_min_size: Size = Size.of(10, 10),
        line_padding: Tuple[int, int] = (2, 1),
        debugging=True
) -> List[Rectangle]:
    """Locate the line locations in the output tensor or array

    Args:
        output_tensor (np.ndarray, torch.Tensor): the prediction output tensor or array with shape and value similar to
            the tensor returned by `image_mask_to_output_tensor()`
        output_threshold (float)
        line_output_density_threshold (float)
        line_output_min_size (Size, Tuple[float, float])
    Returns:
        List[Rectangle]: the line locations
    """
    if isinstance(output_tensor, Tensor):
        output_tensor = output_tensor.numpy()

    if len(output_tensor.shape) == 3:
        output_tensor = output_tensor.mean(axis=0)

    _, thresh = cv2.threshold(output_tensor, output_threshold, 1, cv2.THRESH_BINARY)
    # _debug_show_output(True, thresh)

    output_rects = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh.astype(np.uint8), connectivity=4)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w <= line_output_min_size[0] or h <= line_output_min_size[1]:
            continue

        rect = Rectangle.of_xywh(x, y, w, h)
        rect = rect.expand(line_padding)
        output_rects.append(rect)

    return output_rects


def group_lines_into_paragraphs(lines: List[Rectangle]) -> List[Tuple[Rectangle, List[Rectangle]]]:
    paragraphs: List[Tuple[Rectangle, List[Rectangle]]] = []
    for line in lines:
        for i in range(len(paragraphs)):
            paragraph_rect, paragraph_lines = paragraphs[i]

            if _paragraph_align_left(paragraph_rect, line) or _paragraph_align_center(paragraph_rect, line):
                new_paragraph_rect = Rectangle.union_bounding_rect((paragraph_rect, line))
                paragraphs[i] = (new_paragraph_rect, paragraph_lines + [line])
                break
        else:
            paragraphs.append((line, [line]))

    return paragraphs


def _paragraph_align_left(paragraph_rect: Rectangle, line_rect: Rectangle, x_margin=5, y_min_margin=10):
    y_margin = max(line_rect.height / 5, y_min_margin)

    return abs(paragraph_rect.left - line_rect.left) < x_margin and \
           abs(paragraph_rect.bottom - line_rect.top) < y_margin


def _paragraph_align_center(paragraph_rect: Rectangle, line_rect: Rectangle, x_margin=10, y_min_margin=10):
    y_margin = max(line_rect.height / 5, y_min_margin)

    return abs(paragraph_rect.center[0] - line_rect.center[0]) < x_margin and \
           abs(paragraph_rect.bottom - line_rect.top) < y_margin


if __name__ == '__main__':
    from manga_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    from manga_ocr.utils import image_with_annotations, concatenated_images
    from manga_ocr.utils.files import load_image, get_path_project_dir

    path_output_model = get_path_project_dir('data/output/models/localization.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    # example = load_image(get_path_project_dir('example/manga_generated/image/0001.jpg'))
    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    paragraphs = model.locate_paragraphs(example)

    paragraph_locations = [rect for rect, _ in paragraphs]
    line_locations = [l for _, lines in paragraphs for l in lines]
    output_char, output_line = model.create_output_marks(example)

    concatenated_images([
        example,
        output_char,
        output_line,
        image_with_annotations(example, line_locations),
    ], num_col=4).show()

