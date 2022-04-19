import os
from typing import Callable, Tuple, Union, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor
from torchvision import transforms

from manga_ocr.models.transforms import AddGaussianNoise
from manga_ocr.typing import Size, Rectangle

TRANSFORM_TO_TENSOR = transforms.PILToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()
TRANSFORM_ADD_NOISE = AddGaussianNoise()


def _debug_show_output(debugging, output, text=''):
    if debugging:
        cv2.imshow(text, output)
        cv2.waitKey(0)


def output_tensor_to_image_mask(tensor):
    return Image.fromarray(np.uint8(tensor.numpy()[0] * 255), 'L').convert('RGB')


class LocalizationModel(nn.Module):
    """
    (-1, input_size[0], input_size[1])
    """

    __call__: Callable[..., Tuple[Tensor, Tensor]]

    def __init__(self):
        super().__init__()

    @property
    def image_size(self) -> Size:
        return Size.of(500, 500)

    def compute_loss(self, dataset_batch, criterion=nn.BCEWithLogitsLoss(), char_pred_weight=0.5,
                     line_pre_weight=0.5) -> Tensor:
        input = dataset_batch['image']
        mask_line = dataset_batch['mask_line'].float()
        mask_character = dataset_batch['mask_char'].float()

        output_char, output_line = self(input)
        loss = criterion(output_char, mask_character) * char_pred_weight + \
               criterion(output_line, mask_line) * line_pre_weight

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

    def create_image_mark_lines(self, image) -> Image.Image:
        with torch.no_grad():
            input_tensor = image_to_input_tensor(image).unsqueeze(0)
            _, output = self(input_tensor)
            output = torch.sigmoid(output[0])
        return output_tensor_to_image_mask(output)


def image_to_input_tensor(image):
    input = TRANSFORM_TO_TENSOR(image).float() / 255
    return input


def image_mask_to_output_tensor(image, threshold_min: float = 0.5, threshold_max: float = 1.0):
    output = image_to_input_tensor(image)
    output = TRANSFORM_TO_GRAY_SCALE(output)
    output = ((threshold_max >= output) & (output > threshold_min)).float()
    return output


def locate_lines_in_image_mask(
        output_tensor: Union[np.ndarray, Tensor],
        output_threshold: float = 0.95,
        line_output_density_threshold: float = 0.80,
        line_output_min_size: Size = Size.of(10, 10),
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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w <= line_output_min_size[0] or h <= line_output_min_size[1]:
            continue

        rect = Rectangle.of_xywh(x, y, w, h)
        inner_rect = rect.expand((0, -3))

        density = thresh[inner_rect.top:inner_rect.bottom, inner_rect.left:inner_rect.right].sum() \
                  / inner_rect.height / inner_rect.width
        if density < line_output_density_threshold:
            continue
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
    from manga_ocr.utils import image_with_annotations
    from manga_ocr.utils.files import load_image, get_path_project_dir

    path_output_model = get_path_project_dir('data/output/models/localization.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = ConvUnet()

    #example = load_image(get_path_project_dir('example/manga_generated/image/0001.jpg'))
    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    paragraphs = model.locate_paragraphs(example)

    paragraph_locations = [rect for rect, _ in paragraphs]
    image_with_annotations(example, paragraph_locations).show()

    lines = [l for _, lines in paragraphs for l in lines]
    image_with_annotations(example, lines).show()

    model.create_image_mark_lines(example).show()
