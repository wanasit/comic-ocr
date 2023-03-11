"""Localization methods based-on OpenCV (opencv-python or cv2)
"""
from typing import Union, Tuple, List

import cv2
import numpy as np
import torch

from comic_ocr.types import Rectangle


def locate_paragraphs_in_character_mask(
        input_tensor: Union[np.ndarray, torch.Tensor],
        *args, **kwargs
) -> List[Tuple[Rectangle, List[Rectangle]]]:
    """Locate the lines and paragraphs in the LocalizationModel's character mask output.

    The function expects 2D array/tensor of `white text/characters (1.0) written in the black background (0.0)`

    See locate_lines_in_character_mask() functions for more details and tuning parameters.

    Args:
        input_tensor (np.ndarray or torch.Tensor): the prediction output tensor or array with shape and value
            similar to the tensor returned by `image_mask_to_output_tensor()`.
    Returns:
        List[Tuple[Rectangle, List[Rectangle]]]: The list of paragraphs. For each paragraph, it returns
        the paragraph location (Rectangle) and paragraph's line locations (List[Rectangle])
    """
    lines = locate_lines_in_character_mask(input_tensor, *args, **kwargs)
    return group_lines_into_paragraphs(lines)


def locate_lines_in_character_mask(
        input_tensor: Union[np.ndarray, torch.Tensor],
        input_threshold: float = 0.60,
        expand_detected_component: Tuple[int, int] = (2, 1),
        expand_detected_line: Tuple[int, int] = (2, 2),
        debugging: bool = False
) -> List[Rectangle]:
    """Locate the lines in the LocalizationModel's character mask output.

    The function expects 2D array/tensor of `white text/characters (1.0) written in the black background (0.0)`

    Args:
        input_tensor (np.ndarray or torch.Tensor): the prediction output tensor or array with shape and value
            similar to the tensor returned by `image_mask_to_output_tensor()`.

        input_threshold: to transform the input into binary image
        expand_detected_component: to expand the located line
        expand_detected_line: to expand the located line
        debugging: to pause and show debugging image while running
    Returns:
        List[Rectangle]: the line locations
    """
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.numpy()

    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.mean(axis=0)

    _, thresh = cv2.threshold(input_tensor, input_threshold, 1, cv2.THRESH_BINARY)
    _debugging_show(debugging, thresh)

    output_rects = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh.astype(np.uint8), connectivity=4)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        rect = Rectangle.of_xywh(x, y, w, h).expand(expand_detected_component)
        output_rects.append(rect)

    lines: List[Rectangle] = []
    for char_block in output_rects:
        i = len(lines) - 1
        while i >= 0:
            if not align_line_horizontal(lines[i], char_block):
                i -= 1
                continue
            char_block = Rectangle.union_bounding_rect((lines.pop(i), char_block))
            i = len(lines) - 1
        lines.append(char_block)

    lines = [l.expand(expand_detected_line) for l in lines]
    return lines


def group_lines_into_paragraphs(lines: List[Rectangle]) -> List[Tuple[Rectangle, List[Rectangle]]]:
    paragraphs: List[Tuple[Rectangle, List[Rectangle]]] = []
    for line in lines:
        for i in range(len(paragraphs)):
            paragraph_rect, paragraph_lines = paragraphs[i]

            if align_paragraph_left(paragraph_rect, line) or align_paragraph_center(paragraph_rect, line):
                new_paragraph_rect = Rectangle.union_bounding_rect((paragraph_rect, line))
                paragraphs[i] = (new_paragraph_rect, paragraph_lines + [line])
                break
        else:
            paragraphs.append((line, [line]))

    return paragraphs


def align_line_horizontal(block_a: Rectangle, block_b: Rectangle, x_min_margin=10, y_margin=2):
    """Returns true if two input character blocks are aligned into horizontal line."""

    # First, check if horizontal distance (x-axis) is too far apart
    x_margin = max(block_a.height, block_b.height, x_min_margin)
    x_distance = max(block_a.left, block_b.left) - min(block_a.right, block_b.right)
    if x_distance > x_margin:
        return False

    # Assume block_a is taller than block_b, we check if the block_b is vertically within block_a
    if block_a.height < block_b.height:
        block_a, block_b = block_b, block_a
    return (block_b.bottom <= block_a.bottom + y_margin) and (block_b.top >= block_a.top - y_margin)


def align_paragraph_left(paragraph_rect: Rectangle, line_rect: Rectangle, x_margin=5, y_min_margin=10):
    y_margin = max(line_rect.height / 5, y_min_margin)

    return abs(paragraph_rect.left - line_rect.left) < x_margin and \
           abs(paragraph_rect.bottom - line_rect.top) < y_margin


def align_paragraph_center(paragraph_rect: Rectangle, line_rect: Rectangle, x_margin=10, y_min_margin=10):
    y_margin = max(line_rect.height / 5, y_min_margin)

    return abs(paragraph_rect.center[0] - line_rect.center[0]) < x_margin and \
           abs(paragraph_rect.bottom - line_rect.top) < y_margin


def _debugging_show(debugging, cv_image, text=''):
    if debugging:
        cv2.imshow(text, cv_image)
        cv2.waitKey(0)
