from comic_ocr import Rectangle
from comic_ocr.models.localization import localization_open_cv


def test_align_line_horizontal_positive():
    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(21, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)

    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(21, 10, 10, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)

    block_a = Rectangle.of_xywh(10, 10, 10, 5)
    block_b = Rectangle.of_xywh(21, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)


def test_align_line_horizontal_overlap():
    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(10, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(9, 9, 12, 12)
    block_b = Rectangle.of_xywh(10, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_size(size=(13, 6), at=(433, 520))
    block_b = Rectangle.of_size(size=(35, 20), at=(433, 520))
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)
