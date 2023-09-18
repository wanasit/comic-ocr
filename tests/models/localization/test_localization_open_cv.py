from comic_ocr import Rectangle
from comic_ocr.dataset import generated_manga
from comic_ocr.models.localization import localization_open_cv
from comic_ocr.models.localization import localization_utils
from comic_ocr.utils import files


def test_locate_lines_in_character_mask():
    example_generated_dataset_dir = files.get_path_example_dir('manga_generated')
    _, image_texts, image_masks = generated_manga.load_dataset(example_generated_dataset_dir)

    image_mask = image_masks[0]
    lines = image_texts[0]

    output_tensor = localization_utils.image_mask_to_output_tensor(image_mask)
    located_lines = localization_open_cv.locate_lines_in_character_mask(output_tensor)

    assert len(lines) == len(located_lines)
    lines = sorted(lines, key=lambda l: l.location.top)
    located_lines = sorted(located_lines, key=lambda l: l.top)

    for i in range(len(lines)):
        assert lines[i].location in located_lines[i]


def test_align_line_horizontal_positive_cases():
    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(21, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(21, 10, 10, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(10, 10, 10, 5)
    block_b = Rectangle.of_xywh(21, 10, 10, 10)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    x = Rectangle.of_size(size=(55, 15), at=(520, 690))
    y = Rectangle.of_size(size=(12, 15), at=(580, 690))
    assert localization_open_cv.align_line_horizontal(x, y)
    assert localization_open_cv.align_line_horizontal(y, x)


def test_align_line_horizontal_negative_cases():
    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(50, 10, 10, 10)
    assert not localization_open_cv.align_line_horizontal(block_a, block_b)
    assert not localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(21, 15, 10, 10)
    assert not localization_open_cv.align_line_horizontal(block_a, block_b)
    assert not localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(10, 10, 10, 10)
    block_b = Rectangle.of_xywh(0, 15, 10, 10)
    assert not localization_open_cv.align_line_horizontal(block_a, block_b)
    assert not localization_open_cv.align_line_horizontal(block_b, block_a)


def test_align_line_horizontal_when_overlap():
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

    x = Rectangle.of_size(size=(61, 15), at=(521, 690))
    y = Rectangle.of_size(size=(12, 16), at=(580, 689))
    assert localization_open_cv.align_line_horizontal(x, y)
    assert localization_open_cv.align_line_horizontal(y, x)


def test_align_line_horizontal_when_overlap_vertically():
    block_a = Rectangle.of_xywh(20, 20, 100, 20)
    block_b = Rectangle.of_xywh(15, 19, 5, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(20, 20, 100, 20)
    block_b = Rectangle.of_xywh(121, 19, 5, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(20, 20, 100, 20)
    block_b = Rectangle.of_xywh(15, 35, 5, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    block_a = Rectangle.of_xywh(20, 20, 100, 20)
    block_b = Rectangle.of_xywh(121, 35, 5, 5)
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

def test_align_line_horizontal_when_size_is_differ():

    # Note: block_b is a "," character following block_a
    block_a = Rectangle.of_size((70, 24), at=(238, 698))
    block_b = Rectangle.of_size((13, 13), at=(301, 712))
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)

    # Note: block_a is a "." dot above/overlapping with block_b
    block_a = Rectangle.of_size((6, 4), at=(294, 98))
    block_b = Rectangle.of_size((118, 14), at=(204, 101))
    assert localization_open_cv.align_line_horizontal(block_a, block_b)
    assert localization_open_cv.align_line_horizontal(block_b, block_a)
