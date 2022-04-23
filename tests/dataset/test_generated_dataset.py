import numpy as np
from PIL import ImageFont
from PIL import Image

import manga_ocr.dataset.generated_manga as generated_manga
from manga_ocr import Rectangle
from manga_ocr.dataset.generated_manga.text_area import TextArea
from manga_ocr.typing import Point
from manga_ocr.utils.files import get_path_example_dir


def test_text_area():
    font = ImageFont.truetype(get_path_example_dir('./fonts/Komika_Text.ttf'), size=20)
    text = 'This is very long long long long long sentence trying to create a line break.'

    xy = Point.of(5, 5)
    text_area = TextArea(xy, text=text, font=font, max_width=200)

    assert text_area.text_rect.tl == xy
    assert text_area.text_rect.width == 204
    assert text_area.text_rect.height == 79

    assert len(text_area.text_lines) == 3
    assert text_area.text_lines[0].text == 'This is very long long long long'
    assert text_area.text_lines[0].location.height == 20
    assert (text_area.text_lines[0].location.top - text_area.text_rect.top) > 2

    assert text_area.text_lines[1].text == 'long sentence trying to create'
    assert text_area.text_lines[1].location.height == 20
    assert not Rectangle.is_overlap(text_area.text_lines[1].location, text_area.text_lines[0].location)
    assert not Rectangle.is_overlap(text_area.text_lines[1].location, text_area.text_lines[2].location)

    assert text_area.text_lines[2].text == 'a line break.'
    assert text_area.text_lines[2].location.height == 20
    assert (text_area.text_rect.bottom - text_area.text_lines[0].location.bottom) > 2

    for line in text_area.text_lines:
        assert line.location in text_area.text_rect
        assert line.location.height == 20


def test_text_area_draw_text_rect():
    # given text area with three lines
    font = ImageFont.truetype(get_path_example_dir('./fonts/Komika_Text.ttf'), size=20)
    text = 'This is very long long long long long sentence trying to create a line break.'

    xy = Point.of(5, 5)
    text_area = TextArea(xy, text=text, font=font, max_width=200)
    assert len(text_area.text_lines) == 3
    assert text_area.text_rect == Rectangle.of_size((204, 79), at=xy)

    # when draw_text_rect()
    image = Image.new('RGBA', size=(300, 100), color='#000000ff')
    text_area.draw_text_rect(image, fill='#ff0000ff')

    pixel_values = list(image.getdata())
    for y in range(100):
        for x in range(300):
            p = pixel_values[y * 300 + x]
            if (x, y) in text_area.text_rect:
                assert p[0] >= 255, \
                    f'The pixel {p} at {x, y} (inside rectangle {text_area.text_rect}) should be red'
            else:
                assert p[0] == 0, \
                    f'The pixel {p} at {x, y} (outside rectangle {text_area.text_rect}) should be black'


def test_text_area_draw_line_rects():
    # given text area with three lines
    font = ImageFont.truetype(get_path_example_dir('./fonts/Komika_Text.ttf'), size=20)
    text = 'This is very long long long long long sentence trying to create a line break.'

    xy = Point.of(5, 5)
    text_area = TextArea(xy, text=text, font=font, max_width=200)
    assert len(text_area.text_lines) == 3
    assert text_area.text_rect == Rectangle.of_size((204, 79), at=xy)

    image = Image.new('RGB', size=(300, 100), color=(0, 0, 0))
    fill_value_text_rect = 255 // 5
    fill_value_line_rects = 255 // 2

    # when draw_text_rect() and draw_line_rects()
    text_area.draw_text_rect(image, (fill_value_text_rect,) * 3)
    text_area.draw_line_rects(image, fill=(fill_value_line_rects,) * 3)
    # image.show()

    # then
    pixel_values = list(image.getdata())
    for y in range(100):
        line_rect = next((l.location for l in text_area.text_lines if l.location.top <= y <= l.location.bottom), None)
        for x in range(300):
            p = pixel_values[y * 300 + x]

            if line_rect and (x, y) in line_rect:
                assert p[0] >= fill_value_line_rects, \
                    f'Pixel {p} at {x, y} (inside line {line_rect}) should be {fill_value_line_rects}'
                continue

            if (x, y) in text_area.text_rect:
                assert p[0] >= fill_value_text_rect, \
                    f'Pixel {p} at {x, y} (inside rectangle {text_area.text_rect}) should be {fill_value_text_rect}'
                continue

            assert p[0] == 0, \
                f'Pixel {p} at {x, y} (outside rectangle {text_area.text_rect}) should be empty'



def test_text_area_draw_text():
    # given text area with three lines
    font = ImageFont.truetype(get_path_example_dir('./fonts/Komika_Text.ttf'), size=20)
    text = 'This is very long long long long long sentence trying to create a line break.'

    xy = Point.of(5, 5)
    text_area = TextArea(xy, text=text, font=font, max_width=200)
    assert len(text_area.text_lines) == 3
    assert text_area.text_rect == Rectangle.of_size((204, 79), at=xy)

    image = Image.new('RGB', size=(300, 100), color=(0, 0, 0))
    fill_value_text_rect = 255 // 5
    fill_value_line_rects = 255 // 3
    fill_value_text = 255

    # when draw_text()
    text_area.draw_text_rect(image, fill=(fill_value_text_rect,) * 3)
    text_area.draw_line_rects(image, fill=(fill_value_line_rects,) * 3)
    text_area.draw_text(image, fill=(fill_value_text,) * 3)
    # image.show()

    # then
    pixel_values = list(image.getdata())
    for y in range(100):
        line_rect = next((l.location for l in text_area.text_lines if l.location.top <= y <= l.location.bottom), None)
        for x in range(300):
            p = pixel_values[y * 300 + x]

            if p[0] > 128: # each character pixel should be inside the line and rectangle
                character_pixel = Point.of(x, y)
                assert line_rect and character_pixel in line_rect, f'Pixel {character_pixel} should be inside line {line_rect}'
                assert character_pixel in text_area.text_rect


def test_manga_generator():
    generator = generated_manga.MangaGenerator.create()
    image, text_areas = generator.generate(output_size=(750, 750))

    assert image.size == (750, 750)
    assert 3 < len(text_areas) < 8


def test_load_example_dataset():
    example_dataset_dir = get_path_example_dir('manga_generated')
    images, image_texts, image_masks = generated_manga.load_dataset(example_dataset_dir)
    assert len(images) == 3
    assert len(image_texts) == 3
    assert len(image_masks) == 3

    assert isinstance(images[0], Image.Image)
    assert isinstance(image_masks[0], Image.Image)

    assert len(image_texts[0]) > 0


def test_generate_dataset(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=2)

    images, image_texts, image_masks = generated_manga.load_dataset(dataset_dir)
    assert len(images) == 2
    assert len(image_texts) == 2
    assert len(image_masks) == 2

    assert isinstance(images[0], Image.Image)
    assert isinstance(image_masks[0], Image.Image)

    assert len(image_texts[0]) > 0


def test_generate_on_the_same_location(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=3)
    generated_manga.create_dataset(dataset_dir, output_count=4)
