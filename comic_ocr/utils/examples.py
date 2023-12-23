from typing import List

from PIL import ImageFont, Image

from comic_ocr.utils import files


def load_example_fonts() -> List[ImageFont.ImageFont]:
    example_font_dir = files.get_path_example_dir() + '/fonts/'
    # noinspection PyTypeChecker
    return \
            [ImageFont.truetype(example_font_dir + 'Komika_Text.ttf', size=15)] + \
            [ImageFont.truetype(example_font_dir + 'Komika_Text.ttf', size=20)] + \
            [ImageFont.truetype(example_font_dir + 'Cool Cat.ttf', size=16)] * 3 + \
            [ImageFont.truetype(example_font_dir + 'Cool Cat.ttf', size=21)]


def load_example_drawing() -> List[Image.Image]:
    return files.load_images(files.get_path_example_dir() + '/drawings/*.jpg')[0]


def load_example_texts() -> List[str]:
    return files.load_texts(files.get_path_example_dir() + '/text/texts.txt')
