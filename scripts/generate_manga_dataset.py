import os

from PIL import ImageFont

from manga_ocr.dataset.generated_manga import MangaGenerator, create_dataset
from manga_ocr.typing import Size
from manga_ocr.utils import load_images, load_texts

current_module_dir = os.path.dirname(__file__)

generator_input_dir = current_module_dir + '/../data'
choices_drawings = load_images(generator_input_dir + '/drawings/*.jpg')
choices_texts = load_texts(generator_input_dir + '/text/texts.txt')
choices_texts = [text for text in choices_texts if len(text) < 100]
choices_fonts = [] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Augie.ttf', size=15)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Augie.ttf', size=18)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Augie.ttf', size=20)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/IndieFlower.ttf', size=15)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/IndieFlower.ttf', size=18)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/IndieFlower.ttf', size=20)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Komika_Text.ttf', size=15)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Komika_Text.ttf', size=18)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Komika_Text.ttf', size=20)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Cool Cat.ttf', size=16)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Cool Cat.ttf', size=18)] + \
                [ImageFont.truetype(generator_input_dir + '/fonts/Cool Cat.ttf', size=21)]

generator = MangaGenerator.create(
    choices_drawings=choices_drawings,
    choices_texts=choices_texts,
    choices_fonts=choices_fonts,
    choices_text_counts=[4, 5, 5, 6, 6],
    random_salt='')

create_dataset(
    generator=generator,
    dataset_dir=current_module_dir + '/../data/output/generate_manga_dataset',
    output_size=Size.of(750, 1500),
    output_count=500
)
