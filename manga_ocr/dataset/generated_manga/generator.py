import os
from dataclasses import dataclass
from random import Random
from typing import Optional, Any, List, Tuple

from PIL import Image, ImageDraw
from PIL import ImageFont

from manga_ocr.dataset.generated_manga.text_area import TextArea
from manga_ocr.dataset.generated_manga.text_bubble import TextBubble
from manga_ocr.dataset.generated_manga.text_rect import TextRect
from manga_ocr.typing import Rectangle, Point, Drawable, Size
from manga_ocr.utils import load_images, load_texts


@dataclass
class MangaGenerator():
    choices_drawings: List[Image.Image]
    choices_texts: List[str]
    choices_fonts: List[ImageFont.ImageFont]
    choices_text_counts: List[int]

    random_salt: str = ''
    current_random_seed: float = 0
    output_size: Size = Size.of(768, 768)

    @staticmethod
    def create(
            choices_drawings: Optional[List[Image.Image]] = None,
            choices_texts: Optional[List[str]] = None,
            choices_fonts: Optional[List[ImageFont.ImageFont]] = None,
            choices_text_counts: Optional[List[int]] = None,
            random_salt: str = ''
    ):
        choices_drawings = choices_drawings if choices_drawings else load_example_drawing()
        choices_texts = choices_texts if choices_texts else load_example_texts()
        choices_fonts = choices_fonts if choices_fonts else load_example_fonts()
        choices_text_counts = choices_text_counts if choices_text_counts else (5, 6)
        return MangaGenerator(
            choices_drawings=choices_drawings,
            choices_texts=choices_texts,
            choices_fonts=choices_fonts,
            choices_text_counts=choices_text_counts,
            random_salt=random_salt
        )

    def generate(self, random_seed: Optional[Any] = None, output_size: Optional[Size] = None):
        if not random_seed:
            random_seed = self.current_random_seed

        output_size = output_size if output_size else self.output_size
        random = Random(f'{self.random_salt}_{random_seed}')
        self.current_random_seed = random.random()

        return generate(
            random,
            output_size=output_size,
            choices_drawings=self.choices_drawings,
            choices_texts=self.choices_texts,
            choices_fonts=self.choices_fonts,
            choices_text_counts=self.choices_text_counts,
        )


def generate(
        random: Random,
        choices_drawings: List[Image.Image],
        choices_texts: List[str],
        choices_fonts: List[ImageFont.ImageFont],
        choices_text_counts: List[int] = (5,),
        output_size=(768, 768)
) -> Tuple[Image.Image, List[TextArea]]:
    image: Image.Image = Image.new('RGB', output_size, '#ffffff')
    _draw_random_drawing(random, image, choices_drawings)

    text_count = random.choice(choices_text_counts)
    text_areas = _draw_non_overlap_text_areas(
        random, image, text_count, choices_texts=choices_texts, choices_font=choices_fonts)

    return image, text_areas


# ------------------

current_module_dir = os.path.dirname(__file__)
project_root_dir = current_module_dir + '/../../..'


def load_example_fonts() -> List[ImageFont.ImageFont]:
    return \
        [ImageFont.truetype(project_root_dir + '/example/fonts/Komika_Text.ttf', size=15)] + \
        [ImageFont.truetype(project_root_dir + '/example/fonts/Komika_Text.ttf', size=20)] + \
        [ImageFont.truetype(project_root_dir + '/example/fonts/Cool Cat.ttf', size=16)] * 3 + \
        [ImageFont.truetype(project_root_dir + '/example/fonts/Cool Cat.ttf', size=21)]


def load_example_drawing() -> List[Image.Image]:
    return load_images(project_root_dir + '/example/drawings/*.jpg')


def load_example_texts() -> List[str]:
    return load_texts(project_root_dir + '/example/text/texts.txt')


# ------------------


def _draw_random_drawing(
        random: Random,
        draw: Drawable,
        choices_drawings: List[Image.Image],
        padding: int = 5,
        bound: Optional[Rectangle] = None):
    if not bound:
        bound = Rectangle.of_size(draw.size)

    row = bound.top
    while row < bound.bottom:
        i = random.randint(0, len(choices_drawings) - 1)
        main_drawing = choices_drawings[i].copy()

        if main_drawing.width + 2 * padding > bound.width:
            main_drawing = _random_resize_to_width(random, main_drawing, bound.width - 2 * padding)

        draw.paste(main_drawing, (bound.left + padding, row + padding))
        remaining_width = bound.width - padding - main_drawing.width
        if remaining_width > padding + 50:
            sub_random = Random(random.random())
            sub_bound = Rectangle.of_tl_br(tl=(bound.right - remaining_width, row), br=bound.br)
            _draw_random_drawing(sub_random, draw, choices_drawings, padding, bound=sub_bound)

        row += main_drawing.height + padding


def _random_resize_to_width(random: Random, image: Image.Image, width: float):
    # Todo: random resize vs crop, if crop, random crop location
    ratio = (width / float(image.size[0]))
    height = int((float(image.size[1]) * float(ratio)))
    return image.resize((width, height))


def _draw_non_overlap_text_areas(
        random: Random,
        image: Drawable,
        text_count: int,
        choices_texts: List[str],
        choices_font: List[ImageFont.ImageFont],
        max_retry_count=5
) -> List[TextArea]:
    bound = Rectangle.of_size(image.size)
    drawn_rects: List[Rectangle] = []
    output: List[TextArea] = []

    for i in range(text_count):

        attempt = 0
        while attempt < max_retry_count:
            text = random.choice(choices_texts)
            font = random.choice(choices_font)

            text_area = _create_random_text_area(random, bound, text, font)
            text_rect = text_area.get_text_rect()

            if text_rect in bound:
                if not any(rect for rect in drawn_rects if text_rect.is_overlap(rect)):
                    drawn_rects.append(text_rect)
                    output.append(text_area)
                    text_area.draw(image)
                    break
            attempt += 1

        if attempt >= max_retry_count:
            raise ValueError(
                f'Could not generate non-overlap texts after random {max_retry_count} retries. '
                f'Please try different `choices_*` or reduce `text_count`')

    return output


def _create_random_text_area(
        random: Random,
        bound: Rectangle,
        text: str,
        font: ImageFont,
        bubble_to_rect_ratio=2 / 1
) -> TextArea:
    xy = Point.of(
        x=random.randint(bound.left + 10, bound.right - 100),
        y=random.randint(bound.top + 10, bound.bottom - 100))

    width = min(bound.right - xy.x, random.randint(150, 300))

    if random.random() > bubble_to_rect_ratio / (bubble_to_rect_ratio + 1):
        return TextRect(xy, text=text, font=font, max_width=width)
    else:
        return TextBubble(xy, text=text, font=font, max_width=width)


if __name__ == "__main__":
    generator = MangaGenerator.create()
    image, text_areas = generator.generate()
    image.show()

    for text_area in text_areas:
        drw = ImageDraw.Draw(image, 'RGBA')
        text_area.draw_line_rects(drw, line_fill='#ff0f0f7f')

    image.show()
