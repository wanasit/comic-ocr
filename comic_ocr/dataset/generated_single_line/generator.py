from dataclasses import dataclass
from random import Random
from typing import Sequence, Optional, Any, Tuple

from PIL import ImageFont, Image

from comic_ocr.types import Rectangle, Point, Size, Line, Drawable, Color, to_draw
from comic_ocr.utils import examples


@dataclass
class SingleLineGenerator:
    choices_texts: Sequence[str]
    choices_fonts: Sequence[ImageFont.ImageFont]
    choices_padding: Sequence[int]
    choices_background_colors: Sequence[Color]
    choices_text_colors: Sequence[Color]

    random_salt: str = ''
    current_random_seed: float = 0

    @staticmethod
    def create(
            choices_texts: Optional[Sequence[str]] = None,
            choices_fonts: Optional[Sequence[ImageFont.ImageFont]] = None,
            choices_padding: Optional[Sequence[int]] = None,
            choices_background_colors: Optional[Sequence[Color]] = None,
            choices_text_colors: Optional[Sequence[Color]] = None,
            random_salt: str = ''
    ):
        choices_texts = choices_texts if choices_texts else examples.load_example_texts()
        choices_fonts = choices_fonts if choices_fonts else examples.load_example_fonts()
        choices_padding = choices_padding if choices_padding else (1, 2, 3, 3, 4, 4, 5, 5)
        choices_background_colors = choices_background_colors if choices_background_colors else ('#ffffff',)
        choices_text_colors = choices_text_colors if choices_text_colors else ('#000000',)
        return SingleLineGenerator(
            choices_texts=choices_texts,
            choices_fonts=choices_fonts,
            choices_padding=choices_padding,
            choices_background_colors=choices_background_colors,
            choices_text_colors=choices_text_colors,
            random_salt=random_salt
        )

    def generate(
            self,
            random_seed: Optional[Any] = None,
    ) -> Tuple[Image.Image, str]:
        if not random_seed:
            random_seed = self.current_random_seed

        random = Random(f'{self.random_salt}_{random_seed}')
        self.current_random_seed = random.random()

        return generate(
            random,
            choices_texts=self.choices_texts,
            choices_fonts=self.choices_fonts,
            choices_padding=self.choices_padding,
            choices_background_colors=self.choices_background_colors,
            choices_text_colors=self.choices_text_colors,
        )


def generate(
        r: Random,
        choices_texts: Sequence[str],
        choices_fonts: Sequence[ImageFont.ImageFont],
        choices_padding: Sequence[int],
        choices_background_colors: Sequence[Color],
        choices_text_colors: Sequence[Color],
) -> Tuple[Image.Image, str]:
    text = r.choice(choices_texts)
    font = r.choice(choices_fonts)
    padding = r.choice(choices_padding)
    background_color = r.choice(choices_background_colors)
    text_color = r.choice(choices_text_colors)
    ascent, descent = font.getmetrics()
    width, height = font.getsize(text)

    output_size = Size.of(width + padding * 2, height + descent + padding * 2)

    image: Image.Image = Image.new('RGB', output_size, background_color)
    draw = to_draw(image)
    draw.text((padding, padding), text, font=font, fill=text_color)

    return image, text


if __name__ == "__main__":
    generator = SingleLineGenerator.create(random_salt='xyz')
    image, text = generator.generate()

    image.show()
