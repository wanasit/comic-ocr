from __future__ import annotations

from typing import Union, Tuple

from PIL import Image, ImageDraw

Drawable = Union[ImageDraw.Draw, Image.Image]
Color = Union[str, Tuple[int, int, int, int]]


def to_draw(input: Drawable) -> ImageDraw.Draw:
    if isinstance(input, Image.Image):
        return ImageDraw.Draw(input)
    return input
