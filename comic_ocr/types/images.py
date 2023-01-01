from __future__ import annotations

from typing import Union, Tuple

from PIL import Image, ImageDraw

from comic_ocr.types import PathLike

'''
A type representing an input image or image file. 
'''
ImageInput = Union[Image.Image, PathLike]
ImageRGB = Image.Image

Drawable = Union[ImageDraw.Draw, Image.Image]

'''
A type representing a color input parameter, which could be string (e.g. `#ffffff`) or tuple (255, 255, 255) in RGB or 
RGBA form.
'''
Color = Union[str, Tuple[int, int, int, int], Tuple[int, int, int]]


def to_image_rgb(image_input: ImageInput) -> ImageRGB:
    if isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            return image_input.convert('RGB')
        return image_input
    from comic_ocr.utils.files import load_image
    return load_image(image_input)


def to_draw(drawable_input: Drawable) -> ImageDraw.Draw:
    if isinstance(drawable_input, Image.Image):
        return ImageDraw.Draw(drawable_input, 'RGBA')
    return drawable_input
