from dataclasses import dataclass, field
from typing import List, Optional

from PIL import ImageFont

from manga_ocr.typing import Rectangle, Point, Size, Line, Drawable, Color, to_draw


@dataclass
class TextArea:
    xy: Point = field()
    text: str = field()

    font: ImageFont = field(default=ImageFont.load_default(), repr=False)
    text_fill: str = field(default='#000000', repr=False)
    text_line_space: int = field(default=4, repr=False)

    max_width: Optional[float] = field(default=None, repr=False)
    max_height: Optional[float] = field(default=None, repr=False)

    @property
    def wrapped_text(self):
        if self.max_width and self.max_width > 0:
            return '\n'.join(_text_wrap(self.text, font=self.font, max_width=self.max_width))
        return self.text

    def get_text_rect(self) -> Rectangle:
        size = self.get_text_size()
        return Rectangle.of_size(size, at=self.xy)

    def get_text_size(self) -> Size:
        ascent, descent = self.font.getmetrics()
        width, height = self.font.getsize_multiline(self.wrapped_text, spacing=self.text_line_space)
        return Size.of(width, height + descent)

    def get_lines(self, padding_ascent: int = 3, padding_decent: int = 3) -> List[Line]:
        lines = self.wrapped_text.split("\n")
        rect = self.get_text_rect()

        font_height_ascent, font_height_descent = self.font.getmetrics()
        _, x_y1, _, x_y2 = self.font.getbbox("x")
        font_x_height = x_y2 - x_y1
        text_height = font_x_height + padding_ascent + padding_decent

        text_blob_height = rect.height - font_height_descent - self.text_line_space * (len(lines) - 1)
        line_height = text_blob_height / len(lines) + self.text_line_space

        line_top = rect.top + (font_height_ascent - text_height + padding_decent)

        rects = []
        for i, line in enumerate(lines):
            rects.append(Line.of(text=line, at=Rectangle.of_xywh(
                x=rect.left,
                y=line_top + i * line_height,
                w=self.font.getsize(line)[0],
                h=text_height,
            )))
        return rects

    def draw(self, image: Drawable):
        draw = to_draw(image)
        self.draw_background(draw)
        self.draw_text(draw)
        return self.get_text_rect()

    def draw_text(self, image: Drawable, text_fill: Optional[Color] = None):
        draw = to_draw(image)
        text_fill = text_fill if text_fill else self.text_fill
        draw.multiline_text(self.xy, text=self.wrapped_text, font=self.font, fill=text_fill,
                            spacing=self.text_line_space)

    def draw_line_rects(self, image: Drawable, line_fill: Optional[Color] = None):
        draw = to_draw(image)
        line_fill = line_fill if line_fill else self.text_fill
        for line_location in self.get_lines():
            draw.rectangle(line_location.location, fill=line_fill)

    def draw_background(self, image: Drawable):
        pass


def _text_wrap(text: str, font: ImageFont, max_width: float) -> List[str]:
    lines = []
    for line in text.split('\n'):
        lines += _line_wrap(line, font, max_width)
    return lines


def _line_wrap(line: str, font: ImageFont, max_width: float) -> List[str]:
    lines = []
    remaining = line
    while remaining:
        line = remaining
        remaining = ''

        while font.getsize(line)[0] > max_width:
            index = line.rfind(' ', 0, len(line))
            if index < 0:
                # no white space, force one character at a time
                index = len(line) - 1

            remaining = line[index:] + remaining
            line = line[:index]

        if len(line) == 0:
            raise ValueError(f'Given max_width={max_width} is too small for font size={font.size} of "{remaining}"')

        lines.append(line)
        remaining = remaining.strip()
    return lines
