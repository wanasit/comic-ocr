from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional

from PIL import ImageFont

from comic_ocr.types import Rectangle, Point, Size, Line, Drawable, Color, to_draw


@dataclass
class TextArea:
    xy: Point = field()
    text: str = field()

    padding_x: int = field(default=4, repr=False)
    padding_y: int = field(default=2, repr=False)
    padding_line_ascent: int = field(default=1, repr=False)
    padding_line_decent: int = field(default=1, repr=False)

    font: ImageFont = field(default=ImageFont.load_default(), repr=False)
    text_fill: str = field(default='#000000', repr=False)
    text_line_space: int = field(default=4, repr=False)

    max_width: Optional[float] = field(default=None, repr=False)
    max_height: Optional[float] = field(default=None, repr=False)

    @cached_property
    def wrapped_text_lines(self) -> List[str]:
        if self.max_width and self.max_width > 0:
            return _text_wrap(self.text, font=self.font, max_width=self.max_width)
        return [self.text]

    @cached_property
    def wrapped_text(self):
        return '\n'.join(self.wrapped_text_lines)

    @cached_property
    def text_size(self):
        ascent, descent = self.font.getmetrics()
        width, height = self.font.getsize_multiline(self.wrapped_text, spacing=self.text_line_space)
        return Size.of(width + self.padding_x, height + descent + self.padding_y)

    @cached_property
    def text_rect(self):
        return Rectangle.of_size(self.text_size, at=self.xy)

    # noinspection PyUnresolvedReferences
    @cached_property
    def text_lines(self) -> List[Line]:
        lines = self.wrapped_text_lines
        rect = self.text_rect
        ascent, descent = self.font.getmetrics()
        _, actual_y1, _, actual_y2 = self.font.getbbox("Ay")
        actual_text_height = actual_y2 - actual_y1
        line_height = actual_text_height + self.padding_line_ascent + self.padding_line_decent
        line_top = rect.top + actual_y1 - self.padding_line_ascent + self.padding_y // 2

        text_height_total = rect.height - descent - self.text_line_space * (len(lines) - 1) - self.padding_y
        text_height = text_height_total // len(lines)
        line_space = text_height + self.text_line_space

        rects = []
        for i, line in enumerate(lines):
            rects.append(Line.of(text=line, at=Rectangle.of_xywh(
                x=rect.left, y=line_top + i * line_space,
                w=self.font.getsize(line)[0] + self.padding_x,
                h=line_height,
            )))
        return rects

    def draw(self, image: Drawable):
        draw = to_draw(image)
        self.draw_background(draw)
        self.draw_text(draw)
        return self.text_rect

    def draw_text(self, image: Drawable, fill: Optional[Color] = None):
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        draw.multiline_text(self.xy.move(self.padding_x // 2, self.padding_y // 2),
                            text=self.wrapped_text,
                            font=self.font,
                            fill=fill,
                            spacing=self.text_line_space)

    def draw_text_rect(self, image: Drawable, fill: Optional[Color] = None):
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        draw.rectangle(self.text_rect, fill=fill)

    def draw_line_rects(self, image: Drawable, fill: Optional[Color] = None):
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        for line_location in self.text_lines:
            draw.rectangle(line_location.location, fill=fill)

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
