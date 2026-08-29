from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from comic_ocr.types import Rectangle, Point, Size, Line, Drawable, Color, to_draw

_DUMMY_DRAW = ImageDraw.Draw(Image.new('RGB', (1, 1)))


@dataclass
class TextArea:
    """Represents a rendered text block with wrapping, line bounds, and paragraph bounds.

    Attributes:
        xy: Top-left coordinate (Point) where the text area begins.
        text: Raw text string (may contain newlines).
        padding_x: Horizontal padding added to each line width.
        padding_y: Vertical padding for the overall text area.
        padding_line_ascent: Extra padding above each line's glyph bounding box.
        padding_line_decent: Extra padding below each line's glyph bounding box.
        font: PIL ImageFont used for measurement and rendering.
        text_fill: Color or hex string used for rendering characters.
        text_line_space: Spacing in pixels between lines of text.
        max_width: Maximum pixel width allowed before wrapping lines.
        max_height: Optional maximum height.
    """
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

    @staticmethod
    def text_size(text: str, font: ImageFont.ImageFont, spacing: int = 0) -> Tuple[int, int]:
        """Returns (width, height) in pixels for single-line or multi-line text using font metrics and bounding boxes."""
        if not text:
            return 0, 0
        lines = text.split('\n')
        if len(lines) == 1:
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height
        else:
            bbox = _DUMMY_DRAW.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height

    @cached_property
    def wrapped_text_lines(self) -> List[str]:
        """Returns the list of text lines after applying word-wrapping according to max_width."""
        if self.max_width and self.max_width > 0:
            return _text_wrap(self.text, font=self.font, max_width=self.max_width)
        return [self.text]

    @cached_property
    def wrapped_text(self) -> str:
        """Returns the wrapped text joined by newline characters."""
        return '\n'.join(self.wrapped_text_lines)

    # noinspection PyUnresolvedReferences
    @cached_property
    def text_lines(self) -> List[Line]:
        """Calculates exact Line objects (text and bounding Rectangle) for each wrapped line."""
        lines = self.wrapped_text_lines
        if not lines:
            return []

        first_bbox = self.font.getbbox(lines[0])
        first_y0 = first_bbox[1]
        draw_x = self.xy.x + self.padding_x // 2
        draw_y = self.xy.y + self.padding_line_ascent - first_y0
        line_spacing = _DUMMY_DRAW.textbbox((0, 0), "A", self.font)[3] + self.text_line_space

        rects = []
        for i, line in enumerate(lines):
            if not line:
                continue
            line_w, _ = TextArea.text_size(line, self.font)
            line_y = draw_y + i * line_spacing
            bbox = _DUMMY_DRAW.textbbox((draw_x, line_y), line, font=self.font)

            rects.append(Line.of(text=line, at=Rectangle.of_xywh(
                x=self.xy.x,
                y=bbox[1] - self.padding_line_ascent,
                w=line_w + self.padding_x,
                h=(bbox[3] - bbox[1]) + self.padding_line_ascent + self.padding_line_decent,
            )))
        return rects

    @cached_property
    def text_rect(self) -> Rectangle:
        """The union bounding Rectangle spanning all text lines."""
        if not self.text_lines:
            return Rectangle.of_size((0, 0), at=self.xy)
        return Rectangle.union_bounding_rect([l.location for l in self.text_lines])

    def draw(self, image: Drawable) -> Rectangle:
        """Draws background shape (e.g. bubble or rect) followed by text onto the given image."""
        draw = to_draw(image)
        self.draw_background(draw)
        self.draw_text(draw)
        return self.text_rect

    def draw_text(self, image: Drawable, fill: Optional[Color] = None):
        """Renders the text characters onto the image."""
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        lines = self.wrapped_text_lines
        if not lines:
            return
        first_bbox = self.font.getbbox(lines[0])
        first_y0 = first_bbox[1]
        draw_x = self.xy.x + self.padding_x // 2
        draw_y = self.xy.y + self.padding_line_ascent - first_y0
        draw.multiline_text((draw_x, draw_y),
                            text=self.wrapped_text,
                            font=self.font,
                            fill=fill,
                            spacing=self.text_line_space)

    def draw_text_rect(self, image: Drawable, fill: Optional[Color] = None):
        """Draws a filled rectangle over the overall paragraph text_rect (useful for mask generation)."""
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        draw.rectangle(self.text_rect, fill=fill)

    def draw_line_rects(self, image: Drawable, fill: Optional[Color] = None):
        """Draws filled rectangles over each individual line_rect (useful for mask generation)."""
        draw = to_draw(image)
        fill = fill if fill else self.text_fill
        for line_location in self.text_lines:
            draw.rectangle(line_location.location, fill=fill)

    def draw_background(self, image: Drawable):
        """Hook method overridden by subclasses (e.g., TextBubble, TextRect) to render backgrounds."""
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

        while TextArea.text_size(line, font)[0] > max_width:
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
