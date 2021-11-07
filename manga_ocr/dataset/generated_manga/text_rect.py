from dataclasses import dataclass, field

from manga_ocr.dataset.generated_manga.text_area import TextArea
from manga_ocr.typing import Color, Drawable, to_draw


@dataclass
class TextRect(TextArea):
    rect_padding: int = field(default=3)
    rect_fill_color: Color = field(default=(255, 255, 255, 250))
    rect_outline_width: int = field(default=1)
    rect_outline_color: Color = field(default='#000000')

    def draw_background(self, image: Drawable):
        draw = to_draw(image)
        inner_rect = self.get_text_rect()
        outer_rect = inner_rect.expand(self.rect_padding + self.rect_outline_width)
        draw.rectangle(
            xy=outer_rect,
            fill=self.rect_fill_color,
            outline=self.rect_outline_color,
            width=self.rect_outline_width)
