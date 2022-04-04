from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image

from manga_ocr.dataset.generated_manga.generator import MangaGenerator
from manga_ocr.dataset.generated_manga.text_area import TextArea
from manga_ocr.typing import Color, Paragraph, Size, Line

DEFAULT_CHAR_ALPHA = 1.0
DEFAULT_LINE_ALPHA = 0.6

def create_dataset(
        dataset_dir: str,
        generator: Optional[MangaGenerator] = None,
        output_size: Optional[Size] = None,
        output_count: Optional[int] = 100,
):
    generator = generator if generator else MangaGenerator.create()

    path = Path(dataset_dir)
    (path / 'image').mkdir(parents=True, exist_ok=True)
    (path / 'image_mask').mkdir(parents=True, exist_ok=True)
    (path / 'paragraph').mkdir(parents=True, exist_ok=True)

    for i in range(output_count):
        image, text_areas = generator.generate(i, output_size=output_size)
        image_mask = _create_image_mask(image, text_areas)
        paragraphs = [Paragraph.of(t.get_lines()) for t in text_areas]

        image.save(path / 'image' / '{:04}.jpg'.format(i))
        image_mask.save(path / 'image_mask' / '{:04}.jpg'.format(i))
        Paragraph.save_paragraphs_to_file(path / 'paragraph' / '{:04}.json'.format(i), paragraphs)

def _create_image_mask(
        image,
        text_areas,
        color_background: Color = (0, 0, 0),
        color_char: Color = (int(255 * DEFAULT_CHAR_ALPHA),) * 3,
        color_line: Color = (int(255 * DEFAULT_LINE_ALPHA),) * 3,
):
    image = Image.new('RGB', image.size, color_background)
    for text_area in text_areas:
        text_area.draw_line_rects(image, line_fill=color_line)
        text_area.draw_text(image, text_fill=color_char)
    return image.convert('RGB')


if __name__ == "__main__":
    create_dataset('../../../data/output/generated_manage_test', output_count=10)
