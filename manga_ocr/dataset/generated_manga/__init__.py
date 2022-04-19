from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image

from manga_ocr.dataset.generated_manga.generator import MangaGenerator
from manga_ocr.dataset.generated_manga.text_area import TextArea
from manga_ocr.typing import Color, Paragraph, Size, Line
from manga_ocr.utils import get_path_project_dir
from manga_ocr.utils.files import load_images_with_annotation, load_images, write_json_dict
from manga_ocr.utils.nb_annotation import lines_to_nb_annotation_data, lines_from_nb_annotation_data

DEFAULT_CHAR_ALPHA = 1.0
DEFAULT_LINE_ALPHA = 0.6


def load_dataset(dataset_dir: str) -> List[Tuple[Image.Image, Image.Image, List[Line]]]:
    path = Path(dataset_dir)
    images, line_annotations = load_images_with_annotation(path / 'image/*.jpg', path / 'line_annotation')
    image_masks = load_images(path / 'image_mask/*.jpg')

    return [(images[i], image_masks[i], lines_from_nb_annotation_data(line_annotations[i])) for i in range(len(images))]


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
    (path / 'line_annotation').mkdir(parents=True, exist_ok=True)

    for i in range(output_count):
        image, text_areas = generator.generate(i, output_size=output_size)
        image_mask = _create_image_mask(image, text_areas)
        lines = [line for t in text_areas for line in t.get_lines()]

        image.save(path / 'image' / '{:04}.jpg'.format(i))
        image_mask.save(path / 'image_mask' / '{:04}.jpg'.format(i))
        write_json_dict(path / 'line_annotation' / '{:04}.json'.format(i), lines_to_nb_annotation_data(lines))


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
    import shutil

    dataset_dir = get_path_project_dir('data/output/generated_manage_test')
    shutil.rmtree(dataset_dir)

    create_dataset(dataset_dir, output_count=3)
    dataset = load_dataset(dataset_dir)

    print(dataset[0])
    print(dataset[1])
    print(dataset[2])