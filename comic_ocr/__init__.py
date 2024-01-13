from typing import List, Tuple, Optional

from comic_ocr.model import ComicOCRModel
from comic_ocr.types import Rectangle, Paragraph, Line, ImageInput, to_image_rgb

_default_instance: Optional[ComicOCRModel] = None


def get_default_model(show_download_progress=True, force_reload=False):
    global _default_instance
    if not _default_instance:
        _default_instance = ComicOCRModel.download_default(
            show_download_progress=show_download_progress,
            force_reload=force_reload)
    return _default_instance


def read_paragraphs(image: ImageInput) -> List[Paragraph]:
    return get_default_model().read_paragraphs(image)


def read_lines(image: ImageInput) -> List[Line]:
    return get_default_model().read_lines(image)


def localize_paragraphs(image: ImageInput) -> List[Tuple[Rectangle, List[Rectangle]]]:
    return get_default_model().localize_paragraphs(image)


def localize_lines(image: ImageInput) -> List[Rectangle]:
    return get_default_model().localize_lines(image)

if __name__ == '__main__':
    from comic_ocr.utils.files import load_image, get_path_project_dir
    from comic_ocr.utils import image_with_annotations

    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    example = load_image(get_path_project_dir('example/manga_annotated/normal_02.jpg'))
    # example = load_image(get_path_project_dir('data/manga_line_annotated/u_01.jpg'))

    example_paragraphs = read_paragraphs(example)
    example_lines = sum([p.lines for p in example_paragraphs], [])
    print(example_paragraphs)

    image_with_annotations(example, example_paragraphs, annotation_text_br_offset_y=-20).show()
    image_with_annotations(example, example_lines, annotation_text_br_offset_y=-20).show()
