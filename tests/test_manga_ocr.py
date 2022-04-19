import manga_ocr

from manga_ocr.utils import image_with_annotations
from manga_ocr.utils.files import load_image, get_path_project_dir


def test_locate_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    line_locations = manga_ocr.localize_lines(image)

    assert len(line_locations) == 4
