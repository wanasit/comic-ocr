import manga_ocr

from manga_ocr.utils import image_with_annotations
from manga_ocr.utils.files import load_image, get_path_project_dir


def test_localize_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    paragraphs = manga_ocr.localize_paragraphs(image)
    paragraph_locations = [l for l, _ in paragraphs]

    #assert len(paragraph_locations) == 4
    #image_with_annotations(image, paragraph_locations).show()


def test_locate_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    line_locations = manga_ocr.localize_lines(image)

    # Todo: train a better model
    # assert len(line_locations) == 4
