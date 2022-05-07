import manga_ocr

from manga_ocr.typing import Rectangle
from manga_ocr.utils import image_with_annotations
from manga_ocr.utils.files import load_image, get_path_project_dir


def test_localize_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    paragraphs = manga_ocr.localize_paragraphs(image)
    assert len(paragraphs) == 3

    paragraph_locations = [l for l, _ in paragraphs]
    # image_with_annotations(image, paragraph_locations).show()
    assert paragraph_locations[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert paragraph_locations[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert paragraph_locations[2].can_represent(Rectangle.of_size((108, 28), at=(473, 945)))


def test_locate_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    line_locations = manga_ocr.localize_lines(image)
    # image_with_annotations(image, line_locations).show()
    assert len(line_locations) == 4

    line_locations = sorted(line_locations, key=lambda l: l.top)

    assert line_locations[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert line_locations[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert line_locations[2].can_represent(Rectangle.of_size((42, 14), at=(506, 945)))
    assert line_locations[3].can_represent(Rectangle.of_size((108, 12), at=(473, 961)))