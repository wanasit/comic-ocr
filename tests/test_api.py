import comic_ocr

from comic_ocr.utils.files import load_image, get_path_project_dir


def test_localize_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    paragraphs = comic_ocr.localize_paragraphs(image)
    assert paragraphs


def test_read_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    paragraphs = comic_ocr.read_paragraphs(image)

    paragraph_locations = [p.location for p in paragraphs]
    assert paragraph_locations

    paragraph_texts = [p.text for p in paragraphs]
    assert paragraph_texts


def test_locate_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    line_locations = comic_ocr.localize_lines(image)
    assert line_locations

    line_locations = sorted(line_locations, key=lambda l: l.top)
    assert line_locations


def test_read_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    lines = comic_ocr.read_lines(image)
    assert lines

    lines = sorted(lines, key=lambda l: l.location.top)
    assert lines
