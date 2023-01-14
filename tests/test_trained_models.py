import comic_ocr
from comic_ocr import Rectangle

from comic_ocr.models import localization
from comic_ocr.types import Size
from comic_ocr.utils.files import load_image, get_path_project_dir


def test_trained_localization_model_high_level_metrics():
    model = localization.load_model()
    assert model

    dataset_dir = get_path_project_dir('example/manga_annotated')
    dataset = localization.LocalizationDataset.load_line_annotated_manga_dataset(
        dataset_dir, image_size=Size.of(500, 500))
    assert dataset

    metrics = localization.calculate_high_level_metrics(model, dataset)
    # print(metrics)
    assert metrics
    assert metrics["line_level_precision"] > 0.5
    assert metrics["line_level_recall"] > 0.5


def test_trained_model_localize_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    paragraphs = comic_ocr.localize_paragraphs(image)
    assert len(paragraphs) == 3

    paragraph_locations = [l for l, _ in paragraphs]
    # image_with_annotations(image, paragraph_locations).show()
    assert paragraph_locations[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert paragraph_locations[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert paragraph_locations[2].can_represent(Rectangle.of_size((108, 28), at=(473, 945)))


def test_trained_models_read_paragraphs():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    paragraphs = comic_ocr.read_paragraphs(image)
    # image_with_annotations(image, line_locations).show()
    assert len(paragraphs) == 3

    paragraph_locations = [p.location for p in paragraphs]
    # image_with_annotations(image, paragraphs).show()
    assert paragraph_locations[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert paragraph_locations[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert paragraph_locations[2].can_represent(Rectangle.of_size((108, 28), at=(473, 945)))

    paragraph_texts = [p.text for p in paragraphs]
    # image_with_annotations(image, paragraphs).show()
    assert paragraph_texts[0] == 'DEPRESSION'
    assert paragraph_texts[1] == 'ACCEPTANCE'


def test_trained_models_read_lines():
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    lines = comic_ocr.read_lines(image)
    # image_with_annotations(image, line_locations).show()
    assert len(lines) == 4

    lines = sorted(lines, key=lambda l: l.location.top)

    assert lines[0].location.can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert lines[1].location.can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert lines[2].location.can_represent(Rectangle.of_size((42, 14), at=(506, 945)))
    assert lines[3].location.can_represent(Rectangle.of_size((108, 12), at=(473, 961)))

    assert lines[0].text == 'DEPRESSION'
    assert lines[1].text == 'ACCEPTANCE'
