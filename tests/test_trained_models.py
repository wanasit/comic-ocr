from comic_ocr import Rectangle

from comic_ocr.models import localization
from comic_ocr.models import recognition
from comic_ocr.types import Size
from comic_ocr.utils.files import load_image, get_path_project_dir
from comic_ocr.utils import image_with_annotations


def test_trained_localization_model_high_level_metrics():
    model = localization.load_model()
    assert model

    dataset_dir = get_path_project_dir('example/manga_annotated')
    dataset = localization.LocalizationDataset.load_line_annotated_manga_dataset(dataset_dir)
    assert dataset

    metrics = localization.calculate_high_level_metrics(model, dataset)
    # print(metrics)
    assert metrics
    assert metrics["line_level_precision"] > 0.5
    assert metrics["line_level_recall"] > 0.5


def test_trained_model_locate_paragraphs_in_example():
    model = localization.load_model()
    assert model

    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    paragraphs = model.locate_paragraphs(image)
    paragraph_locations = [l for l, _ in paragraphs]
    # image_with_annotations(image, paragraph_locations).show()
    assert len(paragraph_locations) == 3
    assert paragraph_locations[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert paragraph_locations[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert paragraph_locations[2].can_represent(Rectangle.of_size((108, 28), at=(473, 945)))


def test_trained_model_locate_lines_in_example():
    model = localization.load_model()
    assert model

    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    lines = model.locate_lines(image)
    # image_with_annotations(image, line_locations).show()
    assert len(lines) == 4
    lines = sorted(lines, key=lambda l: l.top)
    assert lines[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert lines[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert lines[2].can_represent(Rectangle.of_size((42, 14), at=(506, 945)))
    assert lines[3].can_represent(Rectangle.of_size((108, 12), at=(473, 961)))


def test_trained_models_read_lines_in_example():
    localization_model = localization.load_model()
    recognition_model = recognition.load_model()

    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    lines = localization_model.locate_lines(image)
    # image_with_annotations(image, line_locations).show()

    assert len(lines) == 4
    lines = sorted(lines, key=lambda l: l.top)
    assert lines[0].can_represent(Rectangle.of_size((87, 16), at=(309, 459)))
    assert lines[1].can_represent(Rectangle.of_size((95, 20), at=(333, 715)))
    assert lines[2].can_represent(Rectangle.of_size((42, 14), at=(506, 945)))
    assert lines[3].can_represent(Rectangle.of_size((108, 12), at=(473, 961)))

    # Test recognition
    assert recognition_model.recognize(image.crop(lines[0])) == 'DEPRESSION'
    assert recognition_model.recognize(image.crop(lines[1])) == 'ACCEPTANCE'
