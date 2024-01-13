from comic_ocr.model import ComicOCRModel, ModelAccuracy
from comic_ocr.utils import files
from comic_ocr.dataset import annotated_manga
from comic_ocr.types import Line, Rectangle


def test_model_download_default():
    model = ComicOCRModel.download_default(show_download_progress=False)
    assert model

    image = files.load_image(files.get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    paragraphs = model.read_paragraphs(image)
    assert paragraphs

    lines = model.read_lines(image)
    assert lines


def test_model_accuracy_compute():
    model = ComicOCRModel.download_default(show_download_progress=False)
    assert model

    dataset_dir = files.get_path_project_dir('example/manga_annotated')
    dataset = annotated_manga.load_line_annotated_dataset(dataset_dir)
    assert dataset

    accuracy = ModelAccuracy.compute(model, dataset)
    assert accuracy
    assert accuracy.line_precision >= 0
    assert accuracy.line_recall >= 0


def test_model_accuracy_calculation_on_perfect_output():
    accuracy = ModelAccuracy()

    accuracy.include(
        [Line.of('abc', Rectangle.of_size((15, 5), at=(10, 10)))],
        [Line.of('abc', Rectangle.of_size((15, 5), at=(10, 10)))],
    )

    assert accuracy.line_precision == (1 / 1)
    assert accuracy.line_recall == (1 / 1)
    assert accuracy.recognition_accuracy == (1 / 1)
    assert accuracy.localization_precision == (1 / 1)
    assert accuracy.localization_recall == (1 / 1)


def test_model_accuracy_calculation_on_partially_correct_output():
    accuracy = ModelAccuracy()

    rectangles = [Rectangle.of_size((35, 5), at=(10, 10)),
                  Rectangle.of_size((35, 5), at=(10, 100)),
                  Rectangle.of_size((35, 5), at=(10, 200))]

    accuracy.include(
        [Line.of('matched (but wrong text)', rectangles[0]), Line.of('unmatch predicted', rectangles[1])],
        [Line.of('matched', rectangles[0]), Line.of('unmatch annotated', rectangles[2])],
    )

    assert accuracy.line_precision == (0 / 2)
    assert accuracy.line_recall == (0 / 2)
    assert accuracy.recognition_accuracy == (0 / 1)
    assert accuracy.localization_precision == (1 / 2)
    assert accuracy.localization_recall == (1 / 2)
