from manga_ocr.dataset.annotated_manga import load_line_annotated_dataset
from manga_ocr.utils import get_path_example_dir


def test_load_dataset():

    example_annotated_dataset = get_path_example_dir() + '/annotated_manga'
    dataset = load_line_annotated_dataset(example_annotated_dataset)

    assert len(dataset) == 1

    image, lines = dataset[0]
    assert image.size == (707, 1000)
    assert lines[0].text == 'DEPRESSION'


