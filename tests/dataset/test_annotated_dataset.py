from manga_ocr.dataset.annotated_manga import load_line_annotated_dataset
from manga_ocr.utils import get_path_example_dir


def test_load_example_dataset():

    example_dataset_dir = get_path_example_dir('manga_annotated')
    dataset = load_line_annotated_dataset(example_dataset_dir)

    assert len(dataset) == 1

    image, lines = dataset[0]
    assert image.size == (707, 1000)
    assert lines[0].text == 'DEPRESSION'


