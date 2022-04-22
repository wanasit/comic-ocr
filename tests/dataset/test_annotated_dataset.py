from manga_ocr.dataset.annotated_manga import load_line_annotated_dataset
from manga_ocr.utils.files import get_path_example_dir


def test_load_example_dataset():

    example_dataset_dir = get_path_example_dir('manga_annotated')
    images, image_texts = load_line_annotated_dataset(example_dataset_dir)

    assert len(images) == 1
    assert len(image_texts) == 1

    image, lines = images[0], image_texts[0]
    assert image.size == (707, 1000)
    assert lines[0].text == 'DEPRESSION'


