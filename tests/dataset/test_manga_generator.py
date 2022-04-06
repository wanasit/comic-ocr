from PIL.Image import Image

import manga_ocr.dataset.generated_manga as generated_manga
from manga_ocr.utils import get_path_example_dir


def test_manga_generator():
    generator = generated_manga.MangaGenerator.create()
    image, text_areas = generator.generate(output_size=(750, 750))

    assert image.size == (750, 750)
    assert 3 < len(text_areas) < 8


def test_load_example_dataset():
    example_dataset_dir = get_path_example_dir('manga_generated')
    dataset = generated_manga.load_dataset(example_dataset_dir)

    assert len(dataset) == 3

    row = dataset[0]
    assert isinstance(row[0], Image)
    assert isinstance(row[1], Image)
    assert len(row[2]) > 0


def test_generate_dataset(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=4)

    dataset = generated_manga.load_dataset(dataset_dir)
    assert len(dataset) == 4

    row = dataset[0]
    assert isinstance(row[0], Image)
    assert isinstance(row[1], Image)
    assert len(row[2]) > 0


def test_generate_on_the_same_location(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=3)
    generated_manga.create_dataset(dataset_dir, output_count=4)
