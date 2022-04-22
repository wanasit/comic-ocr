from PIL.Image import Image

import manga_ocr.dataset.generated_manga as generated_manga
from manga_ocr.utils.files import get_path_example_dir


def test_manga_generator():
    generator = generated_manga.MangaGenerator.create()
    image, text_areas = generator.generate(output_size=(750, 750))

    assert image.size == (750, 750)
    assert 3 < len(text_areas) < 8


def test_load_example_dataset():
    example_dataset_dir = get_path_example_dir('manga_generated')
    images, image_texts, image_masks = generated_manga.load_dataset(example_dataset_dir)
    assert len(images) == 3
    assert len(image_texts) == 3
    assert len(image_masks) == 3

    assert isinstance(images[0], Image)
    assert isinstance(image_masks[0], Image)

    assert len(image_texts[0]) > 0


def test_generate_dataset(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=2)

    images, image_texts, image_masks = generated_manga.load_dataset(dataset_dir)
    assert len(images) == 2
    assert len(image_texts) == 2
    assert len(image_masks) == 2

    assert isinstance(images[0], Image)
    assert isinstance(image_masks[0], Image)

    assert len(image_texts[0]) > 0


def test_generate_on_the_same_location(tmpdir):
    dataset_dir = tmpdir / 'test_generate_dataset'

    generated_manga.create_dataset(dataset_dir, output_count=3)
    generated_manga.create_dataset(dataset_dir, output_count=4)
