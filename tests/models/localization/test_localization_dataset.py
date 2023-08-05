import numpy as np

from comic_ocr.dataset import generated_manga
from comic_ocr.models.localization.localization_dataset import LocalizationDataset, LocalizationDatasetWithAugmentation
from comic_ocr.types import Size
from comic_ocr.utils.files import get_path_example_dir


def test_load_line_annotated_manga_dataset():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(dataset_dir)
    assert len(dataset) == 5
    assert dataset.get_image(0).size == (707, 1000)
    assert dataset.get_mask_char(0).size == (707, 1000)
    assert dataset.get_mask_line(0).size == (707, 1000)
    assert len(dataset.get_line_locations(0)) > 0

    data_loader = dataset.loader()
    batch = next(iter(data_loader))

    assert batch['input'].shape == (1, 3, 1000, 707)
    assert batch['output_mask_line'].shape == (1, 1000, 707)
    assert batch['output_mask_char'].shape == (1, 1000, 707)
    assert set(np.unique(batch['output_mask_line'].numpy())) == {0.0, 1.0}
    assert set(np.unique(batch['output_mask_char'].numpy())) == {0.0, 1.0}


def test_load_generated_manga_dataset():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_manga_dataset(dataset_dir)
    assert len(dataset) == 3
    assert dataset.get_image(0).size == (768, 768)
    assert dataset.get_mask_char(0).size == (768, 768)
    assert dataset.get_mask_line(0).size == (768, 768)

    train_dataloader = dataset.loader()
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (1, 3, 768, 768)
    assert batch['output_mask_char'].shape == (1, 768, 768)
    assert batch['output_mask_line'].shape == (1, 768, 768)
    assert batch['output_mask_paragraph'].shape == (1, 768, 768)


def test_load_line_annotated_manga_dataset_with_augmentation():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(dataset_dir,
                                                                                    batch_image_size=Size.of(400, 500))
    assert len(dataset) == 5
    assert dataset.get_image(0).size == (707, 1000)
    assert dataset.get_mask_char(0).size == (707, 1000)
    assert dataset.get_mask_line(0).size == (707, 1000)
    assert len(dataset.get_line_locations(0)) > 0

    # dataset.get_mask_char(0).show()

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 500, 400)
    assert batch['output_mask_line'].shape == (2, 500, 400)
    assert batch['output_mask_char'].shape == (2, 500, 400)
    assert set(np.unique(batch['output_mask_line'].numpy())) == {0.0, 1.0}
    assert set(np.unique(batch['output_mask_char'].numpy())) == {0.0, 1.0}


def test_load_generated_manga_dataset_with_augmentation():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(dataset_dir,
                                                                               batch_image_size=Size.of(500, 500))
    assert len(dataset) == 3
    assert dataset.get_image(0).size == (768, 768)
    assert dataset.get_mask_char(0).size == (768, 768)
    assert dataset.get_mask_line(0).size == (768, 768)

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 500, 500)
    assert batch['output_mask_char'].shape == (2, 500, 500)
    assert batch['output_mask_line'].shape == (2, 500, 500)
    assert batch['output_mask_paragraph'].shape == (2, 500, 500)


def test_load_and_resize_generated_manga_dataset_with_augmentation(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    generated_manga.create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(dataset_dir,
                                                                               batch_image_size=Size.of(200, 200))
    dataset = dataset.shuffle()
    assert len(dataset) > 0

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 200, 200)
    assert batch['output_mask_char'].shape == (2, 200, 200)
    assert batch['output_mask_line'].shape == (2, 200, 200)
    assert batch['output_mask_paragraph'].shape == (2, 200, 200)


def test_resize_dataset_with_augmentation(tmpdir):
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(dataset_dir,
                                                                                    batch_image_size=Size.of(400, 500))
    assert len(dataset) == 5

    dataset = dataset.with_batch_image_size(Size.of(500, 500))
    assert len(dataset) == 5
    assert dataset.get_image(0).size == (707, 1000)
    assert dataset.get_mask_char(0).size == (707, 1000)
    assert dataset.get_mask_line(0).size == (707, 1000)
    assert len(dataset.get_line_locations(0)) > 0

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 500, 500)
    assert batch['output_mask_char'].shape == (2, 500, 500)
    assert batch['output_mask_line'].shape == (2, 500, 500)


def test_merge_annotated_generated_dataset_with_augmentation():
    dataset_annotated = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(
        get_path_example_dir('manga_annotated'), batch_image_size=Size.of(500, 500))

    dataset_generated = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(
        get_path_example_dir('manga_annotated'), batch_image_size=Size.of(500, 500))

    dataset = LocalizationDatasetWithAugmentation.merge(dataset_generated, dataset_annotated)
    dataset = dataset.shuffle()
    assert len(dataset) > 0
    assert len(dataset) == len(dataset_generated) + len(dataset_annotated)

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert 'output_mask_paragraph' not in batch
    assert batch['input'].shape == (2, 3, 500, 500)
    assert batch['output_mask_line'].shape == (2, 500, 500)
    assert batch['output_mask_char'].shape == (2, 500, 500)


def test_merge_datasets_with_different_batch_size():
    dataset_annotated = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(
        get_path_example_dir('manga_annotated'), batch_image_size=Size.of(500, 400))

    dataset_generated = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(
        get_path_example_dir('manga_annotated'), batch_image_size=Size.of(400, 500))

    dataset = LocalizationDatasetWithAugmentation.merge(dataset_generated, dataset_annotated)
    dataset = dataset.shuffle()
    assert len(dataset) > 0
    assert len(dataset) == len(dataset_generated) + len(dataset_annotated)

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert 'output_mask_paragraph' not in batch
    assert batch['input'].shape == (2, 3, 400, 400)
    assert batch['output_mask_line'].shape == (2, 400, 400)
    assert batch['output_mask_char'].shape == (2, 400, 400)
