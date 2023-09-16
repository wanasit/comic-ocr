import numpy as np
import torch

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


def test_repeat_and_shuffle_dataset():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(dataset_dir)

    assert len(dataset) == 5
    assert dataset.get_image(0).size == (707, 1000)
    assert dataset.get_image(4).size == (800, 600)

    dataset = dataset.repeat(3)
    assert len(dataset) == 15
    assert dataset.get_image(0).size == (707, 1000)
    assert dataset.get_image(4).size == (800, 600)
    assert dataset.get_image(5).size == (707, 1000)
    assert dataset.get_image(9).size == (800, 600)

    dataset = dataset.shuffle(random_seed=123)
    assert len(dataset) == 15
    assert dataset.get_image(0).size == (800, 600)
    assert dataset.get_image(1).size == (1024, 1446)


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

    train_dataloader = dataset.loader(batch_size=2, shuffle=False, num_workers=1)
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


def test_shared_random_seed_for_cropping():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset_a = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), random_seed="test", enable_color_jitter=False)
    dataset_b = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), random_seed="test", enable_color_jitter=False)

    assert len(dataset_a) == 3
    assert len(dataset_b) == 3

    train_dataloader_a = dataset_a.loader(batch_size=2, shuffle=False, num_workers=1)
    train_dataloader_b = dataset_b.loader(batch_size=2, shuffle=False, num_workers=1)
    batch_a = next(iter(train_dataloader_a))
    batch_b = next(iter(train_dataloader_b))

    assert torch.eq(batch_a['input'], batch_b['input']).all()
    assert torch.eq(batch_a['output_mask_char'], batch_b['output_mask_char']).all()
    assert torch.eq(batch_a['output_mask_line'], batch_b['output_mask_line']).all()
    assert torch.eq(batch_a['output_mask_paragraph'], batch_b['output_mask_paragraph']).all()


def test_color_jitter():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset_no_jitter = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), random_seed="test", enable_color_jitter=False)
    dataset_with_jitter = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), random_seed="test", enable_color_jitter=True,
        color_jitter_brightness=.5, color_jitter_hue=.3)

    train_dataloader_no_jitter = dataset_no_jitter.loader(batch_size=2, shuffle=False, num_workers=1)
    train_dataloader_with_jitter = dataset_with_jitter.loader(batch_size=2, shuffle=False, num_workers=1)

    batch_no_jitter = next(iter(train_dataloader_no_jitter))
    batch_with_jitter = next(iter(train_dataloader_with_jitter))

    assert torch.not_equal(batch_no_jitter['input'], batch_with_jitter['input']).any()
    assert torch.eq(batch_no_jitter['output_mask_char'], batch_with_jitter['output_mask_char']).all()
    assert torch.eq(batch_no_jitter['output_mask_line'], batch_with_jitter['output_mask_line']).all()
    assert torch.eq(batch_no_jitter['output_mask_paragraph'], batch_with_jitter['output_mask_paragraph']).all()


def test_load_dataset_with_padding_augmentation():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), choices_padding_width=[1])
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

    # check if the input is really padded
    assert batch['input'][0][:, 0:1, :].sum() == 0
    assert batch['input'][0][:, :, 0:1].sum() == 0
    assert batch['input'][0][:, -1:0, :].sum() == 0
    assert batch['input'][0][:, :, -1:0].sum() == 0
    assert batch['input'][0][:, 0:2, :].sum() > 0
    assert batch['input'][0][:, :, 0:2].sum() > 0


def test_assign_dataset_with_padding_augmentation():
    dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        dataset_dir, batch_image_size=Size.of(500, 500), choices_padding_width=[1])
    dataset = dataset.with_choices_padding_width([5])
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

    # check if the input is really padded
    assert batch['input'][0][:, 0:5, :].sum() == 0
    assert batch['input'][0][:, :, 0:5].sum() == 0
    assert batch['input'][0][:, -5:0, :].sum() == 0
    assert batch['input'][0][:, :, -5:0].sum() == 0
    assert batch['input'][0][:, 0:10, :].sum() > 0
    assert batch['input'][0][:, :, 0:10].sum() > 0


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


def test_merge_duplicated_datasets():
    dataset_annotated = LocalizationDatasetWithAugmentation.load_line_annotated_manga_dataset(
        get_path_example_dir('manga_annotated'), batch_image_size=Size.of(500, 400))

    dataset = LocalizationDatasetWithAugmentation.merge(dataset_annotated, dataset_annotated, dataset_annotated)
    assert len(dataset) == len(dataset_annotated) * 3

    train_dataloader = dataset.loader(batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert 'output_mask_paragraph' not in batch
    assert batch['input'].shape == (2, 3, 400, 500)
    assert batch['output_mask_line'].shape == (2, 400, 500)
    assert batch['output_mask_char'].shape == (2, 400, 500)
