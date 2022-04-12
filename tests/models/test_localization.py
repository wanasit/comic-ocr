from torch.utils.data import DataLoader

from manga_ocr.dataset import generated_manga
from manga_ocr.models.localization import divine_rect_into_overlapping_tiles
from manga_ocr.models.localization.conv_unet.conv_unet import ConvUnet
from manga_ocr.models.localization.localization_dataset import LocalizationDataset
from manga_ocr.models.localization.localization_model import LocalizationModel, image_to_input_tensor
from manga_ocr.models.localization.train import train
from manga_ocr.typing import Size, Rectangle
from manga_ocr.utils import get_path_example_dir


def test_loading_generated_manga_dataset(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    generated_manga.create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = LocalizationDataset.load_generated_manga_dataset(dataset_dir, image_size=Size.of(500, 500))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['image'].shape == (2, 3, 500, 500)
    assert batch['mask_char'].shape == (2, 1, 500, 500)
    assert batch['mask_line'].shape == (2, 1, 500, 500)


def test_loading_generated_manga_dataset_with_resize(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    generated_manga.create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = LocalizationDataset.load_generated_manga_dataset(dataset_dir, image_size=Size.of(200, 200))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['image'].shape == (2, 3, 200, 200)
    assert batch['mask_char'].shape == (2, 1, 200, 200)
    assert batch['mask_line'].shape == (2, 1, 200, 200)


def test_divine_rect_into_overlapping_tiles():
    tiles = divine_rect_into_overlapping_tiles(
        rect=Size.of(100, 100), tile_size=Size.of(50, 50), min_overlap_x=10, min_overlap_y=10)

    assert list(tiles) == [
        Rectangle.of_size((50, 50), at=(0, 0)),
        Rectangle.of_size((50, 50), at=(25, 0)),
        Rectangle.of_size((50, 50), at=(50, 0)),
        Rectangle.of_size((50, 50), at=(0, 25)),
        Rectangle.of_size((50, 50), at=(25, 25)),
        Rectangle.of_size((50, 50), at=(50, 25)),
        Rectangle.of_size((50, 50), at=(0, 50)),
        Rectangle.of_size((50, 50), at=(25, 50)),
        Rectangle.of_size((50, 50), at=(50, 50)),
    ]


def test_divine_rect_into_overlapping_tiles_large():
    tiles = divine_rect_into_overlapping_tiles(
        rect=Size.of(100, 100), tile_size=Size.of(100, 100), min_overlap_x=10, min_overlap_y=10)
    assert list(tiles) == [Rectangle.of_size((100, 100))]

    tiles = divine_rect_into_overlapping_tiles(
        rect=Size.of(100, 100), tile_size=Size.of(120, 120), min_overlap_x=10, min_overlap_y=10)
    assert list(tiles) == [Rectangle.of_size((120, 120), at=(0, 0))]


def test_divine_rect_into_overlapping_tiles_real_cases():
    tiles = divine_rect_into_overlapping_tiles(
        rect=Size.of(750, 1000), tile_size=Size.of(750, 750), min_overlap_x=750 // 4, min_overlap_y=750 // 4)
    assert list(tiles) == [(0, 0, 750, 750), (0, 250, 750, 1000)]

    tiles = divine_rect_into_overlapping_tiles(
        rect=Size.of(750, 1500), tile_size=Size.of(750, 750), min_overlap_x=750 // 4, min_overlap_y=750 // 4)
    assert list(tiles) == [(0, 0, 750, 750), (0, 375, 750, 1125), (0, 750, 750, 1500)]


def test_conv_unet_loss():
    model = ConvUnet()
    assert model.image_size == (750, 750)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_manga_dataset(example_generated_dataset_dir,
                                                               image_size=model.image_size)

    input_image = image_to_input_tensor(dataset.images[0]).unsqueeze(0)
    output_char, output_line = model(input_image)
    assert output_char.shape == (1, 1, 750, 750)
    assert output_line.shape == (1, 1, 750, 750)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    assert loss.item() > 0


def test_conv_unet_training():
    model = ConvUnet()
    assert model.image_size == (750, 750)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_manga_dataset(example_generated_dataset_dir,
                                                               image_size=model.image_size)

    train(model, training_dataset=dataset, tqdm_disable=True)
