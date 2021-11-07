from torch.utils.data import DataLoader

from manga_ocr.dataset.generated_manga import create_dataset
from manga_ocr.text.localization import divine_rect_into_overlapping_tiles
from manga_ocr.text.localization.conv_unet.conv_unet import ConvUnet
from manga_ocr.text.localization.localization_model import LocalizationModel, \
    image_to_input_tensor
from manga_ocr.text.localization.train_with_generated_manga import GeneratedMangaDataset, train
from manga_ocr.typing import Size, Rectangle


def test_loading_generated_manga_dataset(tmpdir):

    dataset_dir = tmpdir.join('dataset')
    create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = GeneratedMangaDataset.load(dataset_dir, image_size=Size.of(500, 500))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['image'].shape == (2, 3, 500, 500)
    assert batch['mask_char'].shape == (2, 1, 500, 500)
    assert batch['mask_line'].shape == (2, 1, 500, 500)


def test_loading_generated_manga_dataset_with_resize(tmpdir):

    dataset_dir = tmpdir.join('dataset')
    create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = GeneratedMangaDataset.load(dataset_dir, image_size=Size.of(200, 200))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['image'].shape == (2, 3, 200, 200)
    assert batch['mask_char'].shape == (2, 1, 200, 200)
    assert batch['mask_line'].shape == (2, 1, 200, 200)


def test_training_conv_unet(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    create_dataset(dataset_dir, output_count=3)

    model = ConvUnet()
    assert model.image_size == (768, 768)

    dataset = GeneratedMangaDataset.load(dataset_dir, image_size=model.image_size)

    input_image = image_to_input_tensor(dataset.images[0]).unsqueeze(0)
    output_char, output_line = model(input_image)
    assert output_char.shape == (1, 1, 768, 768)
    assert output_line.shape == (1, 1, 768, 768)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    assert loss.item() > 0

    train(model, train_dataset=dataset, tqdm_disable=True)



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
