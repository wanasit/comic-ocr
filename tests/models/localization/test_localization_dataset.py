from torch.utils.data import DataLoader

from manga_ocr.dataset import generated_manga
from manga_ocr.models.localization.localization_dataset import LocalizationDataset
from manga_ocr.typing import Size
from manga_ocr.utils.files import get_path_example_dir


def test_load_line_annotated_manga_dataset():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(dataset_dir, image_size=Size.of(300, 500))

    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 500, 300)
    assert batch['output_mask_line'].shape == (2, 500, 300)



def test_load_generated_manga_dataset(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    generated_manga.create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = LocalizationDataset.load_generated_manga_dataset(dataset_dir, image_size=Size.of(500, 500))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 500, 500)
    assert batch['output_mask_char'].shape == (2, 500, 500)
    assert batch['output_mask_line'].shape == (2, 500, 500)
    assert batch['output_mask_paragraph'].shape == (2, 500, 500)


def test_load_generated_manga_dataset_with_resize(tmpdir):
    dataset_dir = tmpdir.join('dataset')
    generated_manga.create_dataset(dataset_dir, output_count=5, output_size=Size.of(500, 500))

    dataset = LocalizationDataset.load_generated_manga_dataset(dataset_dir, image_size=Size.of(200, 200))
    assert len(dataset) > 0

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (2, 3, 200, 200)
    assert batch['output_mask_char'].shape == (2, 200, 200)
    assert batch['output_mask_line'].shape == (2, 200, 200)
    assert batch['output_mask_paragraph'].shape == (2, 200, 200)
