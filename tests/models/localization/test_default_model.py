from torch.utils.data import DataLoader

from manga_ocr.models import localization
from manga_ocr.models.localization import LocalizationDataset
from manga_ocr.typing import Size
from manga_ocr.utils.files import get_path_example_dir


def test_default_model_high_level_metrics():

    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(dataset_dir, image_size=Size.of(500, 500))
    assert dataset

    model = localization.load_model()
    assert model

    metrics = localization.calculate_high_level_metrics(model, dataset)
    assert metrics

