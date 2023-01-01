from torch.utils.data import DataLoader

from comic_ocr.models import localization
from comic_ocr.models.localization import LocalizationDataset
from comic_ocr.typing import Size
from comic_ocr.utils.files import get_path_example_dir


def test_default_model_high_level_metrics():

    model = localization.load_model()
    assert model

    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = LocalizationDataset.load_line_annotated_dataset(model, dataset_dir)
    assert dataset

    metrics = localization.calculate_high_level_metrics(model, dataset)
    assert metrics

