from typing import List, Tuple

from PIL.Image import Image

from manga_ocr.models.localization.localization_model import LocalizationModel
from manga_ocr.typing import Rectangle
from manga_ocr.utils.files import get_path_project_dir

_localization_model_path = get_path_project_dir('trained_models/localization.bin')
_localization_model = None


def localize_lines(image: Image) -> List[Rectangle]:
    model = get_localization_model()
    return model.locate_lines(image)


def localize_paragraphs(image: Image) -> List[Tuple[Rectangle, List[Rectangle]]]:
    model = get_localization_model()
    return model.locate_paragraphs(image)


def get_localization_model() -> LocalizationModel:
    global _localization_model
    if not _localization_model:
        import torch
        _localization_model = torch.load(_localization_model_path)

    return _localization_model




