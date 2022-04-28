from typing import List, Tuple

from PIL.Image import Image

from manga_ocr.models import localization
from manga_ocr.typing import Rectangle
from manga_ocr.utils.files import get_path_project_dir

_localization_model = None


def localize_lines(image: Image) -> List[Rectangle]:
    model = get_localization_model()
    return model.locate_lines(image)


def localize_paragraphs(image: Image) -> List[Tuple[Rectangle, List[Rectangle]]]:
    model = get_localization_model()
    return model.locate_paragraphs(image)


def get_localization_model() -> localization.LocalizationModel:
    global _localization_model
    if not _localization_model:
        _localization_model = localization.load_model()

    return _localization_model
