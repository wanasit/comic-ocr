"""Uses Pytorch Hub to download and cached trained models.
"""

import logging
import os

import torch

from comic_ocr.utils import files
from comic_ocr.models import localization
from comic_ocr.models import recognition

URL_LOCALIZATION_MODEL = 'https://github.com/wanasit/comic-ocr/raw/main/trained_models/localization.pth'
URL_RECOGNITION_MODEL = 'https://github.com/wanasit/comic-ocr/raw/main/trained_models/recognition.pth'
DEFAULT_EXAMPLE_IMAGE = files.get_path_project_dir('example/manga_annotated/normal_01.jpg')

logger = logging.getLogger(__name__)


def download_localization_model(
        url: str = URL_LOCALIZATION_MODEL,
        progress: bool = False,
        force_reload: bool = False,
        test_executing_model: bool = True) -> localization.LocalizationModel:
    file_name = 'localization.pth'
    cached_file = os.path.join(torch.hub.get_dir(), file_name)
    if force_reload or not os.path.exists(cached_file):
        logger.info(f'Downloading "{url}" to "{cached_file}"...')
        torch.hub.download_url_to_file(url, cached_file, progress=progress)

    return localization.load_model(cached_file, test_executing_model=test_executing_model)


def download_recognition_model(
        url: str = URL_RECOGNITION_MODEL,
        progress: bool = False,
        force_reload: bool = False,
        test_executing_model: bool = True) -> recognition.RecognitionModel:
    file_name = 'recognition.pth'
    cached_file = os.path.join(torch.hub.get_dir(), file_name)
    if force_reload or not os.path.exists(cached_file):
        logger.info(f'Downloading "{url}" to "{cached_file}"...')
        torch.hub.download_url_to_file(url, cached_file, progress=progress)

    return recognition.load_model(cached_file, test_executing_model=test_executing_model)
