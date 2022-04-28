"""A module for localization problem (aka. locating text inside image)

This top-level module provide shortcut APIs for working with the LocalizationModel

"""
import logging

import torch

from manga_ocr.models.localization.localization_model import LocalizationModel
from manga_ocr.utils.files import PathLike, get_path_project_dir, load_image

DEFAULT_TRAINED_MODEL_FILE = get_path_project_dir('trained_models/localization.bin')
DEFAULT_EXAMPLE_IMAGE = get_path_project_dir('example/manga_annotated/normal_01.jpg')

logger = logging.getLogger(__name__)


def load_or_create_default_model(model_file: PathLike = DEFAULT_TRAINED_MODEL_FILE) -> LocalizationModel:
    try:
        model = load_model(model_file)
        model()
    except:
        logger.info(f'Fail loading model at [{model_file}]. Creating a new model.')

    from manga_ocr.models.localization.conv_unet.conv_unet import ConvUnet
    return ConvUnet()


def load_model(
        model_file: PathLike = DEFAULT_TRAINED_MODEL_FILE,
        test_executing_model: bool = True
) -> LocalizationModel:
    logger.info(f'Loading localization model [{model_file}]')
    model: LocalizationModel = torch.load(model_file)

    if test_executing_model:
        logger.info(f'Testing the model')
        image = load_image(DEFAULT_EXAMPLE_IMAGE)
        _ = model.locate_paragraphs(image)

    return model
