"""A module for text recognition (aka. reading character sequence from text image)

"""

import logging
from typing import Optional, List

import torch

from comic_ocr.models.recognition.encoding import encode, decode
from comic_ocr.models.recognition.recognition_model import RecognitionModel, CharBaseRecognitionModel, \
    BasicCharBaseRecognitionModel, SUPPORT_DICT_SIZE
from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset, RecognitionDatasetWithAugmentation
from comic_ocr.models.recognition.recognition_utils import calculate_high_level_metrics
from comic_ocr.utils.files import PathLike, get_path_project_dir

DEFAULT_LOCAL_TRAINED_MODEL_FILE = get_path_project_dir('trained_models/recognition.pth')

logger = logging.getLogger(__name__)


def load_or_create_new_model(model_file: PathLike = DEFAULT_LOCAL_TRAINED_MODEL_FILE) -> RecognitionModel:
    try:
        model = load_model(model_file)
        model()
    except:
        logger.info(f'Fail loading model at [{model_file}]. Creating a new model.')

    return create_new_model()


def create_new_model(**kwargs) -> RecognitionModel:
    from comic_ocr.models.recognition.crnn.crnn import CRNN
    return CRNN(**kwargs)


def load_model(
        model_file: PathLike = DEFAULT_LOCAL_TRAINED_MODEL_FILE,
        test_executing_model: bool = True
) -> RecognitionModel:
    logger.info(f'Loading localization model [{model_file}]')
    model: RecognitionModel = torch.load(model_file)

    if test_executing_model:
        # TODO: test the model here
        pass

    return model


if __name__ == '__main__':
    from comic_ocr.utils import get_path_project_dir

    dataset_dir = get_path_project_dir('example/manga_annotated')
    model = load_model()
    dataset = RecognitionDataset.load_annotated_dataset(dataset_dir)

    for i in range(len(dataset)):
        print(dataset.get_line_text(i), model.recognize(dataset.get_line_image(i)))
