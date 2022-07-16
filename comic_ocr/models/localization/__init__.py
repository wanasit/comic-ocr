"""A module for localization problem (aka. locating text inside image)

This top-level module provide shortcut and high-level APIs for working with the LocalizationModel and its dependencies
or implementation details (both Pytorch/ML or OpenCV).

"""
import logging
from typing import Iterable

import torch

from comic_ocr.typing import Rectangle
from comic_ocr.models.localization.localization_model import LocalizationModel
from comic_ocr.models.localization.localization_dataset import LocalizationDataset
from comic_ocr.utils.files import PathLike, get_path_project_dir, load_image

DEFAULT_TRAINED_MODEL_FILE = get_path_project_dir('trained_models/localization.bin')
DEFAULT_EXAMPLE_IMAGE = get_path_project_dir('example/manga_annotated/normal_01.jpg')

logger = logging.getLogger(__name__)


def load_or_create_new_model(model_file: PathLike = DEFAULT_TRAINED_MODEL_FILE) -> LocalizationModel:
    try:
        model = load_model(model_file)
        model()
    except:
        logger.info(f'Fail loading model at [{model_file}]. Creating a new model.')

    return create_new_model()


def create_new_model() -> LocalizationModel:
    from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
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


def calculate_high_level_metrics(
        model: LocalizationModel,
        dataset: LocalizationDataset
):
    assert len(dataset) > 0
    assert dataset.output_locations_lines, 'Requires dataset with line locations information'
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(dataset)):
        baseline_line_locations = dataset.get_line_locations(i)
        line_locations = model.locate_lines(dataset.get_image(i))
        tp, fp, fn = match_location_rectangles_with_baseline(line_locations, baseline_line_locations)
        tp, fp, fn = len(tp), len(fp), len(fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return {
        "dataset_size": len(dataset),
        "total_line_level_true_positive": total_tp,
        "total_line_level_false_positive": total_fp,
        "total_line_level_false_negative": total_fn,
        "line_level_precision": total_tp / (total_tp + total_fp),
        "line_level_recall": total_tp / (total_tp + total_fn)
    }


def match_location_rectangles_with_baseline(locations: Iterable[Rectangle], baseline_locations: Iterable[Rectangle]):
    matched_pairs = []
    unmatched_locations = []

    for location in locations:
        for i, baseline_location in enumerate(baseline_locations):
            if location.can_represent(baseline_location):
                matched_pairs.append((location, baseline_location))
                baseline_locations = baseline_locations[:i] + baseline_locations[i + 1:]
                break
        else:
            unmatched_locations.append(location)

    return matched_pairs, unmatched_locations, baseline_locations
