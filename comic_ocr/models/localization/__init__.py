"""A module for localization problem (aka. locating text inside image)

This top-level module provide shortcut and high-level APIs for working with the LocalizationModel and its dependencies
or implementation details (both Pytorch/ML or OpenCV).

"""
import logging
from typing import Iterable, Optional

import torch

from comic_ocr.types import Rectangle
from comic_ocr.models.localization.localization_model import LocalizationModel, BasicLocalizationModel
from comic_ocr.models.localization.localization_dataset import LocalizationDataset
from comic_ocr.utils.files import PathLike, get_path_project_dir, load_image

DEFAULT_LOCAL_TRAINED_MODEL_FILE = get_path_project_dir('trained_models/localization.pth')
DEFAULT_EXAMPLE_IMAGE = get_path_project_dir('example/manga_annotated/xkcd_100.jpg')

logger = logging.getLogger(__name__)


def load_or_create_new_model(model_file: PathLike = DEFAULT_LOCAL_TRAINED_MODEL_FILE) -> LocalizationModel:
    try:
        model = load_model(model_file, test_executing_model=True)
        return model
    except (Exception,):
        logger.info(f'Fail loading model at [{model_file}]. Creating a new model.')
    return create_new_model()


def create_new_model() -> LocalizationModel:
    from comic_ocr.models.localization.conv_unet import conv_unet
    return conv_unet.BaselineConvUnet()


def load_model(
        model_file: PathLike = DEFAULT_LOCAL_TRAINED_MODEL_FILE,
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
        dataset: LocalizationDataset,
        sample_size_limit: Optional[int] = None
):
    """
    Calculate understandable high-level metrics (e.g. accuracy for locating lines).
    """
    assert len(dataset) > 0
    assert dataset.output_line_locations, 'Requires dataset with line locations information'
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(dataset)):
        if sample_size_limit and i >= sample_size_limit:
            break
        baseline_line_locations = dataset.get_line_locations(i)
        line_locations = model.locate_lines(dataset.get_image(i))
        tp, fp, fn = match_location_rectangles_with_baseline(line_locations, baseline_line_locations)
        tp, fp, fn = len(tp), len(fp), len(fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return {
        "sample_size": i,
        "total_line_level_true_positive": total_tp,
        "total_line_level_false_positive": total_fp,
        "total_line_level_false_negative": total_fn,
        "line_level_precision": total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0,
        "line_level_recall": total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0,
        "line_level_accuracy": total_tp / (total_tp + total_fn + total_fp) if total_tp + total_fn + total_fp > 0 else 0
    }


def match_location_rectangles_with_baseline(locations: Iterable[Rectangle], baseline_locations: Iterable[Rectangle]):
    matched_pairs = []
    unmatched_locations = []
    matched_base_line_indexes = set()

    for location in locations:
        for i, baseline_location in enumerate(baseline_locations):
            if i in matched_base_line_indexes:
                continue
            if location.can_represent(baseline_location):
                matched_pairs.append((location, baseline_location))
                matched_base_line_indexes.add(i)
                break
        else:
            unmatched_locations.append(location)

    unmatched_baseline_locations = [l for i, l in enumerate(baseline_locations) if i not in matched_base_line_indexes]
    return matched_pairs, unmatched_locations, unmatched_baseline_locations
