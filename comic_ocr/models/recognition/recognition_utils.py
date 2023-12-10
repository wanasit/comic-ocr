from typing import Optional

import textdistance
import torch

from torchvision.transforms import transforms

from comic_ocr.models.recognition.recognition_model import RecognitionModel
from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset

TRANSFORM_TO_TENSOR = transforms.PILToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()


def calculate_high_level_metrics(
        model: RecognitionModel,
        dataset: RecognitionDataset,
        sample_size_limit: Optional[int] = None,
        device: Optional[torch.device] = None
):
    assert len(dataset) > 0
    sample_count = 0
    perfect_match_count = 0
    similarity_total = 0

    for i in range(len(dataset)):
        if sample_size_limit and i >= sample_size_limit:
            break
        sample_count += 1
        line_text_expected = dataset.get_line_text(i)
        line_text_image = dataset.get_line_image(i)
        line_text_actual = model.recognize(line_text_image, device=device)

        similarity_total += textdistance.levenshtein.normalized_similarity(line_text_expected, line_text_actual)

        if line_text_actual == line_text_expected:
            perfect_match_count += 1

    return {
        "dataset_size": sample_count,
        "perfect_match_count": perfect_match_count,
        "perfect_match_accuracy": perfect_match_count / sample_count,
        "similarity": similarity_total / sample_count,
    }
