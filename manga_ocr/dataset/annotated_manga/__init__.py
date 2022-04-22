"""A module for loading annotated manga dataset

This module uses `manga_ocr.utils.labelling_notebook` for loading the annotation data.
The annotation should be in `labelling-notebook` format:
(ref: https://github.com/wanasit/labelling-notebook)

Example:
  images, texts = load_line_annotated_dataset('./example/manga_annotated')

"""

from typing import List, Tuple

from PIL.Image import Image

from manga_ocr.typing import Line
from manga_ocr.utils.files import load_images_with_annotation
from manga_ocr.utils.nb_annotation import lines_from_nb_annotation_data


def load_line_annotated_dataset(dataset_dir: str) -> Tuple[List[Image], List[List[Line]]]:
    """Load dataset with annotation-per text line

    Args:
        dataset_dir (Str, Path): path to the dataset directory

    Returns:
        images (List[Image])
        image_texts (List[List[Line]])
    """
    images, _, image_annotations = load_images_with_annotation(dataset_dir + '/*.jpg')
    image_texts = []
    for image, annotation_data in zip(images, image_annotations):
        if not annotation_data:
            continue

        lines = lines_from_nb_annotation_data(annotation_data)
        image_texts.append(lines)

    return images, image_texts
