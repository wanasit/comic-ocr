"""A module for loading annotated comic dataset

This module uses `comic_ocr.utils.labelling_notebook` for loading the annotation data.
The annotation should be in `labelling-notebook` format:
(ref: https://github.com/wanasit/labelling-notebook)

Example:
  images, texts = load_line_annotated_dataset('./example/manga_annotated')

"""

from typing import List, Tuple, Optional

from PIL.Image import Image

from comic_ocr.types import Line
from comic_ocr.utils.files import load_images_with_annotation
from comic_ocr.utils.nb_annotation import lines_from_nb_annotation_data


def load_line_annotated_dataset(
        dataset_dir: str,
        include_empty_text: bool = False,
        skip_empty_check: bool = False
) -> Tuple[List[Image], List[List[Line]]]:
    """Load dataset with annotation-per text line

    Args:
        dataset_dir (Str, Path): path to the dataset directory
        include_empty_text (bool): should include the annotation without text or empty text

    Returns:
        images (List[Image])
        image_texts (List[List[Line]])
    """
    images, _, image_annotations = load_images_with_annotation(dataset_dir + '/*')
    assert skip_empty_check or len(images) > 0
    image_texts = []
    for image, annotation_data in zip(images, image_annotations):
        if not annotation_data:
            continue

        lines = lines_from_nb_annotation_data(
            annotation_data, empty_text='' if include_empty_text else None)
        image_texts.append(lines)

    return images, image_texts


if __name__ == '__main__':
    from comic_ocr.utils.files import get_path_project_dir
    images, lines = load_line_annotated_dataset(get_path_project_dir('data/manga_line_annotated'))
    print(len(images), len(lines))