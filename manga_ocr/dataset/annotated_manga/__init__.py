from typing import List, Dict, Tuple

from PIL.Image import Image

from manga_ocr.typing import Line, Rectangle
from manga_ocr.utils.files import load_images_with_annotation
from manga_ocr.utils.nb_annotation import lines_from_nb_annotation_data


def load_line_annotated_dataset(dataset_dir: str) -> List[Tuple[Image, List[Line]]]:
    images, _, image_annotations = load_images_with_annotation(dataset_dir + '/*.jpg')

    output = []
    for image, annotation_data in zip(images, image_annotations):
        if not annotation_data:
            continue

        lines = lines_from_nb_annotation_data(annotation_data)
        output.append((image, lines))

    return output