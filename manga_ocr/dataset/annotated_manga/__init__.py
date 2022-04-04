from typing import List, Dict, Tuple

from PIL.Image import Image

from manga_ocr.typing import Line, Rectangle
from manga_ocr.utils import load_images_with_annotation


def load_line_annotated_dataset(dataset_dir: str) -> List[Tuple[Image, List[Line]]]:
    images, image_annotations = load_images_with_annotation(dataset_dir + '/*.jpg')

    output = []
    for image, annotation_data in zip(images, image_annotations):
        if not annotation_data:
            continue

        lines = read_line_annotations(annotation_data)
        output.append((image, lines))

    return output


def read_line_annotations(annotation_data: Dict) -> List[Line]:
    """
    annotation_data from labelling-notebook
    https://githubplus.com/wanasit/labelling-notebook
    """
    lines = []
    for a in annotation_data['annotations']:
        line = _read_labelling_nb_annotation_line(a)
        lines.append(line)

    return lines


def _read_labelling_nb_annotation_line(annotation) -> Line:
    rect = _read_labelling_nb_annotation_rect(annotation)
    text = annotation['text']
    return Line.of(text, rect)


def _read_labelling_nb_annotation_rect(annotation) -> Rectangle:
    return Rectangle.of_xywh(annotation['x'], annotation['y'], annotation['width'], annotation['height'])
