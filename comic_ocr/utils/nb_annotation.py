"""
For reading and writing JSON annotation that compatible with labelling-notebook
https://github.com/wanasit/labelling-notebook

This assumes each annotation is per-line.

"""
import json
import os
from typing import List, Dict, Optional

from comic_ocr.types import Line, Rectangle


def find_annotation_data_for_image(img_path: str, alt_directory: Optional[str] = None) -> Optional[Dict]:
    """
    Look for the annotation file for the image file (starting at the same directory)
    :param img_path: image file (full or relative path)
    :param alt_directory: (Optional) an alternative directory to look for the annotation file
    """

    expected_annotation_file = os.path.splitext(img_path)[0] + '.json'
    if os.path.isfile(expected_annotation_file):
        return _load_json_dict(expected_annotation_file)

    if alt_directory:
        expected_annotation_file = os.path.join(alt_directory, os.path.split(expected_annotation_file)[-1])
        if os.path.isfile(expected_annotation_file):
            return _load_json_dict(expected_annotation_file)

    return None


def write_annotation_data_for_image(img_path: str, annotation_data: Dict, alt_directory: Optional[str] = None):
    annotation_file = os.path.splitext(img_path)[0] + '.json'
    if alt_directory:
        annotation_file = os.path.join(alt_directory, os.path.split(annotation_file)[-1])

    _write_json_dict(annotation_file, annotation_data)


def lines_to_nb_annotation_data(lines: List[Line]) -> Dict:
    annotations = []
    for line in lines:
        annotation = line_to_nb_annotation(line)
        annotations.append(annotation)
    return {
        'annotations': annotations
    }


def lines_from_nb_annotation_data(
        annotation_data: Dict,
        empty_text: Optional[str] = None
) -> List[Line]:
    lines = []
    if 'annotations' in annotation_data:
        for a in annotation_data['annotations']:
            line = line_from_nb_annotation(a, empty_text=empty_text)
            if line:
                lines.append(line)

    return lines


def line_to_nb_annotation(line: Line) -> Dict:
    annotation = rect_to_nb_annotation(line.location)
    annotation['text'] = line.text
    return annotation


def line_from_nb_annotation(
        annotation,
        empty_text: Optional[str] = None
) -> Optional[Line]:
    rect = rect_from_annotation(annotation)
    if 'text' in annotation:
        text = annotation['text']
        return Line.of(text, rect)

    if empty_text is not None:
        return Line.of(empty_text, rect)

    return None


def rect_to_nb_annotation(rect: Rectangle) -> Dict:
    return {
        'x': int(rect.left),
        'y': int(rect.top),
        'width': int(rect.width),
        'height': int(rect.height)
    }


def rect_from_annotation(annotation) -> Rectangle:
    return Rectangle.of_xywh(annotation['x'], annotation['y'], annotation['width'], annotation['height'])


def _load_json_dict(json_file: str) -> Dict:
    with open(json_file) as f:
        return json.load(f)


def _write_json_dict(json_file: str, json_dict: Dict):
    with open(json_file, 'w') as f:
        return json.dump(json_dict, f)
