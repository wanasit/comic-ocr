import glob
import os
import json
from typing import List, Tuple, Dict, Optional

from PIL import Image

current_module_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(current_module_dir, '../../')


def get_path_project_dir(child='') -> str:
    path = os.path.join(project_root_dir, child)
    return path


def get_path_example_dir(child='') -> str:
    path = get_path_project_dir('example')
    path = os.path.join(path, child)
    return path


def load_images_with_annotation(glob_file_pattern) -> Tuple[List[Image.Image], List[Optional[Dict]]]:
    files = glob.glob(glob_file_pattern)
    images = []
    annotations = []
    for file in sorted(files):
        with Image.open(file) as img:
            images.append(img.copy())

        expected_annotation_file = os.path.splitext(file)[0] + '.json'
        if os.path.isfile(expected_annotation_file):
            annotation = load_json_dict(expected_annotation_file)
            annotations.append(annotation)
        else:
            annotations.append(None)

    return images, annotations


def load_images(glob_file_pattern) -> List[Image.Image]:
    files = glob.glob(glob_file_pattern)
    images = []
    for file in sorted(files):
        images.append(load_image(file))

    return images


def load_image(file: str) -> Image.Image:
    with Image.open(file) as img:
        return img.copy()


def load_texts(text_file):
    with open(text_file) as f:
        return [line.strip() for line in f.readlines()]


def load_json_dict(json_file) -> Dict:
    with open(json_file) as f:
        return json.load(f)
