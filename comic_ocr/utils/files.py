import glob
import json
import os
from typing import Optional, Tuple, List, Dict

from PIL import Image

from comic_ocr.types import PathLike, ImageRGB
from comic_ocr.utils.nb_annotation import find_annotation_data_for_image

current_module_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(current_module_dir, '../../')


def get_path_project_dir(child='') -> str:
    path = os.path.join(project_root_dir, child)
    return path


def get_path_example_dir(child='') -> str:
    path = get_path_project_dir('example')
    path = os.path.join(path, child)
    return path


def load_images_with_annotation(
        glob_file_pattern: PathLike,
        alt_annotation_directory: Optional[PathLike] = None
) -> Tuple[List[Image.Image], List[str], List[Optional[Dict]]]:
    files = glob.glob(str(glob_file_pattern))
    images = []
    image_files = []
    annotations = []
    for file in sorted(files):
        if os.path.isdir(file):
            continue
        if file.endswith('.json'):
            continue
        images.append(load_image(file))
        image_files.append(os.path.abspath(file))

        annotation_file = find_annotation_data_for_image(file, alt_annotation_directory)
        annotations.append(annotation_file)

    return images, image_files, annotations


def load_images(glob_file_pattern: PathLike) -> Tuple[List[ImageRGB], List[str]]:
    files = glob.glob(str(glob_file_pattern))
    images = []
    image_files = []
    for file in sorted(files):
        images.append(load_image(file))
        image_files.append(os.path.abspath(file))

    return images, image_files


def load_image(file: PathLike) -> ImageRGB:
    with Image.open(file) as img:
        return img.copy().convert('RGB')


def load_texts(text_file: PathLike):
    with open(text_file) as f:
        return [line.strip() for line in f.readlines()]


def load_json_dict(json_file: PathLike) -> Dict:
    with open(json_file) as f:
        return json.load(f)


def write_json_dict(json_file: PathLike, data: Dict):
    with open(json_file, 'w') as f:
        return json.dump(data, f)
