import glob
from typing import List

from PIL import Image


def load_images(glob_file_pattern) -> List[Image.Image]:
    files = glob.glob(glob_file_pattern)
    images = []
    for file in sorted(files):
        with Image.open(file) as img:
            images.append(img.copy())

    return images


def load_texts(text_file):
    with open(text_file) as f:
        return [line.strip() for line in f.readlines()]