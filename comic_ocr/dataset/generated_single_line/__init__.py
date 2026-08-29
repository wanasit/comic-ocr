from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image

from comic_ocr.dataset.generated_single_line.generator import SingleLineGenerator
from comic_ocr.utils import files


def load_dataset(dataset_dir: files.PathLike) -> Tuple[List[Image.Image], List[str]]:
    """Load the dataset created by `create_dataset()`

    Args:
        dataset_dir (Str, Path): path to the dataset directory

    Returns:
        images (List[Image])
        texts (List[str])
    """
    path = Path(dataset_dir)
    images, _, annotations = files.load_images_with_annotation(path / 'image/*.jpg', path / 'text_annotation')
    texts = [a['text'] for a in annotations]
    return images, texts


def create_dataset(
        dataset_dir: files.PathLike,
        generator: Optional[SingleLineGenerator] = None,
        output_count: Optional[int] = 100,
):
    """Generates and writes a single-line OCR training dataset to disk.

    Creates two subdirectories in `dataset_dir`:
    - `image/`: Cropped single-line text images (.jpg).
    - `text_annotation/`: JSON files containing the ground truth transcription string `{"text": "..."}`.

    Args:
        dataset_dir: Destination directory path.
        generator: Optional SingleLineGenerator instance.
        output_count: Number of single-line text crops to generate.
    """
    generator = generator if generator else SingleLineGenerator.create()

    path = Path(dataset_dir)
    (path / 'image').mkdir(parents=True, exist_ok=True)
    (path / 'text_annotation').mkdir(parents=True, exist_ok=True)

    for i in range(output_count):
        image, text = generator.generate(i)
        image.save(path / 'image' / '{:04}.jpg'.format(i))
        files.write_json_dict(path / 'text_annotation' / '{:04}.json'.format(i), {'text': text})


if __name__ == '__main__':
    from comic_ocr.utils.files import get_path_project_dir

    dataset_dir = get_path_project_dir('data/output/generated_single_line_test')
    create_dataset(dataset_dir, output_count=3)
    print(dataset_dir)