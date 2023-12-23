from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image

from comic_ocr.dataset.generated_single_line.generator import SingleLineGenerator
from comic_ocr.utils import files


def load_dataset(dataset_dir: str) -> Tuple[List[Image.Image], List[str]]:
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
        dataset_dir: str,
        generator: Optional[SingleLineGenerator] = None,
        output_count: Optional[int] = 100,
):
    """Creates dataset in the directory."""
    generator = generator if generator else SingleLineGenerator.create()

    path = Path(dataset_dir)
    (path / 'image').mkdir(parents=True, exist_ok=True)
    (path / 'text_annotation').mkdir(parents=True, exist_ok=True)

    for i in range(output_count):
        image, text = generator.generate(i)
        image.save(path / 'image' / '{:04}.jpg'.format(i))
        files.write_json_dict(path / 'text_annotation' / '{:04}.json'.format(i), {'text': text})
