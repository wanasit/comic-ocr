from PIL import ImageFont
from PIL import Image

from comic_ocr.dataset import generated_single_line
from comic_ocr import Rectangle
from comic_ocr.dataset.generated_manga.text_area import TextArea
from comic_ocr.types import Point
from comic_ocr.utils.files import get_path_example_dir



def test_generate_dataset(tmpdir):
    dataset_dir = tmpdir / 'test_generate_single_line_dataset'

    generated_single_line.create_dataset(dataset_dir, output_count=2)

    images, texts = generated_single_line.load_dataset(dataset_dir)
    assert len(images) == 2
    assert len(texts) == 2

    assert isinstance(images[0], Image.Image)
    assert isinstance(images[1], Image.Image)

    assert len(texts[0]) > 0
    assert len(texts[1]) > 0
