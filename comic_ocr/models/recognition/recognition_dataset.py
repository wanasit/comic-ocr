from __future__ import annotations
from random import Random
from typing import List, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

import comic_ocr.dataset.annotated_manga as annotated_manga
import comic_ocr.dataset.generated_manga as generated_manga

from comic_ocr.models.recognition import encode
from comic_ocr.models.recognition.recognition_model import image_to_single_input_tensor, DEFAULT_INPUT_HEIGHT
from comic_ocr.typing import Rectangle


class RecognitionDataset(Dataset):
    """
    TODO: support input_max_width and add custom padding logic for images
    """

    def __init__(self,
                 line_images: List[Image.Image],
                 line_texts: List[str],
                 input_height: int = DEFAULT_INPUT_HEIGHT,
                 ):
        assert len(line_images) == len(line_texts)
        self.line_images = line_images
        self.line_texts = [t.strip() for t in line_texts]
        self.input_height = input_height
        self.output_max_len = max((len(t) for t in self.line_texts))

    def __len__(self):
        return len(self.line_images)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_tensor = image_to_single_input_tensor(image=self.line_images[idx], input_height=self.input_height)
        output_tensor = torch.tensor(encode(text=self.line_texts[idx], padded_output_size=self.output_max_len))
        output_length_tensor = torch.tensor([len(self.line_texts[idx])])

        return {
            'input': input_tensor,
            'output': output_tensor,
            'output_length': output_length_tensor
        }

    def get_line_image(self, idx):
        return self.line_images[idx]

    def get_line_text(self, idx):
        return self.line_texts[idx]

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None) -> RecognitionDataset:
        from_idx = from_idx if from_idx is not None else 0
        to_idx = to_idx if to_idx is not None else len(self.line_images)
        return RecognitionDataset(self.line_images[from_idx:to_idx], self.line_texts[from_idx:to_idx],
                                  self.input_height)

    def shuffle(self, random_seed: any = '') -> RecognitionDataset:
        indexes = list(range(len(self.line_images)))
        random = Random(random_seed)
        random.shuffle(indexes)

        line_images = [self.line_images[i] for i in indexes]
        line_texts = [self.line_texts[i] for i in indexes]
        return RecognitionDataset(line_images=line_images, line_texts=line_texts, input_height=self.input_height)

    @staticmethod
    def merge(dataset_a: RecognitionDataset, dataset_b: RecognitionDataset) -> RecognitionDataset:
        assert dataset_a.input_height == dataset_b.input_height, \
            'Can only merge dataset with the same input_height. TODO: add this later'

        line_images = dataset_a.line_images + dataset_b.line_images
        line_texts = dataset_a.line_texts + dataset_b.line_texts
        return RecognitionDataset(line_images=line_images, line_texts=line_texts, input_height=dataset_a.input_height)

    @staticmethod
    def load_annotated_dataset(
            directory: str,
            input_height: int = DEFAULT_INPUT_HEIGHT,
            random_padding_x: Union[int, Tuple[int, int]] = (0, 10),
            random_padding_y: Union[int, Tuple[int, int]] = (0, 5),
            random_padding_copy_count: int = 1,
            random_seed: any = ''
    ):
        random = Random(random_seed)
        images, image_texts = annotated_manga.load_line_annotated_dataset(directory)

        line_images = []
        line_texts = []
        for image, lines in zip(images, image_texts):
            for line in lines:
                if not line.text:
                    continue
                for padding_copy_i in range(random_padding_copy_count):
                    rect = _rect_with_random_padding(random, line.location, random_padding_x, random_padding_y)
                    line_images.append(image.crop(rect))
                    line_texts.append(line.text)

        return RecognitionDataset(
            line_images=line_images,
            line_texts=line_texts,
            input_height=input_height
        )

    @staticmethod
    def load_generated_dataset(
            directory: str,
            input_height: int = DEFAULT_INPUT_HEIGHT,
            random_padding_x: Union[int, Tuple[int, int]] = (0, 10),
            random_padding_y: Union[int, Tuple[int, int]] = (2, 10),
            random_padding_copy_count: int = 1,
            random_seed: any = ''
    ):
        random = Random(random_seed)
        images, image_texts, _ = generated_manga.load_dataset(directory)

        line_images = []
        line_texts = []

        for image, lines in zip(images, image_texts):
            for line in lines:
                for padding_copy_i in range(random_padding_copy_count):
                    rect = _rect_with_random_padding(random, line.location, random_padding_x, random_padding_y)
                    line_images.append(image.crop(rect))
                    line_texts.append(line.text)

        return RecognitionDataset(
            line_images=line_images,
            line_texts=line_texts,
            input_height=input_height
        )


def _rect_with_random_padding(
        random: Random,
        rect: Rectangle,
        random_padding_x: Union[int, Tuple[int, int]],
        random_padding_y: Union[int, Tuple[int, int]],
):
    random_padding_x = (random_padding_x, random_padding_x) if isinstance(random_padding_x, int) else random_padding_x
    random_padding_y = (random_padding_y, random_padding_y) if isinstance(random_padding_y, int) else random_padding_y

    padding_x = random.randint(random_padding_x[0], random_padding_x[1]) if random_padding_x else 0
    padding_y = random.randint(random_padding_y[0], random_padding_y[1]) if random_padding_y else 0
    return rect.expand((padding_x, padding_y))


if __name__ == '__main__':
    from comic_ocr.utils import get_path_project_dir

    dataset_annotated = RecognitionDataset.load_annotated_dataset(get_path_project_dir('example/manga_annotated'))
    dataset_annotated.get_line_image(0).show()

    dataset_annotated = RecognitionDataset.load_annotated_dataset(get_path_project_dir('data/manga_line_annotated'))
    dataset_annotated.get_line_image(0).show()

    dataset_generated = RecognitionDataset.load_generated_dataset(get_path_project_dir('example/manga_generated'))
    dataset_generated.get_line_image(0).show()
    dataset_generated.get_line_image(1).show()
    dataset_generated.get_line_image(3).show()
