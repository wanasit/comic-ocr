from __future__ import annotations
from random import Random
from typing import List, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

import comic_ocr.dataset.annotated_manga as annotated_manga
import comic_ocr.dataset.generated_manga as generated_manga

from comic_ocr.models.recognition import encode
from comic_ocr.models.recognition.recognition_model import TransformImageToTensor, RecognitionModel
from comic_ocr.types import Rectangle


class RecognitionDataset(Dataset):
    """
    TODO: support input_max_width and add custom padding logic for images
    """

    def __init__(self,
                 line_images: List[Image.Image],
                 line_texts: List[str],
                 transform_image_to_input_tensor: TransformImageToTensor,
                 ):
        assert len(line_images) == len(line_texts)
        self.line_images = line_images
        self.line_texts = [t.strip() for t in line_texts]
        self.transform_image_to_input_tensor = transform_image_to_input_tensor
        self.output_max_len = max((len(t) for t in self.line_texts))

    def __len__(self):
        return len(self.line_images)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        raw_image = self.line_images[idx]
        raw_text = self.line_texts[idx]
        input_tensor = self.transform_image_to_input_tensor(raw_image)
        output_tensor = torch.tensor(encode(text=raw_text, padded_output_size=self.output_max_len))
        output_length_tensor = torch.tensor([len(raw_text)])
        return {
            'text': raw_text,
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
        return RecognitionDataset(line_images=self.line_images[from_idx:to_idx],
                                  line_texts=self.line_texts[from_idx:to_idx],
                                  transform_image_to_input_tensor=self.transform_image_to_input_tensor)

    def shuffle(self, random_seed: any = '') -> RecognitionDataset:
        indexes = list(range(len(self.line_images)))
        random = Random(random_seed)
        random.shuffle(indexes)

        line_images = [self.line_images[i] for i in indexes]
        line_texts = [self.line_texts[i] for i in indexes]
        return RecognitionDataset(line_images=line_images, line_texts=line_texts,
                                  transform_image_to_input_tensor=self.transform_image_to_input_tensor)

    @staticmethod
    def merge(dataset_a: RecognitionDataset, dataset_b: RecognitionDataset) -> RecognitionDataset:
        assert dataset_a.transform_image_to_input_tensor == dataset_b.transform_image_to_input_tensor, \
            'Can only merge dataset with the same transforming. TODO: add this later'

        line_images = dataset_a.line_images + dataset_b.line_images
        line_texts = dataset_a.line_texts + dataset_b.line_texts
        return RecognitionDataset(line_images=line_images, line_texts=line_texts,
                                  transform_image_to_input_tensor=dataset_a.transform_image_to_input_tensor)

    @staticmethod
    def load_annotated_dataset(
            model: RecognitionModel,
            directory: str,
            random_padding_x: Union[int, Tuple[int, int]] = (0, 5),
            random_padding_y: Union[int, Tuple[int, int]] = (0, 4),
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
            transform_image_to_input_tensor=model.transform_image_to_input_tensor
        )

    @staticmethod
    def load_generated_dataset(
            model: RecognitionModel,
            directory: str,
            random_padding_x: Union[int, Tuple[int, int]] = (0, 5),
            random_padding_y: Union[int, Tuple[int, int]] = (0, 5),
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
            transform_image_to_input_tensor=model.transform_image_to_input_tensor
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

    model = RecognitionModel()

    dataset_annotated = RecognitionDataset.load_annotated_dataset(model,
                                                                  get_path_project_dir('example/manga_annotated'))
    dataset_annotated.get_line_image(0).show()

    dataset_annotated = RecognitionDataset.load_annotated_dataset(model,
                                                                  get_path_project_dir('data/manga_line_annotated'))
    dataset_annotated.get_line_image(0).show()

    dataset_generated = RecognitionDataset.load_generated_dataset(model,
                                                                  get_path_project_dir('example/manga_generated'))
    dataset_generated.get_line_image(0).show()
    dataset_generated.get_line_image(1).show()
    dataset_generated.get_line_image(3).show()
