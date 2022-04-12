from typing import List, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

import manga_ocr.dataset.annotated_manga as annotated_manga
import manga_ocr.dataset.generated_manga as generated_manga

from manga_ocr.models.recognition import encode
from manga_ocr.models.recognition.recognition_module import image_to_single_input_tensor, DEFAULT_INPUT_HEIGHT


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


    def subset(self, from_idx: Optional[int] = None, to_dix: Optional[int] = None):
        from_idx = from_idx if from_idx is not None else 0
        to_dix = to_dix if to_dix is not None else len(self.line_images)
        return RecognitionDataset(self.line_images[from_idx:to_dix], self.line_texts[from_idx:to_dix], self.input_height)

    @staticmethod
    def load_annotated_dataset(directory: str, input_height: int = DEFAULT_INPUT_HEIGHT):
        line_images = []
        line_texts = []

        line_annotated_dataset = annotated_manga.load_line_annotated_dataset(directory)
        for image, lines in line_annotated_dataset:

            for line in lines:
                # todo: add random padding
                line_texts.append(line.text)
                line_images.append(image.crop(line.location))

        return RecognitionDataset(
            line_images=line_images,
            line_texts=line_texts,
            input_height=input_height
        )

    @staticmethod
    def load_generated_dataset(directory: str, input_height: int = DEFAULT_INPUT_HEIGHT):
        line_images = []
        line_texts = []

        generated_dataset = generated_manga.load_dataset(directory)
        for image, _, lines in generated_dataset:

            for line in lines:
                # todo: add random padding
                line_texts.append(line.text)
                line_images.append(image.crop(line.location))

        return RecognitionDataset(
            line_images=line_images,
            line_texts=line_texts,
            input_height=input_height
        )
