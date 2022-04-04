from typing import List, Dict

import torch
from PIL.Image import Image
from torch.utils.data import Dataset

from manga_ocr.dataset.annotated_manga import load_line_annotated_dataset
from manga_ocr.models.recognition import encode
from manga_ocr.models.recognition.recognition_module import image_to_single_input_tensor, DEFAULT_INPUT_HEIGHT


class RecognitionDataset(Dataset):
    """
    TODO: support input_max_width and add custom padding logic for images
    """

    def __init__(self,
                 line_images: List[Image],
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

    @staticmethod
    def load_annotated_dataset(directory: str, input_height: int = DEFAULT_INPUT_HEIGHT):
        line_images = []
        line_texts = []

        line_annotated_dataset = load_line_annotated_dataset(directory)
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
