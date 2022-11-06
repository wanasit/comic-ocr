import os
from typing import Union

import PIL
import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

from comic_ocr.models.recognition.recognition_model import RecognitionModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TRANSFORM_TO_TENSOR = transforms.ToTensor()


def to_transformer_compatible_input(tensor_or_image: Union[Image, torch.Tensor]) -> torch.Tensor:
    input_tensor = tensor_or_image
    if isinstance(tensor_or_image, Image):
        input_tensor = TRANSFORM_TO_TENSOR(tensor_or_image)

    # Padding to square
    # As TrOCR and ViTEncoder accept only
    original_h, original_w = input_tensor.shape[-2], input_tensor.shape[-1]
    padding = [0, 0, 0, 0]
    if original_w >= original_h:
        padding[2] = (original_w - original_h) // 2
        padding[3] = (original_w - original_h + 1) // 2
    else:
        padding[0] = (original_h - original_w) // 2
        padding[1] = (original_h - original_w + 1) // 2
    return F.pad(input_tensor, padding, value=1.0)


# As TrOCR and ViTEncoder accept only

class TrOCR(RecognitionModel):
    """

    """

    def __init__(
            self,
            transformer_tokenizer,
            transformer_model,
            transform_image_to_input_tensor,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_tokenizer = transformer_tokenizer
        self.transformer_model = transformer_model
        self.transform_image_to_input_tensor = transform_image_to_input_tensor

        self.transformer_model.config.decoder_start_token_id = self.transformer_tokenizer.cls_token_id
        self.transformer_model.config.pad_token_id = self.transformer_tokenizer.pad_token_id
        self.transformer_model.config.vocab_size = self.transformer_model.config.decoder.vocab_size

        # set beam search parameters
        self.transformer_model.config.eos_token_id = self.transformer_tokenizer.sep_token_id
        self.transformer_model.config.max_length = 64
        self.transformer_model.config.early_stopping = True
        self.transformer_model.config.num_beams = 4

    @staticmethod
    def from_pretrain(pretrained_model_name_or_path="microsoft/trocr-base-handwritten", **kwargs):
        processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        vision_encoder_model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return TrOCR(
            transformer_model=vision_encoder_model,
            transformer_tokenizer=processor.tokenizer,
            transform_image_to_input_tensor=VitImageTransform(vision_encoder_model.encoder.config.image_size)
        )

    def forward(self, **kwargs) -> torch.Tensor:
        raise AttributeError("TrOCR model does not support forward operation")

    def compute_loss(self, dataset_batch) -> torch.Tensor:
        input_tensor = dataset_batch['input']
        labels = self.transformer_tokenizer(dataset_batch['text'], return_tensors="pt").input_ids
        outputs = self.transformer_model(input_tensor, labels=labels)
        return outputs.loss

    def recognize(self, tensor_or_image: Union[Image, torch.Tensor]) -> str:
        input_tensor = self.transform_image_to_input_tensor(tensor_or_image)
        generated_ids = self.transformer_model.generate(input_tensor.unsqueeze(0), max_length=100)
        return self.transformer_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class VitImageTransform(nn.Module):
    def __init__(self, expected_size):
        super().__init__()
        self._to_tensor = transforms.ToTensor()
        self._resize = transforms.Resize(expected_size)
        self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, tensor_or_image: Union[Image, torch.Tensor]):
        input_tensor = tensor_or_image
        if isinstance(tensor_or_image, Image):
            input_tensor = self._to_tensor(tensor_or_image)
        input_tensor = pad_to_square(input_tensor)
        input_tensor = self._resize(input_tensor)
        return self._normalize(input_tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# As TrOCR and ViTEncoder accept only square input
def pad_to_square(input_tensor: torch.Tensor) -> torch.Tensor:
    original_h, original_w = input_tensor.shape[-2], input_tensor.shape[-1]
    padding = [0, 0, 0, 0]
    if original_w >= original_h:
        padding[2] = (original_w - original_h) // 2
        padding[3] = (original_w - original_h + 1) // 2
    else:
        padding[0] = (original_h - original_w) // 2
        padding[1] = (original_h - original_w + 1) // 2
    return F.pad(input_tensor, padding, value=1.0)


if __name__ == '__main__':
    from comic_ocr.utils.files import get_path_example_dir, get_path_project_dir
    from comic_ocr.utils.pytorch_model import get_total_parameters_count
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset
    from torch.utils.data import DataLoader

    recognizer = TrOCR.from_pretrain()
    print('get_total_parameters_count', get_total_parameters_count(recognizer))

    dataset = RecognitionDataset.load_annotated_dataset(recognizer, get_path_project_dir('data/manga_line_annotated'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(recognizer.recognize(dataset.get_line_image(0)), dataset.get_line_text(0))

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    print('loss', loss)

    # input = image_to_single_input_tensor(recognizer.input_height, image)
    # recognizer(input.unsqueeze(0))

