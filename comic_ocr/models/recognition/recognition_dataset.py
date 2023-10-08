from __future__ import annotations

import math
from random import Random
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

import comic_ocr.dataset.annotated_manga as annotated_manga
import comic_ocr.dataset.generated_manga as generated_manga

from comic_ocr.models.recognition import encode
from comic_ocr.models import transforms
from comic_ocr.types import Rectangle, Size


class RecognitionDataset(Dataset):
    """A torch dataset for evaluating a recognition model.

    Each dataset entry is a text line. Each line includes the image, the location (rectangle) withing the image, and the
    line's text (string). Because each line has a different size, the dataset should be load with batch size 1.

    For training a recognition model, consider using `RecognitionDatasetWithAugmentation` instead.
    """

    def __init__(self,
                 images: List[Image.Image],
                 line_image_indexes: List[int],
                 line_rectangles: List[Rectangle],
                 line_texts: List[str],
                 image_to_tensor: transforms.ImageToTensorTransformFunc):
        assert len(line_image_indexes) == len(line_rectangles) == len(line_texts)

        self.images = images
        self.line_image_indexes = line_image_indexes
        self.line_rectangles = line_rectangles
        self.line_texts = [t.strip() for t in line_texts]
        self.image_to_tensor = image_to_tensor
        self.text_max_length = max((len(t) for t in self.line_texts))

    def __len__(self):
        return len(self.line_texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        text = self.line_texts[idx]
        text_length = torch.tensor([len(text)])
        text_encoded = torch.tensor(encode(text=text, padded_output_size=self.text_max_length))

        image = self.get_line_image(idx)
        image = self.image_to_tensor(image)
        return {
            'image': image,
            'text': text,
            'text_length': text_length,
            'text_encoded': text_encoded
        }

    def loader(self, **kwargs):
        kwargs.pop('batch_size', None)
        kwargs.pop('num_workers', None)
        return torch.utils.data.DataLoader(self, batch_size=1, num_workers=0, **kwargs)

    def get_line_image(self, idx):
        image_idx = self.line_image_indexes[idx]
        line_rect = self.line_rectangles[idx]
        image = self.images[image_idx]
        return image.crop(line_rect)

    def get_line_text(self, idx):
        return self.line_texts[idx]

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None) -> RecognitionDataset:
        from_idx = from_idx if from_idx is not None else 0
        to_idx = to_idx if to_idx is not None else len(self.line_texts)
        current_image_indexes = self.line_image_indexes[from_idx:to_idx]

        images = []
        line_image_indexes = []
        image_index_by_curent_image_index = {}
        for idx in current_image_indexes:
            if idx not in image_index_by_curent_image_index:
                image_index_by_curent_image_index[idx] = len(images)
                images.append(self.images[idx])
            line_image_indexes.append(image_index_by_curent_image_index[idx])

        return RecognitionDataset(
            image_to_tensor=self.image_to_tensor,
            images=images,
            line_image_indexes=line_image_indexes,
            line_rectangles=self.line_rectangles[from_idx:to_idx],
            line_texts=self.line_texts[from_idx:to_idx])

    def shuffle(self, random_seed: any = '') -> RecognitionDataset:
        indexes = list(range(len(self.line_texts)))
        random = Random(random_seed)
        random.shuffle(indexes)

        line_image_indexes = [self.line_image_indexes[i] for i in indexes]
        line_rectangles = [self.line_rectangles[i] for i in indexes]
        line_texts = [self.line_texts[i] for i in indexes]
        return RecognitionDataset(
            image_to_tensor=self.image_to_tensor, images=self.images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )

    def repeat(self, n_times: int) -> RecognitionDataset:
        line_image_indexes = self.line_image_indexes * n_times
        line_rectangles = self.line_rectangles * n_times
        line_texts = self.line_texts * n_times
        return RecognitionDataset(
            image_to_tensor=self.image_to_tensor, images=self.images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )

    @staticmethod
    def merge(dataset_a: RecognitionDataset, dataset_b: RecognitionDataset) -> RecognitionDataset:
        assert dataset_a.image_to_tensor == dataset_b.image_to_tensor, \
            'Can only merge dataset with the same image to tensor transform.'

        images = dataset_a.images + dataset_b.images

        appending_line_image_indexes = [len(dataset_a.images) + i for i in dataset_b.line_image_indexes]
        line_image_indexes = dataset_a.line_image_indexes + appending_line_image_indexes
        line_rectangles = dataset_a.line_rectangles + dataset_b.line_rectangles
        line_texts = dataset_a.line_texts + dataset_b.line_texts

        return RecognitionDataset(image_to_tensor=dataset_a.image_to_tensor, images=images,
                                  line_image_indexes=line_image_indexes,
                                  line_rectangles=line_rectangles,
                                  line_texts=line_texts)

    @staticmethod
    def load_annotated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor
    ):
        images, image_texts = annotated_manga.load_line_annotated_dataset(directory)

        line_image_indexes = []
        line_rectangles = []
        line_texts = []
        for i, lines in enumerate(image_texts):
            for line in lines:
                if not line.text:
                    continue

                line_image_indexes.append(i)
                line_rectangles.append(line.location)
                line_texts.append(line.text)

        return RecognitionDataset(
            image_to_tensor=image_to_tensor,
            images=images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )

    @staticmethod
    def load_generated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor
    ):
        images, image_texts, _ = generated_manga.load_dataset(directory)

        line_image_indexes = []
        line_rectangles = []
        line_texts = []
        for i, lines in enumerate(image_texts):
            for line in lines:
                if not line.text:
                    continue

                line_image_indexes.append(i)
                line_rectangles.append(line.location)
                line_texts.append(line.text)

        return RecognitionDataset(
            image_to_tensor=image_to_tensor,
            images=images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )


class RecognitionDatasetWithAugmentation(RecognitionDataset):
    """A torch dataset for training a recognition model.

    This dataset augments the normal recognition dataset with random augmentations. This dataset supports batch loading
    by resizing and padding the images to the maximum size of the batch.
    """

    def loader(self, **kwargs):
        def collate_fn(batch):
            # Scale all images to the same height, then pad them to the same width
            max_height = max((b['image'].shape[-2] for b in batch))
            resize_images = [_scale_image_tensor_to_height(i['image'], max_height) for i in batch]
            max_width = max((i.shape[-1] for i in resize_images))
            resize_images = [_pad_image_tensor_to_width(i, max_width) for i in resize_images]

            resize_images = torch.stack(resize_images)
            texts = [b['text'] for b in batch]
            return {
                'image': resize_images,
                'text': texts,
                'text_length': torch.stack([b['text_length'] for b in batch]),
                'text_encoded': torch.stack([b['text_encoded'] for b in batch]),
            }

        return torch.utils.data.DataLoader(self, collate_fn=collate_fn, **kwargs)

    def shuffle(self, random_seed: any = '') -> RecognitionDatasetWithAugmentation:
        shuffled = super().shuffle(random_seed)
        return RecognitionDatasetWithAugmentation.of_dataset(shuffled)

    def subset(self, from_idx: Optional[int] = None,
               to_idx: Optional[int] = None) -> RecognitionDatasetWithAugmentation:
        subset = super().subset(from_idx, to_idx)
        return RecognitionDatasetWithAugmentation.of_dataset(subset)

    @staticmethod
    def of_dataset(dataset: RecognitionDataset) -> RecognitionDatasetWithAugmentation:
        return RecognitionDatasetWithAugmentation(
            image_to_tensor=dataset.image_to_tensor,
            images=dataset.images,
            line_image_indexes=dataset.line_image_indexes,
            line_rectangles=dataset.line_rectangles,
            line_texts=dataset.line_texts
        )

    @staticmethod
    def load_generated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor
    ):
        dataset = RecognitionDataset.load_generated_dataset(directory, image_to_tensor)
        return RecognitionDatasetWithAugmentation.of_dataset(dataset)

    @staticmethod
    def load_annotated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor
    ):
        dataset = RecognitionDataset.load_annotated_dataset(directory, image_to_tensor)
        return RecognitionDatasetWithAugmentation.of_dataset(dataset)


def _scale_image_tensor_to_height(image: torch.Tensor, height: int) -> torch.Tensor:
    assert len(image.shape) == 3
    scale_factor = height / image.shape[1]
    return F.interpolate(image.unsqueeze(0), size=(height, int(image.shape[2] * scale_factor)), mode='bilinear',
                         align_corners=False)


def _pad_image_tensor_to_width(image: torch.Tensor, width: int) -> torch.Tensor:
    assert len(image.shape) == 4
    padding_width = width - image.shape[-1]
    padding_x = (math.floor(padding_width / 2), math.ceil(padding_width / 2))
    return F.pad(image, padding_x, mode='constant', value=0)[0]


def _resize_line_image_tensor(image: torch.Tensor, size: Size) -> torch.Tensor:
    # Scale image proportionally to fit the target height.
    scale_factor = size.height / image.shape[1]
    scaled_image = F.interpolate(image.unsqueeze(0), size=(size.height, int(image.shape[2] * scale_factor)),
                                 mode='bilinear', align_corners=False)

    # Pad image to fit the target width.
    padding_width = size.width - scaled_image.shape[3]
    padding_x = (math.floor(padding_width / 2), math.ceil(padding_width / 2))
    padded_scaled_image = F.pad(scaled_image, padding_x, mode='constant', value=0)

    return padded_scaled_image[0]
