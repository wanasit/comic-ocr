from __future__ import annotations

import math
from random import Random
from typing import List, Dict, Optional, Tuple, Union, Sequence

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

import comic_ocr.dataset.annotated_manga as annotated_manga
import comic_ocr.dataset.generated_manga as generated_manga

from comic_ocr.models.recognition import encode
from comic_ocr.models import transforms
from comic_ocr.types import Rectangle


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

        self._images = images
        self._line_image_indexes = line_image_indexes
        self._line_rectangles = line_rectangles
        self._line_texts = [t.strip() for t in line_texts]
        self._image_to_tensor = image_to_tensor
        self._text_max_length = max((len(t) for t in self._line_texts))

    def __len__(self):
        return len(self._line_texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        text = self._line_texts[idx]
        text_length = torch.tensor([len(text)])
        text_encoded = torch.tensor(encode(text=text, padded_output_size=self.text_max_length))

        image = self.get_line_image(idx)
        image = self._image_to_tensor(image)
        return {
            'image': image,
            'text': text,
            'text_length': text_length,
            'text_encoded': text_encoded
        }

    @property
    def text_max_length(self) -> int:
        return self._text_max_length

    def loader(self, **kwargs):
        kwargs.pop('batch_size', None)
        kwargs.pop('num_workers', None)
        return torch.utils.data.DataLoader(self, batch_size=1, num_workers=0, **kwargs)

    def get_line_image(self, idx, padding: Union[int, Tuple[int, int]] = 0) -> Image.Image:
        image_idx = self._line_image_indexes[idx]
        line_rect = self._line_rectangles[idx].expand(padding)
        image = self._images[image_idx]
        return image.crop(line_rect)

    def get_line_text(self, idx):
        return self._line_texts[idx]

    def subset(self, from_idx: Optional[int] = None, to_idx: Optional[int] = None) -> RecognitionDataset:
        from_idx = from_idx if from_idx is not None else 0
        to_idx = to_idx if to_idx is not None else len(self._line_texts)
        current_image_indexes = self._line_image_indexes[from_idx:to_idx]

        images = []
        line_image_indexes = []
        image_index_by_curent_image_index = {}
        for idx in current_image_indexes:
            if idx not in image_index_by_curent_image_index:
                image_index_by_curent_image_index[idx] = len(images)
                images.append(self._images[idx])
            line_image_indexes.append(image_index_by_curent_image_index[idx])

        return RecognitionDataset(
            image_to_tensor=self._image_to_tensor,
            images=images,
            line_image_indexes=line_image_indexes,
            line_rectangles=self._line_rectangles[from_idx:to_idx],
            line_texts=self._line_texts[from_idx:to_idx])

    def shuffle(self, random_seed: any = '') -> RecognitionDataset:
        indexes = list(range(len(self._line_texts)))
        random = Random(random_seed)
        random.shuffle(indexes)

        line_image_indexes = [self._line_image_indexes[i] for i in indexes]
        line_rectangles = [self._line_rectangles[i] for i in indexes]
        line_texts = [self._line_texts[i] for i in indexes]
        return RecognitionDataset(
            image_to_tensor=self._image_to_tensor, images=self._images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )

    def repeat(self, n_times: int) -> RecognitionDataset:
        line_image_indexes = self._line_image_indexes * n_times
        line_rectangles = self._line_rectangles * n_times
        line_texts = self._line_texts * n_times
        return RecognitionDataset(
            image_to_tensor=self._image_to_tensor, images=self._images,
            line_image_indexes=line_image_indexes,
            line_rectangles=line_rectangles,
            line_texts=line_texts
        )

    @staticmethod
    def merge(dataset_a: RecognitionDataset, dataset_b: RecognitionDataset) -> RecognitionDataset:
        assert dataset_a._image_to_tensor == dataset_b._image_to_tensor, \
            'Can only merge dataset with the same image to tensor transform.'

        images = dataset_a._images + dataset_b._images

        appending_line_image_indexes = [len(dataset_a._images) + i for i in dataset_b._line_image_indexes]
        line_image_indexes = dataset_a._line_image_indexes + appending_line_image_indexes
        line_rectangles = dataset_a._line_rectangles + dataset_b._line_rectangles
        line_texts = dataset_a._line_texts + dataset_b._line_texts

        return RecognitionDataset(image_to_tensor=dataset_a._image_to_tensor, images=images,
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

    def __init__(self,
                 r: Random,
                 choices_padding_width: Sequence[int] = (0,),
                 choices_padding_height: Sequence[int] = (0,),
                 batch_height: Optional[int] = None,
                 enable_color_jitter: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r = r
        self._choices_padding_width = choices_padding_width
        self._choices_padding_height = choices_padding_height
        self._batch_height = batch_height
        image_transforms = []
        if enable_color_jitter:
            color_jitter_kwargs = {k[13:]: v for k, v in kwargs.items() if
                                   k.startswith('color_jitter_')}
            image_transforms += [transforms.ColorJitter(**color_jitter_kwargs)]
        self._image_transform = transforms.Compose(image_transforms)

    @property
    def choices_padding_width(self) -> Sequence[int]:
        return self._choices_padding_width

    @property
    def choices_padding_height(self) -> Sequence[int]:
        return self._choices_padding_height

    @property
    def batch_height(self) -> Optional[int]:
        return self._batch_height

    @property
    def image_transform(self) -> transforms.TensorTransformFunc:
        return self._image_transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = item['image']
        image = self._image_transform(image)
        return {
            **item,
            'image': image,
        }

    def loader(self, **kwargs):
        def collate_fn(batch):
            images = [b['image'].unsqueeze(0) for b in batch]
            images = [_pad_random_choices(
                self._r, i,
                choices_padding_width=self._choices_padding_width, choices_padding_height=self._choices_padding_height)
                for i in images]

            # Scale all images to the same height, then pad them to the same width
            batch_height = self._batch_height
            if batch_height is None:
                batch_height = max((i.shape[-2] for i in images))
            images = [_scale_image_tensor_to_height(i, batch_height) for i in images]
            max_width = max((i.shape[-1] for i in images))
            images = [_pad_image_tensor_to_width(i, max_width) for i in images]

            images = [i[0] for i in images]
            images = torch.stack(images)
            texts = [b['text'] for b in batch]
            return {
                'image': images,
                'text': texts,
                'text_length': torch.stack([b['text_length'] for b in batch]),
                'text_encoded': torch.stack([b['text_encoded'] for b in batch]),
            }

        return torch.utils.data.DataLoader(self, collate_fn=collate_fn, **kwargs)

    def shuffle(self, random_seed: any = '') -> RecognitionDatasetWithAugmentation:
        shuffled = super().shuffle(random_seed)
        shuffled = RecognitionDatasetWithAugmentation.of_dataset(shuffled)
        shuffled._batch_height = self._batch_height
        shuffled._image_transform = self._image_transform
        shuffled._choices_padding_width = self._choices_padding_width
        shuffled._choices_padding_height = self._choices_padding_height
        return shuffled

    def subset(self, from_idx: Optional[int] = None,
               to_idx: Optional[int] = None) -> RecognitionDatasetWithAugmentation:
        subset = super().subset(from_idx, to_idx)
        subset = RecognitionDatasetWithAugmentation.of_dataset(subset)
        subset._batch_height = self._batch_height
        subset._image_transform = self._image_transform
        subset._choices_padding_width = self._choices_padding_width
        subset._choices_padding_height = self._choices_padding_height
        return subset

    def repeat(self, n_times: int) -> RecognitionDatasetWithAugmentation:
        repeated = super().repeat(n_times)
        repeated = RecognitionDatasetWithAugmentation.of_dataset(repeated)
        repeated._batch_height = self._batch_height
        repeated._image_transform = self._image_transform
        repeated._choices_padding_width = self._choices_padding_width
        repeated._choices_padding_height = self._choices_padding_height
        return repeated

    def without_augmentation(self) -> RecognitionDataset:
        return RecognitionDataset(
            image_to_tensor=self._image_to_tensor,
            images=self._images,
            line_image_indexes=self._line_image_indexes,
            line_rectangles=self._line_rectangles,
            line_texts=self._line_texts
        )

    @staticmethod
    def of_dataset(dataset: RecognitionDataset,
                   r: Optional[Random] = None,
                   **kwargs) -> RecognitionDatasetWithAugmentation:
        r = r if r is not None else Random()
        return RecognitionDatasetWithAugmentation(
            image_to_tensor=dataset._image_to_tensor,
            images=dataset._images,
            line_image_indexes=dataset._line_image_indexes,
            line_rectangles=dataset._line_rectangles,
            line_texts=dataset._line_texts,
            r=r,
            **kwargs
        )

    @staticmethod
    def load_generated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor,
            **kwargs
    ):
        dataset = RecognitionDataset.load_generated_dataset(directory, image_to_tensor)
        return RecognitionDatasetWithAugmentation.of_dataset(dataset, **kwargs)

    @staticmethod
    def load_annotated_dataset(
            directory: str,
            image_to_tensor: transforms.ImageToTensorTransformFunc = transforms.image_to_tensor,
            **kwargs
    ):
        dataset = RecognitionDataset.load_annotated_dataset(directory, image_to_tensor)
        return RecognitionDatasetWithAugmentation.of_dataset(dataset, **kwargs)


def _scale_image_tensor_to_height(image: torch.Tensor, height: int) -> torch.Tensor:
    assert len(image.shape) == 4
    scale_factor = height / image.shape[-2]
    return F.interpolate(image, size=(height, int(image.shape[-1] * scale_factor)), mode='bilinear',
                         align_corners=False)


def _pad_random_choices(r: Random, image: torch.Tensor, choices_padding_width, choices_padding_height) -> torch.Tensor:
    assert len(image.shape) == 4
    padding_width = r.choice(choices_padding_width)
    padding_height = r.choice(choices_padding_height)
    padding_y = (math.floor(padding_height / 2), math.ceil(padding_height / 2))
    padding_x = (math.floor(padding_width / 2), math.ceil(padding_width / 2))
    return F.pad(image, padding_x + padding_y, mode='constant', value=1.0)


def _pad_image_tensor_to_width(image: torch.Tensor, width: int) -> torch.Tensor:
    assert len(image.shape) == 4
    padding_width = width - image.shape[-1]
    padding_x = (math.floor(padding_width / 2), math.ceil(padding_width / 2))
    return F.pad(image, padding_x, mode='constant', value=1.0)
