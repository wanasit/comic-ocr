import torch
from torch.utils.data import DataLoader

from comic_ocr.models.recognition import RecognitionDataset
from comic_ocr.models.recognition.trocr.trocr import TrOCR
from comic_ocr.utils.files import get_path_example_dir


def test_recognize():
    recognizer = TrOCR.from_pretrain()

    dataset = RecognitionDataset.load_annotated_dataset(recognizer, get_path_example_dir('manga_annotated'))

    output = recognizer.recognize(dataset.line_images[0])
    assert isinstance(output, str)


def test_loss_computing():
    recognizer = TrOCR.from_pretrain()

    dataset = RecognitionDataset.load_annotated_dataset(recognizer, get_path_example_dir('manga_annotated'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
