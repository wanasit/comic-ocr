import torch
from comic_ocr.models.recognition import encode, SUPPORT_DICT_SIZE

from comic_ocr.models.recognition import recognition_dataset
from comic_ocr.models.recognition import recognition_model
from comic_ocr.utils.files import load_image, get_path_example_dir


def test_ctc_loss():
    encoded_chars = encode(text="Hello", padded_output_size=10)
    assert encoded_chars == [46, 17, 24, 24, 27, 0, 0, 0, 0, 0]

    expected_output = torch.Tensor(encoded_chars).type(torch.int64)
    expected_output_length = torch.Tensor([5]).type(torch.int64)

    output_sequence = torch.as_tensor(encoded_chars)
    model_output = torch.nn.functional.one_hot(output_sequence, num_classes=SUPPORT_DICT_SIZE).type(torch.float)

    loss = recognition_model.compute_ctc_loss(
        model_output.unsqueeze(0),
        expected_output.unsqueeze(0),
        expected_output_length.unsqueeze(0)
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


def test_recognizer_basic():
    recognizer = recognition_model.BasicCharBaseRecognitionModel()

    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        get_path_example_dir('manga_annotated'))
    dataloader = dataset.loader(batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

    line_image = dataset.get_line_image(0)
    predicted_text = recognizer.recognize(line_image)
    assert isinstance(predicted_text, str)


def test_recognizer_crnn_basic():
    from comic_ocr.models.recognition.crnn.crnn import CRNN
    recognizer = CRNN()

    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        get_path_example_dir('manga_annotated'))
    dataloader = dataset.loader(batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

    line_image = dataset.get_line_image(0)
    predicted_text = recognizer.recognize(line_image)
    assert isinstance(predicted_text, str)
