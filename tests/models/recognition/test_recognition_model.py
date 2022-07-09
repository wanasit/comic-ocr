import torch
from torch.utils.data import DataLoader

from manga_ocr.models.recognition import encode, SUPPORT_DICT_SIZE
from manga_ocr.models.recognition.crnn.crnn import CRNN
from manga_ocr.models.recognition.recognition_dataset import RecognitionDataset
from manga_ocr.models.recognition.recognition_model import image_to_single_input_tensor, compute_ctc_loss
from manga_ocr.utils.files import load_image, get_path_example_dir


def test_image_to_single_input_tensor_scale_down():
    image = load_image(get_path_example_dir('manga_annotated/normal_01.jpg'))
    assert image.size == (707, 1000)
    input_height = 30

    input = image_to_single_input_tensor(input_height, image)
    assert isinstance(input, torch.Tensor)
    assert input.shape[0] == 3
    assert input.shape[1] == input_height == 30
    assert input.shape[2] == 21


def test_image_to_single_input_tensor_scale_up():
    image = load_image(get_path_example_dir('manga_annotated/normal_01.jpg'))
    assert image.size == (707, 1000)
    input_height = 2000

    input = image_to_single_input_tensor(input_height, image)
    assert isinstance(input, torch.Tensor)
    assert input.shape[0] == 3
    assert input.shape[1] == input_height == 2000
    assert input.shape[2] == 1414


def test_recognizer_loss_computing():
    recognizer = CRNN()

    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


def test_ctc_loss():
    encoded_chars = encode(text="Hello", padded_output_size=10)
    assert encoded_chars == [46, 17, 24, 24, 27, 0, 0, 0, 0, 0]

    expected_output = torch.Tensor(encoded_chars).type(torch.int64)
    expected_output_length = torch.Tensor([5]).type(torch.int64)

    output_sequence = torch.as_tensor(encoded_chars)
    model_output = torch.nn.functional.one_hot(output_sequence, num_classes=SUPPORT_DICT_SIZE).type(torch.float)

    loss = compute_ctc_loss(
        model_output.unsqueeze(0),
        expected_output.unsqueeze(0),
        expected_output_length.unsqueeze(0)
    )


def test_recognizer_basic():
    recognizer = CRNN()

    image = load_image(get_path_example_dir('manga_annotated/normal_01.jpg'))
    input = image_to_single_input_tensor(recognizer.input_height, image)

    output = recognizer(input.unsqueeze(0))[0]
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == SUPPORT_DICT_SIZE

