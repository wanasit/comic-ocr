import torch
from torch.utils.data import DataLoader

from manga_ocr.models.recognition import SUPPORT_DICT_SIZE, encode
from manga_ocr.models.recognition.crnn.crnn import CRNN
from manga_ocr.models.recognition.recognition_dataset import RecognitionDataset
from manga_ocr.models.recognition.recognition_module import image_to_single_input_tensor, compute_ctc_loss, DEFAULT_INPUT_HEIGHT

from manga_ocr.utils import get_path_example_dir, load_image


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


def test_loading_annotated_dataset():
    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'))

    assert len(dataset) > 0

    row = dataset[0]
    assert row.keys() == {'input', 'output', 'output_length'}
    assert row['input'].shape[0] == 3
    assert row['input'].shape[1] == DEFAULT_INPUT_HEIGHT == 24

    input_width = row['input'].shape[2]
    assert input_width > 0

    assert row['output'].shape[0] == dataset.output_max_len
    assert row['output'].tolist() == encode('DEPRESSION', padded_output_size=dataset.output_max_len)

    assert row['output_length'].shape[0] == 1
    assert row['output_length'].tolist() == [len('DEPRESSION')]


def test_loading_generated_dataset():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    assert len(dataset) > 0

    row = dataset[0]
    assert row.keys() == {'input', 'output', 'output_length'}
    assert row['input'].shape[0] == 3
    assert row['input'].shape[1] == DEFAULT_INPUT_HEIGHT == 24

    input_width = row['input'].shape[2]
    assert input_width > 0

    assert row['output'].shape[0] == dataset.output_max_len
    assert row['output'].tolist() == encode('Hehe', padded_output_size=dataset.output_max_len)

    assert row['output_length'].shape[0] == 1
    assert row['output_length'].tolist() == [len('Hehe')]


def test_dataset_with_loader():
    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'))
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (1, 3, 24, 130)
    assert batch['output'].shape == (1, 17)
    assert batch['output_length'].shape == (1, 1)


def test_recognizer_basic():
    recognizer = CRNN()

    image = load_image(get_path_example_dir('manga_annotated/normal_01.jpg'))
    input = image_to_single_input_tensor(recognizer.input_height, image)

    output = recognizer(input.unsqueeze(0))[0]
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == SUPPORT_DICT_SIZE


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