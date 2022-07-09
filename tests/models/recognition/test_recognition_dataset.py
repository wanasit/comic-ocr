from torch.utils.data import DataLoader

from manga_ocr.models.recognition import encode
from manga_ocr.models.recognition.recognition_dataset import RecognitionDataset
from manga_ocr.models.recognition.recognition_model import DEFAULT_INPUT_HEIGHT
from manga_ocr.utils.files import get_path_example_dir


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


def test_dataset_shuffle():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    # Fixed the seed: [0, 1, 2] => [2, 0, 1]
    dataset = dataset.subset(from_idx=0, to_idx=3)
    dataset_shuffled = dataset.shuffle(random_seed='0')

    assert len(dataset_shuffled) == len(dataset) == 3
    assert dataset_shuffled.input_height == dataset.input_height == DEFAULT_INPUT_HEIGHT
    assert \
        [dataset.get_line_image(2), dataset.get_line_image(0), dataset.get_line_image(1)] == \
        [dataset_shuffled.get_line_image(0), dataset_shuffled.get_line_image(1), dataset_shuffled.get_line_image(2)]
    assert \
        [dataset.get_line_text(2), dataset.get_line_text(0), dataset.get_line_text(1)] == \
        [dataset_shuffled.get_line_text(0), dataset_shuffled.get_line_text(1), dataset_shuffled.get_line_text(2)]


def test_dataset_merge():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    # Merge: [0, 1, 2] + [2, 3, 4] => [0, 1, 2, 2, 3, 4]
    dataset_a = dataset.subset(from_idx=0, to_idx=3)
    dataset_b = dataset.subset(from_idx=2, to_idx=5)
    merged_dataset = RecognitionDataset.merge(dataset_a, dataset_b)

    assert len(merged_dataset) == 6
    assert merged_dataset.input_height == dataset_a.input_height == DEFAULT_INPUT_HEIGHT
    assert merged_dataset.get_line_image(0) == dataset.get_line_image(0)
    assert merged_dataset.get_line_image(1) == dataset.get_line_image(1)
    assert merged_dataset.get_line_image(2) == dataset.get_line_image(2)
    assert merged_dataset.get_line_image(3) == dataset.get_line_image(2)
    assert merged_dataset.get_line_image(4) == dataset.get_line_image(3)
    assert merged_dataset.get_line_image(5) == dataset.get_line_image(4)


def test_dataset_with_loader():
    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'),
                                                        random_padding_x=0, random_padding_y=0)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(train_dataloader))

    assert batch['input'].shape == (1, 3, 24, 130)
    assert batch['output'].shape == (1, 17)
    assert batch['output_length'].shape == (1, 1)
