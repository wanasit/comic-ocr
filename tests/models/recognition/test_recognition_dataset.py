from torch.utils.data import DataLoader

from comic_ocr.models.recognition import encode
from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset, RecognitionDatasetWithAugmentation
from comic_ocr.models.recognition.recognition_model import RecognitionModel, DEFAULT_INPUT_HEIGHT
from comic_ocr.utils.files import get_path_example_dir


def test_loading_annotated_dataset():
    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'))

    assert len(dataset) > 0

    row = dataset[0]
    assert row.keys() == {'image', 'text', 'text_length', 'text_encoded'}
    assert row['image'].shape[0] == 3
    assert row['image'].shape[1] == 16
    assert row['image'].shape[2] == 87

    assert row['text_encoded'].shape[0] == dataset.text_max_length
    assert row['text_encoded'].tolist() == encode('DEPRESSION', padded_output_size=dataset.text_max_length)

    assert row['text_length'].shape[0] == 1
    assert row['text_length'].tolist() == [len('DEPRESSION')]


def test_loading_generated_dataset():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    assert len(dataset) > 0

    row = dataset[0]
    assert row.keys() == {'image', 'text', 'text_length', 'text_encoded'}
    assert row['image'].shape[0] == 3
    assert row['image'].shape[1] == 20
    assert row['image'].shape[2] == 37

    assert row['text_encoded'].shape[0] == dataset.text_max_length
    assert row['text_encoded'].tolist() == encode('Hehe', padded_output_size=dataset.text_max_length)

    assert row['text_length'].shape[0] == 1
    assert row['text_length'].tolist() == [len('Hehe')]


def test_dataset_shuffle():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    # Fixed the seed: [0, 1, 2] => [2, 0, 1]
    dataset = dataset.subset(from_idx=0, to_idx=3)
    dataset_shuffled = dataset.shuffle(random_seed='0')

    assert len(dataset_shuffled) == len(dataset) == 3
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
    assert merged_dataset.get_line_image(0) == dataset.get_line_image(0) == dataset_a.get_line_image(0)
    assert merged_dataset.get_line_image(1) == dataset.get_line_image(1) == dataset_a.get_line_image(1)
    assert merged_dataset.get_line_image(2) == dataset.get_line_image(2) == dataset_a.get_line_image(2)
    assert merged_dataset.get_line_image(3) == dataset.get_line_image(2) == dataset_b.get_line_image(0)
    assert merged_dataset.get_line_image(4) == dataset.get_line_image(3) == dataset_b.get_line_image(1)
    assert merged_dataset.get_line_image(5) == dataset.get_line_image(4) == dataset_b.get_line_image(2)


def test_dataset_repeat():
    dataset = RecognitionDataset.load_generated_dataset(get_path_example_dir('manga_generated'))

    # Merge: [0, 1, 2] => [0, 1, 2, 0, 1, 2, 0, 1, 2]
    dataset = dataset.subset(from_idx=0, to_idx=3)
    repeated_dataset = dataset.repeat(3)

    assert len(repeated_dataset) == 9
    assert repeated_dataset.get_line_image(3) == repeated_dataset.get_line_image(0) == dataset.get_line_image(0)
    assert repeated_dataset.get_line_image(4) == repeated_dataset.get_line_image(1) == dataset.get_line_image(1)
    assert repeated_dataset.get_line_image(5) == repeated_dataset.get_line_image(2) == dataset.get_line_image(2)
    assert repeated_dataset.get_line_image(6) == repeated_dataset.get_line_image(3) == dataset.get_line_image(0)
    assert repeated_dataset.get_line_image(7) == repeated_dataset.get_line_image(4) == dataset.get_line_image(1)
    assert repeated_dataset.get_line_image(8) == repeated_dataset.get_line_image(5) == dataset.get_line_image(2)

    assert repeated_dataset.get_line_text(3) == repeated_dataset.get_line_text(0) == dataset.get_line_text(0)
    assert repeated_dataset.get_line_text(4) == repeated_dataset.get_line_text(1) == dataset.get_line_text(1)
    assert repeated_dataset.get_line_text(5) == repeated_dataset.get_line_text(2) == dataset.get_line_text(2)
    assert repeated_dataset.get_line_text(6) == repeated_dataset.get_line_text(3) == dataset.get_line_text(0)
    assert repeated_dataset.get_line_text(7) == repeated_dataset.get_line_text(4) == dataset.get_line_text(1)
    assert repeated_dataset.get_line_text(8) == repeated_dataset.get_line_text(5) == dataset.get_line_text(2)


def test_dataset_loader():
    dataset = RecognitionDataset.load_annotated_dataset(get_path_example_dir('manga_annotated'))
    train_dataloader = dataset.loader()
    batch = next(iter(train_dataloader))

    assert batch['image'].shape == (1, 3, 16, 87)
    assert batch['text_encoded'].shape == (1, 23)
    assert batch['text_length'].shape == (1, 1)


def test_dataset_with_augmentation_loader():
    dataset = RecognitionDatasetWithAugmentation.load_annotated_dataset(get_path_example_dir('manga_annotated'))
    assert dataset.get_line_image(0).size == (87, 16) # should be scaled to (108, 20)
    assert dataset.get_line_image(1).size == (95, 20)
    assert dataset.get_line_image(2).size == (42, 14) # should be scaled to (60, 20)

    train_dataloader = dataset.loader(batch_size=3, num_workers=0)
    batch = next(iter(train_dataloader))
    assert batch['image'].shape == (3, 3, 20, 108)
    assert batch['text_encoded'].shape == (3, 23)
    assert batch['text_length'].shape == (3, 1)
