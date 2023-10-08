import pytest
import torch

from comic_ocr.models.recognition import recognition_dataset
from comic_ocr.models.recognition import recognition_model
from comic_ocr.models.recognition.train import train
from comic_ocr.utils import files


def test_train_to_finish_and_return_metrics():
    model = recognition_model.BasicCharBaseRecognitionModel()
    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        files.get_path_project_dir('example/manga_annotated'))
    dataset = dataset.subset(0, 4)
    assert len(dataset) == 4

    train_metrics, validate_metrics = train('testing_model', model,
                                            train_dataset=dataset,
                                            train_epoch_count=3,  # Train the same batch 3 times
                                            batch_size=4,
                                            validate_dataset=dataset,
                                            update_every_n=1,
                                            tqdm_enable=False, tensorboard_log_enable=False)

    # The metrics should have updated 3 times
    assert 'loss' in train_metrics
    assert len(train_metrics['loss']) == 3
    assert 'loss' in validate_metrics
    assert len(validate_metrics['loss']) == 3

    assert 'perfect_match_accuracy' in validate_metrics
    assert 'perfect_match_count' in validate_metrics


def test_train_to_send_update_callback():
    model = recognition_model.BasicCharBaseRecognitionModel()
    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        files.get_path_project_dir('example/manga_annotated'))
    dataset = dataset.subset(0, 4)
    assert len(dataset) == 4

    callback_at_steps = []

    def callback(steps, train_metrics, validate_metrics):
        assert 'loss' in train_metrics
        assert 'loss' in validate_metrics
        callback_at_steps.append(steps)

    train('testing_model', model,
          train_dataset=dataset,
          train_epoch_count=2,  # 2 epoch x 4/1 batch-per-epoch -> 8 steps
          batch_size=1,
          validate_dataset=dataset,
          update_every_n=3,
          update_callback=callback,
          tqdm_enable=False, tensorboard_log_enable=False)

    # The metrics should have updated 3 times
    assert callback_at_steps == [3, 6]


def test_train_and_validate_on_cpu():
    model = recognition_model.BasicCharBaseRecognitionModel()

    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        files.get_path_project_dir('example/manga_annotated'))
    dataset = dataset.subset(0, 4)
    assert len(dataset) == 4

    def callback(steps, train_metrics, validate_metrics):
        assert 'loss' in train_metrics
        assert 'loss' in validate_metrics

    train('testing_model_with_meta_train_device', model,
          train_dataset=dataset,
          train_epoch_count=2,  # 2 epoch x 4/1 batch-per-epoch -> 8 steps
          batch_size=1,
          validate_dataset=dataset,
          update_every_n=3,
          update_callback=callback,
          tqdm_enable=False,
          tensorboard_log_enable=False,
          train_device=torch.device('cpu'))

    train('testing_model_with_meta_validate_device', model,
          train_dataset=dataset,
          train_epoch_count=2,  # 2 epoch x 4/1 batch-per-epoch -> 8 steps
          batch_size=1,
          validate_dataset=dataset,
          update_every_n=3,
          update_callback=callback,
          tqdm_enable=False,
          tensorboard_log_enable=False,
          validate_device=torch.device('cpu'))


def test_train_and_validate_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip()

    model = recognition_model.BasicCharBaseRecognitionModel()
    dataset = recognition_dataset.RecognitionDatasetWithAugmentation.load_annotated_dataset(
        files.get_path_project_dir('example/manga_annotated'))
    dataset = dataset.subset(0, 4)
    assert len(dataset) == 4

    def callback(steps, train_metrics, validate_metrics):
        assert 'loss' in train_metrics
        assert 'loss' in validate_metrics

    train('testing_model_with_meta_train_device', model,
          train_dataset=dataset,
          train_epoch_count=2,  # 2 epoch x 4/1 batch-per-epoch -> 8 steps
          batch_size=1,
          validate_dataset=dataset,
          update_every_n=3,
          update_callback=callback,
          tqdm_enable=False,
          tensorboard_log_enable=False,
          train_device=torch.device('cuda:0'),
          validate_device=torch.device('cuda:0'))

    train('testing_model_with_meta_validate_device', model,
          train_dataset=dataset,
          train_epoch_count=2,  # 2 epoch x 4/1 batch-per-epoch -> 8 steps
          batch_size=1,
          validate_dataset=dataset,
          update_every_n=3,
          update_callback=callback,
          tqdm_enable=False,
          tensorboard_log_enable=False,
          train_device=torch.device('cpu'),
          validate_device=torch.device('cuda:0'))
