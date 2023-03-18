import os

import torch

from comic_ocr.models import localization
from comic_ocr.models.localization.train import train, save_on_increasing_validate_metric
from comic_ocr.utils import files

import hashlib


def hash_file(filename):
    h = hashlib.sha1()
    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    return h.hexdigest()


def test_save_on_increasing_validate_metric(tmpdir):
    model_path = tmpdir / 'model_saving_dir/model.bin'
    model_path = str(model_path)

    model = localization.BasicLocalizationModel()
    update_func = save_on_increasing_validate_metric(model, model_path, 'acc')

    model.reset_parameters()
    update_func(1, {}, {'acc': [0.26]})
    assert os.path.exists(model_path)
    model_hash = hash_file(model_path)

    # Not save when the metric is below the previous max
    model.reset_parameters()
    update_func(2, {}, {'acc': [0.26, 0.25]})
    assert os.path.exists(model_path)
    assert model_hash == hash_file(model_path)

    # Save when the metric increase above the previous max
    model.reset_parameters()
    update_func(3, {}, {'acc': [0.26, 0.25, 0.30]})
    assert os.path.exists(model_path)
    assert model_hash != hash_file(model_path)


def test_train_to_finish_and_return_metrics():
    model = localization.BasicLocalizationModel()
    assert model.preferred_image_size == (500, 500)

    dataset = localization.LocalizationDataset.load_line_annotated_manga_dataset(
        files.get_path_project_dir('example/manga_annotated'),
        batch_image_size=model.preferred_image_size)
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

    # The training loss should be decreasing
    assert train_metrics['loss'][0] >= train_metrics['loss'][2]

    assert 'line_level_precision' in validate_metrics
    assert 'line_level_recall' in validate_metrics


def test_train_to_send_update_callback():
    model = localization.BasicLocalizationModel()
    assert model.preferred_image_size == (500, 500)

    dataset = localization.LocalizationDataset.load_line_annotated_manga_dataset(
        files.get_path_project_dir('example/manga_annotated'),
        batch_image_size=model.preferred_image_size)
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