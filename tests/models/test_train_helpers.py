import os
import hashlib

from comic_ocr.models import localization, train_helpers


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
    update_func = train_helpers.callback_to_save_model_on_increasing_metric(model, model_path, 'acc')

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
