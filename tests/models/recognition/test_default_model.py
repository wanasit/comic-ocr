from comic_ocr.models import recognition
from comic_ocr.models.recognition import RecognitionDataset
from comic_ocr.utils.files import get_path_example_dir


def test_default_model_high_level_metrics():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = RecognitionDataset.load_annotated_dataset(dataset_dir)
    assert dataset

    model = recognition.load_model()
    assert model

    metrics = recognition.calculate_high_level_metrics(model, dataset)
    assert metrics
