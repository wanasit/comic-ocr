from comic_ocr.models import recognition
from comic_ocr.utils.files import get_path_example_dir


def test_high_level_metrics_on_perfect_score():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = recognition.RecognitionDataset.load_annotated_dataset(dataset_dir)
    assert dataset

    class PrefectModel(recognition.recognition_model.RecognitionModel):
        def recognize(self, image, **kwargs):
            for i in range(len(dataset)):
                if dataset.get_line_image(i) == image:
                    return dataset.get_line_text(i)
            raise Exception('Unrecognized image')

    model = PrefectModel()
    metrics = recognition.calculate_high_level_metrics(model, dataset)
    assert metrics
    assert metrics['dataset_size'] == len(dataset)
    assert metrics['perfect_match_count'] == len(dataset)
    assert metrics['perfect_match_accuracy'] == 1.0
    assert metrics['similarity'] == 1.0

    model = PrefectModel()
    metrics = recognition.calculate_high_level_metrics(model, dataset, sample_size_limit=2)
    assert metrics
    assert metrics['dataset_size'] == 2
    assert metrics['perfect_match_count'] == 2
    assert metrics['perfect_match_accuracy'] == 1.0
    assert metrics['similarity'] == 1.0


def test_high_level_metrics_on_empty_response():
    dataset_dir = get_path_example_dir('manga_annotated')
    dataset = recognition.RecognitionDataset.load_annotated_dataset(dataset_dir)
    assert dataset

    class EmptyResponseModel(recognition.recognition_model.RecognitionModel):
        def recognize(self, image, **kwargs):
            return ''

    model = EmptyResponseModel()
    metrics = recognition.calculate_high_level_metrics(model, dataset)
    assert metrics
    assert metrics['dataset_size'] == len(dataset)
    assert metrics['perfect_match_count'] == 0
    assert metrics['perfect_match_accuracy'] == 0.0
    assert metrics['similarity'] == 0.0

    model = EmptyResponseModel()
    metrics = recognition.calculate_high_level_metrics(model, dataset, sample_size_limit=2)
    assert metrics
    assert metrics['dataset_size'] == 2
    assert metrics['perfect_match_count'] == 0
    assert metrics['perfect_match_accuracy'] == 0.0
    assert metrics['similarity'] == 0.0
