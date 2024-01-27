from typing import List, Tuple, Sequence

from comic_ocr import hub, types
from comic_ocr.models import localization
from comic_ocr.models import recognition
from comic_ocr.dataset import annotated_manga
from comic_ocr.types import Rectangle, Percentage, Line, PathLike


class ComicOCRModel:

    def __init__(self,
                 localization_model: localization.LocalizationModel,
                 recognition_model: recognition.RecognitionModel):
        self._localization_model = localization_model
        self._recognition_model = recognition_model

    @staticmethod
    def download_default(show_download_progress=True, force_reload=False) -> 'ComicOCRModel':
        localization_model = hub.download_localization_model(
            progress=show_download_progress, force_reload=force_reload, test_executing_model=True)
        recognition_model = hub.download_recognition_model(
            progress=show_download_progress, force_reload=force_reload, test_executing_model=True)

        return ComicOCRModel(localization_model, recognition_model)

    @staticmethod
    def load_local(
            localization_model_path: PathLike = localization.DEFAULT_LOCAL_TRAINED_MODEL_FILE,
            recognition_model_path: PathLike = recognition.DEFAULT_LOCAL_TRAINED_MODEL_FILE) -> 'ComicOCRModel':
        localization_model = localization.load_model(localization_model_path)
        recognition_model = recognition.load_model(recognition_model_path)
        return ComicOCRModel(localization_model, recognition_model)

    def read_paragraphs(self, image: types.ImageInput) -> List[types.Paragraph]:
        image: types.ImageRGB = types.to_image_rgb(image)
        locations = self.localize_paragraphs(image)

        paragraphs = []
        for paragraph_location, line_locations in locations:
            lines = []
            for line_location in line_locations:
                line_text = self._recognition_model.recognize(image.crop(line_location))
                lines.append(types.Line.of(line_text, line_location))
            paragraphs.append(types.Paragraph(lines=lines, location=paragraph_location))
        return paragraphs

    def read_lines(self, image: types.ImageInput) -> List[types.Line]:
        image: types.ImageRGB = types.to_image_rgb(image)
        line_locations = self.localize_lines(image)
        return [types.Line.of(self._recognition_model.recognize(image.crop(l)), l) for l in line_locations]

    def localize_lines(self, image: types.ImageInput) -> List[types.Rectangle]:
        image: types.ImageRGB = types.to_image_rgb(image)
        return self._localization_model.locate_lines(image)

    def localize_paragraphs(self, image: types.ImageInput) -> List[Tuple[types.Rectangle, List[types.Rectangle]]]:
        image: types.ImageRGB = types.to_image_rgb(image)
        return self._localization_model.locate_paragraphs(image)


class ModelAccuracy:
    num_lines_matched: int = 0
    num_lines_predicted: int = 0
    num_lines_annotated: int = 0
    num_lines_text_correct: int = 0

    num_chars_predicted: int = 0
    num_chars_annotated: int = 0
    num_chars_correct: int = 0

    def include(self, predicted_lines: Sequence[Line], annotated_lines: Sequence[Line]):
        matched_indexes, unmatched_predicted_indexes, unmatched_annotated_indexes = Rectangle.match(
            [l.location for l in predicted_lines],
            [l.location for l in annotated_lines])

        for i, j in matched_indexes:
            self.num_lines_predicted += 1
            self.num_lines_annotated += 1
            self.num_lines_matched += 1

            predicted_text = predicted_lines[i].text
            annotated_text = annotated_lines[j].text
            self.num_chars_predicted += len(predicted_text)
            self.num_chars_annotated += len(annotated_text)
            if predicted_text == annotated_text:
                self.num_lines_text_correct += 1
                self.num_chars_correct += len(predicted_text)

        for i in unmatched_predicted_indexes:
            self.num_lines_predicted += 1
            self.num_chars_predicted += len(predicted_lines[i].text)

        for j in unmatched_annotated_indexes:
            self.num_lines_annotated += 1
            self.num_chars_annotated += len(annotated_lines[j].text)

    @staticmethod
    def compute(model, dataset: annotated_manga.AnnotatedMangaDataset):
        result = ModelAccuracy()

        for image, lines in zip(dataset[0], dataset[1]):
            pred_lines = model.read_lines(image)
            result.include(pred_lines, lines)
        return result

    @property
    def line_recall(self):
        return Percentage.of(self.num_lines_text_correct, self.num_lines_annotated)

    @property
    def line_precision(self):
        return Percentage.of(self.num_lines_text_correct, self.num_lines_predicted)

    @property
    def recognition_accuracy(self):
        return Percentage.of(self.num_lines_text_correct, self.num_lines_matched)

    @property
    def localization_precision(self):
        return Percentage.of(self.num_lines_matched, self.num_lines_predicted)

    @property
    def localization_recall(self):
        return Percentage.of(self.num_lines_matched, self.num_lines_annotated)


if __name__ == '__main__':
    from comic_ocr.utils.files import get_path_project_dir

    model = ComicOCRModel.download_default()
    # model = ComicOCRModel.load_local(
    #     recognition_model_path=get_path_project_dir('data/output/models/recognition_crnn_small.bin'),
    # )

    # dataset = annotated_manga.load_line_annotated_dataset(get_path_project_dir('example/manga_annotated'))
    dataset = annotated_manga.load_line_annotated_dataset(get_path_project_dir('data/manga_line_annotated'))
    accuracy = ModelAccuracy.compute(model, dataset)

    print('line_recall', accuracy.line_recall)
    print('line_precision', accuracy.line_precision)
    print('recognition_accuracy', accuracy.recognition_accuracy)
    print('localization_precision', accuracy.localization_precision)
    print('localization_recall', accuracy.localization_recall)
