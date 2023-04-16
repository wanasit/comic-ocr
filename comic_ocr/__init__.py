from typing import List, Tuple, Optional

from comic_ocr import hub
from comic_ocr.models import localization
from comic_ocr.models import recognition
from comic_ocr.types import Rectangle, Paragraph, Line, ImageInput, to_image_rgb
from comic_ocr.utils.files import get_path_project_dir

_localization_model: Optional[localization.LocalizationModel] = None
_recognition_model: Optional[recognition.RecognitionModel] = None


def read_paragraphs(image: ImageInput) -> List[Paragraph]:
    image = to_image_rgb(image)
    locations = localize_paragraphs(image)
    model = get_recognition_model()

    paragraphs = []
    for paragraph_location, line_locations in locations:
        lines = []
        for line_location in line_locations:
            line_text = model.recognize(image.crop(line_location))
            lines.append(Line.of(line_text, line_location))
        paragraphs.append(Paragraph(lines=lines, location=paragraph_location))
    return paragraphs


def read_lines(image: ImageInput) -> List[Line]:
    image = to_image_rgb(image)
    line_locations = localize_lines(image)
    model = get_recognition_model()
    return [Line.of(model.recognize(image.crop(l)), l) for l in line_locations]


def localize_lines(image: ImageInput) -> List[Rectangle]:
    image = to_image_rgb(image)
    model = get_localization_model()
    return model.locate_lines(image)


def localize_paragraphs(image: ImageInput) -> List[Tuple[Rectangle, List[Rectangle]]]:
    image = to_image_rgb(image)
    model = get_localization_model()
    return model.locate_paragraphs(image)


def get_localization_model(show_download_progress=True, force_reload=False) -> localization.LocalizationModel:
    global _localization_model
    if not _localization_model:
        _localization_model = hub.download_localization_model(
            progress=show_download_progress, force_reload=force_reload, test_executing_model=True)

    return _localization_model


def get_recognition_model(show_download_progress=True, force_reload=False) -> recognition.RecognitionModel:
    global _recognition_model
    if not _recognition_model:
        _recognition_model = hub.download_recognition_model(
            progress=show_download_progress, force_reload=force_reload, test_executing_model=True)

    return _recognition_model


if __name__ == '__main__':
    from comic_ocr.utils.files import load_image
    from comic_ocr.utils import image_with_annotations

    example = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    example = load_image(get_path_project_dir('example/manga_annotated/normal_02.jpg'))
    # example = load_image(get_path_project_dir('data/manga_line_annotated/u_01.jpg'))

    example_paragraphs = read_paragraphs(example)
    example_lines = sum([p.lines for p in example_paragraphs], [])
    print(example_paragraphs)

    image_with_annotations(example, example_paragraphs, annotation_text_br_offset_y=-20).show()
    image_with_annotations(example, example_lines, annotation_text_br_offset_y=-20).show()
