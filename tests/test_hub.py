import comic_ocr

from comic_ocr.utils import files


def test_load_localization_model():
    image = files.load_image(files.get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    localization_model = comic_ocr.hub.download_localization_model(force_reload=True)
    assert localization_model

    paragraphs = comic_ocr.localize_paragraphs(image)
    assert paragraphs


def test_load_recognition_model():
    image = files.load_image(files.get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    recognition_model = comic_ocr.hub.download_recognition_model(force_reload=True)
    assert recognition_model
