from typing import List, Union

from PIL import Image, ImageDraw, ImageFont

from manga_ocr.typing import Rectangle, Paragraph, Line
from manga_ocr.utils.files import get_path_project_dir

def image_with_annotations(
        image: Image.Image,
        annotations: List[Union[Rectangle, Paragraph, Line]],
        annotation_fill: str = '#44ff2288',
        annotation_text_fill: str = '#00bb00ff',
        annotation_text_font: ImageFont.ImageFont = ImageFont.load_default(),
):
    image = image.copy()
    draw = ImageDraw.Draw(image, 'RGBA')
    for annotation in annotations:
        location = annotation.location if hasattr(annotation, 'location') else annotation
        text = annotation.text if hasattr(annotation, 'text') else ''

        draw.rectangle(location, fill=annotation_fill)
        draw.text(location.br, text, annotation_text_fill, font=annotation_text_font)

    return image


if __name__ == '__main__':
    from files import load_image, find_annotation_data_for_image, get_path_project_dir
    from nb_annotation import lines_from_nb_annotation_data
    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    annotation_data = find_annotation_data_for_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    lines = lines_from_nb_annotation_data(annotation_data)

    image_with_annotations(image, annotations=lines).show()
