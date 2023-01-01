from typing import List, Union

from PIL import Image, ImageDraw, ImageFont

from comic_ocr.types import Rectangle, Paragraph, Line, Color
from comic_ocr.utils.files import get_path_project_dir


def concatenated_images(
        images: List[Image.Image],
        num_col: int = 3,
        padding: int = 3,
        background: Color = '#ffffff',
):
    img_width = max(i.width for i in images)
    img_height = max(i.height for i in images)

    col_count = min(num_col, len(images))
    row_count = (len(images) + 1) // num_col
    width = padding + (img_width + padding) * col_count
    height = padding + (img_height + padding) * row_count

    concat_image = Image.new('RGB', (width, height), background)
    for i, img in enumerate(images):
        x = padding + (i % col_count) * (img_width + padding)
        y = padding + (i // col_count) * (img_height + padding)
        concat_image.paste(img, (x, y))

    return concat_image


def image_with_annotations(
        image: Image.Image,
        annotations: List[Union[Rectangle, Paragraph, Line]],
        annotation_fill: str = '#44ff2288',
        annotation_text_fill: str = '#00bb00ff',
        annotation_text_font: ImageFont.ImageFont = ImageFont.load_default(),
        annotation_text_br_offset_x: int = 0,
        annotation_text_br_offset_y: int = 0,
):
    image = image.copy()
    draw = ImageDraw.Draw(image, 'RGBA')
    for annotation in annotations:
        location = annotation.location if hasattr(annotation, 'location') else annotation
        text = annotation.text if hasattr(annotation, 'text') else ''
        text = text.replace('’', "'")
        draw.rectangle(location, fill=annotation_fill)
        draw.text(
            location.br.move(annotation_text_br_offset_x, annotation_text_br_offset_y),
            text, annotation_text_fill, font=annotation_text_font)

    return image


if __name__ == '__main__':
    from files import load_image, find_annotation_data_for_image, get_path_project_dir
    from nb_annotation import lines_from_nb_annotation_data

    image = load_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))

    annotation_data = find_annotation_data_for_image(get_path_project_dir('example/manga_annotated/normal_01.jpg'))
    lines = lines_from_nb_annotation_data(annotation_data)

    image_with_annotations(image, annotations=lines).show()
