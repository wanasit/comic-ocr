from manga_ocr.generator.manga_generator import MangaGenerator


def test_manga_generator():

    generator = MangaGenerator.create()
    image, text_areas = generator.generate(output_size=(750, 750))

    assert image.size == (750, 750)
    assert 3 < len(text_areas) < 8


