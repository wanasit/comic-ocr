from comic_ocr.dataset import generated_manga
from comic_ocr.utils.files import get_path_example_dir
from comic_ocr.models.localization.localization_utils import output_mark_image_to_tensor, output_tensor_to_image_mask
from comic_ocr.models.localization.localization_open_cv import locate_lines_in_character_mask


def test_locate_lines_in_image_mask():
    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    _, image_texts, image_masks = generated_manga.load_dataset(example_generated_dataset_dir)

    image_mask = image_masks[0]
    lines = image_texts[0]

    output_tensor = output_mark_image_to_tensor(image_mask)
    # output_tensor_to_image_mask(output_tensor).show()

    located_lines = locate_lines_in_character_mask(output_tensor)

    assert len(lines) == len(located_lines)
    lines = sorted(lines, key=lambda l: l.location.top)
    located_lines = sorted(located_lines, key=lambda l: l.top)

    for i in range(len(lines)):
        assert lines[i].location in located_lines[i]
