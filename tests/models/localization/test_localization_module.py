from manga_ocr.dataset import generated_manga
from manga_ocr.models.localization.localization_model import image_mask_to_output_tensor, output_tensor_to_image_mask, locate_lines_in_image_mask
from manga_ocr.utils.files import get_path_example_dir


def test_locate_lines_in_image_mask():

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    _, image_texts, image_masks = generated_manga.load_dataset(example_generated_dataset_dir)

    image_mask = image_masks[0]
    lines = image_texts[0]

    output_tensor = image_mask_to_output_tensor(image_mask)
    # output_tensor_to_image_mask(output_tensor).show()

    located_lines = locate_lines_in_image_mask(output_tensor)

    assert len(lines) == len(located_lines)
    lines = sorted(lines, key=lambda l: l.location.top)
    located_lines = sorted(located_lines, key=lambda l: l.top)

    for i in range(len(lines)):
        assert lines[i].location in located_lines[i]

