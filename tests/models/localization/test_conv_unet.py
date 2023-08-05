import pytest
from torch.utils.data import DataLoader

from comic_ocr.models.localization.conv_unet import conv_unet
from comic_ocr.models.localization.localization_dataset import LocalizationDatasetWithAugmentation
from comic_ocr.models.localization.localization_utils import image_to_input_tensor
from comic_ocr.models.localization.train import train
from comic_ocr.utils.files import get_path_example_dir


@pytest.mark.parametrize('model_class', [conv_unet.BaselineConvUnet, conv_unet.DeepConvUnet])
def test_conv_unet_forward(model_class):
    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        example_generated_dataset_dir)

    model = model_class()
    input_image = image_to_input_tensor(dataset.images[0]).unsqueeze(0)
    output_char, output_line, _ = model(input_image)
    assert output_char.shape == (1, 768, 768)
    assert output_line.shape == (1, 768, 768)


@pytest.mark.parametrize('model_class', [conv_unet.BaselineConvUnet, conv_unet.DeepConvUnet])
def test_conv_unet_loss(model_class):
    model = model_class()
    assert model.preferred_image_size == (500, 500)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        example_generated_dataset_dir, batch_image_size=model.preferred_image_size)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    assert loss.item() > 0


def test_conv_unet_training():
    model = conv_unet.BaselineConvUnet()
    assert model.preferred_image_size == (500, 500)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDatasetWithAugmentation.load_generated_manga_dataset(
        example_generated_dataset_dir, batch_image_size=model.preferred_image_size)

    train('testing_model', model, train_dataset=dataset, train_epoch_count=1, tqdm_enable=False,
          tensorboard_log_enable=False)
