from torch.utils.data import DataLoader

from comic_ocr.models.localization.conv_unet.conv_unet import ConvUnet
from comic_ocr.models.localization.localization_dataset import LocalizationDataset
from comic_ocr.models.localization.localization_utils import image_to_tensor
from comic_ocr.models.localization.train import train
from comic_ocr.utils.files import get_path_example_dir


def test_conv_unet_forward():
    model = ConvUnet()
    assert model.preferred_image_size == (500, 500)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_dataset(model, example_generated_dataset_dir)

    input_image = image_to_tensor(dataset.images[0]).unsqueeze(0)
    output_char, output_line = model(input_image)
    assert output_char.shape == (1, 500, 500)
    assert output_line.shape == (1, 500, 500)


def test_conv_unet_loss():
    model = ConvUnet()
    assert model.preferred_image_size == (500, 500)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_dataset(model, example_generated_dataset_dir)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    assert loss.item() > 0


def test_conv_unet_training():
    model = ConvUnet()
    assert model.preferred_image_size == (500, 500)

    example_generated_dataset_dir = get_path_example_dir('manga_generated')
    dataset = LocalizationDataset.load_generated_dataset(model, example_generated_dataset_dir)

    train(model, train_dataset=dataset, tqdm_disable=True)
