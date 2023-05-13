import torch
from torch.utils.data import DataLoader

from comic_ocr.dataset import generated_manga
from comic_ocr.utils.files import get_path_example_dir
from comic_ocr.models.localization.localization_utils import image_mask_to_output_tensor, output_tensor_to_image_mask
from comic_ocr.models.localization.localization_open_cv import locate_lines_in_character_mask
from comic_ocr.models.localization import localization_model
from comic_ocr.models.localization import localization_dataset


def test_weighted_bce_loss():
    loss = localization_model.WeightedBCEWithLogitsLoss(1.0, pos_weight=2.0)

    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert loss(y, torch.zeros_like(y)) > loss(y, y)
    assert loss(y, torch.ones_like(y)) > loss(y, y)

    half_loss = localization_model.WeightedBCEWithLogitsLoss(0.5, pos_weight=2.0)
    assert loss(y, torch.zeros_like(y)) == half_loss(y, torch.zeros_like(y)) * 2


def test_weighted_dice_loss():
    loss = localization_model.WeightedDiceLoss(0.5)

    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert loss(y, y) == 0.0
    assert loss(y, torch.zeros_like(y)) > 0.0
    assert loss(y, torch.ones_like(y)) > 0.0

    half_loss = localization_model.WeightedDiceLoss(0.25)
    assert loss(y, torch.zeros_like(y)) == half_loss(y, torch.zeros_like(y)) * 2
