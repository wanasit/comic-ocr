import os
from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch import nn

import torch.nn.functional as F

from comic_ocr.models.localization.conv_unet.conv_unet import ConvWithPoolingToHalfSize, DoubleConvWithSecondInput
from comic_ocr.models.localization.localization_model import LocalizationModel
from comic_ocr.utils.files import load_images


class AbstractConvFPN(LocalizationModel):
    def __init__(self,
                 transformed_channel_size=64 + 3,
                 hidden_size_output_char=16,
                 hidden_size_output_line=16,
                 **kwargs):
        super(AbstractConvFPN, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.up_layer_predicts = nn.ModuleList()
        self.output_conv_char = nn.Sequential(
            nn.Conv2d(transformed_channel_size, hidden_size_output_char, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size_output_line, 1, kernel_size=1),
        )
        self.output_conv_line = nn.Sequential(
            nn.Conv2d(transformed_channel_size, hidden_size_output_line, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size_output_line, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        down_outputs = [x]
        for i, layer in enumerate(self.down_layers):
            down = layer(down_outputs[i])
            down_outputs.append(down)

        up_outputs = [down_outputs.pop()]
        for i, layer in enumerate(self.up_layers):
            up = F.interpolate(up_outputs[i], down_outputs[-i - 1].size()[2:], mode='bilinear', align_corners=False)
            up = layer(up, down_outputs[-i - 1])
            up_outputs.append(up)

        up_predicts = []
        for i, up_output in enumerate(up_outputs[1:]):
            self.up_layer_predicts[i](up_output)
            up_predict = F.interpolate(up_output, x.size()[2:], mode='bilinear', align_corners=False)
            up_predicts.append(up_predict)

        y = torch.cat(up_predicts, dim=1)
        y_char = self.output_conv_char(y)
        y_line = self.output_conv_line(y)
        return y_char[:, 0, :], y_line[:, 0, :], up_predicts


class BaselineConvFPN(AbstractConvFPN):
    def __init__(self,
                 **kwargs):
        super(BaselineConvFPN, self).__init__(transformed_channel_size=64 * 3,
                                              hidden_size_output_char=64, hidden_size_output_line=64, **kwargs)
        self.down_layers.append(ConvWithPoolingToHalfSize(3, kernel_size=5, num_output_channel=8))
        self.down_layers.append(ConvWithPoolingToHalfSize(8, kernel_size=5, padding=2, num_output_channel=16))
        self.down_layers.append(ConvWithPoolingToHalfSize(16, kernel_size=5, padding=2, num_output_channel=32))
        self.down_layers.append(ConvWithPoolingToHalfSize(32, kernel_size=5, padding=2, num_output_channel=64))

        self.up_layers.append(DoubleConvWithSecondInput(64, 32, num_output_channel=64))
        self.up_layer_predicts.append(nn.Conv2d(64, 1, kernel_size=1))

        self.up_layers.append(DoubleConvWithSecondInput(64, 16, num_output_channel=64))
        self.up_layer_predicts.append(nn.Conv2d(64, 1, kernel_size=1))

        self.up_layers.append(DoubleConvWithSecondInput(64, 8, num_output_channel=64))
        self.up_layer_predicts.append(nn.Conv2d(64, 1, kernel_size=1))


if __name__ == '__main__':
    from torchvision import models
    from comic_ocr.utils.pytorch_model import get_total_parameters_count
    from comic_ocr.models.localization.localization_dataset import LocalizationDataset
    from comic_ocr.utils import get_path_project_dir
    from torch.utils.data import DataLoader

    model = BaselineConvFPN()
    print(get_total_parameters_count(model))

    path_dataset = get_path_project_dir('data/manga_line_annotated')
    dataset = LocalizationDataset.load_line_annotated_manga_dataset(path_dataset,
                                                                    batch_image_size=model.preferred_image_size)
    # try predicting
    output_char, output_mask, up_predicts = model(dataset[0]['input'].unsqueeze(0))

    # try loss calculation
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    batch = next(iter(train_dataloader))
    loss = model.compute_loss(batch)
    print(loss)
