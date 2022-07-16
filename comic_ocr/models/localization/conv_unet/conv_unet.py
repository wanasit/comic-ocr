import os

import torch
from torch import nn

import torch.nn.functional as F

from comic_ocr.models.localization.localization_model import LocalizationModel
from comic_ocr.utils.files import load_images


class ConvUnet(LocalizationModel):

    def __init__(self):
        super(ConvUnet, self).__init__()

        self.down_conv_3 = ConvWithPoolingToHalfSize(3, num_output_channel=8)
        self.down_conv_2 = ConvWithPoolingToHalfSize(8, kernel_size=5, padding=2, num_output_channel=16)
        self.down_conv_1 = ConvWithPoolingToHalfSize(16, kernel_size=5, padding=2, num_output_channel=32)
        self.down_conv_0 = ConvWithPoolingToHalfSize(32, kernel_size=5, padding=2, num_output_channel=64)

        self.up_conv1 = DoubleConvWithSecondInput(64, 32, num_output_channel=64)
        self.up_conv2 = DoubleConvWithSecondInput(64, 16, num_output_channel=64)
        self.up_conv3 = DoubleConvWithSecondInput(64, 8, num_output_channel=64)

        self.output_conv_char = nn.Sequential(
            nn.Conv2d(64 + 3, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.output_conv_line = nn.Sequential(
            nn.Conv2d(64 + 3, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        down_3 = self.down_conv_3(x)
        down_2 = self.down_conv_2(down_3)
        down_1 = self.down_conv_1(down_2)
        down_0 = self.down_conv_0(down_1)

        up0 = down_0

        up1 = F.interpolate(up0, down_1.size()[2:], mode='bilinear', align_corners=False)
        up1 = self.up_conv1(up1, down_1)

        up2 = F.interpolate(up1, down_2.size()[2:], mode='bilinear', align_corners=False)
        up2 = self.up_conv2(up2, down_2)

        up3 = F.interpolate(up2, down_3.size()[2:], mode='bilinear', align_corners=False)
        up3 = self.up_conv3(up3, down_3)

        y = F.interpolate(up3, x.size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, x], dim=1)

        y_char = self.output_conv_char(y)
        y_line = self.output_conv_line(y)

        return y_char[:, 0, :], y_line[:, 0, :]


class ConvWithPoolingToHalfSize(nn.Module):
    def __init__(self,
                 num_input_channel,
                 num_output_channel,
                 kernel_size=3,
                 padding=1):
        super(ConvWithPoolingToHalfSize, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channel, num_output_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_output_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConvWithSecondInput(nn.Module):
    def __init__(self,
                 num_main_input_channel,
                 num_second_input_channel,
                 num_output_channel):
        super(DoubleConvWithSecondInput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_second_input_channel + num_main_input_channel, num_main_input_channel, kernel_size=1),
            nn.BatchNorm2d(num_main_input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_main_input_channel, num_output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, second_input):
        x = torch.cat([input, second_input], dim=1)
        x = self.conv(x)
        return x



if __name__ == '__main__':
    from torchvision import models
    from torchvision.models.vgg import model_urls
    from comic_ocr.utils.pytorch_model import get_total_parameters_count
    module_path = os.path.dirname(__file__)
    input_images = load_images(module_path + "/../../../out/generate/input/*.jpg")
    output_images = load_images(module_path + "/../../../out/generate/output/*.jpg")

    model = ConvUnet()
    print(get_total_parameters_count(model))
    # input = model.image_to_input(input_images[0]).unsqueeze(0)
    # output_char, output_mask = model(input)
