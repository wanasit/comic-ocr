import math
import os

import torch
from torch import nn

from comic_ocr.models.recognition import recognition_model


class CRNN(recognition_model.CharBaseRecognitionModel):
    """
    A Text-Recognition Module based-on CRNN (Convolutional Recurrent Neural Network) framework:
    https://arxiv.org/abs/1507.05717
    """

    def __init__(
            self,
            feature_extraction_num_features=(64, 128, 256),
            sequential_input_size=256,
            sequential_num_features=(256, 512),
            prediction_hidden_size=1024,
            resnet=False,
            **kwargs,
    ):
        super().__init__(**kwargs, prediction_hidden_size=prediction_hidden_size)
        input_channel, input_height = self._init_feature_extraction_model(
            num_features=feature_extraction_num_features, resnet=resnet)
        self.sequential_model = nn.Sequential(
            nn.Linear(input_channel * input_height, sequential_input_size),
        )
        for i, num_features in enumerate(sequential_num_features):
            self.sequential_model.add_module(
                'RNNLayer_%d' % i,
                BidirectionalRNNBlock(sequential_input_size, num_features)
            )
            sequential_input_size = 2 * num_features
        self.sequential_model.add_module(
            'RNN_output_projection',
            nn.Linear(sequential_input_size, prediction_hidden_size))

    def _init_feature_extraction_model(self, num_features,
                                       resnet=False):
        # Todo: Try resnet
        # resnet = resnet18(pretrained=True)
        # resnet_modules = list(resnet.children())[:-3]
        if resnet:
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=True)

            # Take the resnet until the 2nd last block
            # Input Shape: [<batch>, self.input_channel, self.preferred_input_height, <input_width>]
            # Output Shape: [<batch>, 512, ceil(self.preferred_input_height / 16), ceil(<input_width> / 16)]
            resnet_modules = list(resnet.children())[:-3]
            self.feature_extraction_model = nn.Sequential(*resnet_modules)
            return 256, math.ceil(self.preferred_image_height / 16)

        self.feature_extraction_model = nn.Sequential()
        input_channel = self.input_channel
        input_height = self.input_height
        for i, num_feature in enumerate(num_features):
            self.feature_extraction_model.add_module(
                'ConvolutionLayer_%d' % i,
                nn.Sequential(
                    nn.Conv2d(input_channel, num_feature, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_feature),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            input_channel = num_feature
            input_height = input_height // 2
        return input_channel, input_height

    @staticmethod
    def create():
        return CRNN()

    @staticmethod
    def create_resnet():
        return CRNN(resnet=True)

    @staticmethod
    def create_small_model():
        return CRNN(
            feature_extraction_num_features=(64, 128),
            sequential_input_size=128,
            sequential_num_features=(128,),
            prediction_hidden_size=128,
        )


class BidirectionalRNNBlock(nn.Module):
    """
    input : visual feature [batch_size x T x input_size]
    output : contextual feature [batch_size x T x output_size]
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BidirectionalRNNBlock, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        return recurrent


if __name__ == '__main__':
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDatasetWithAugmentation
    from comic_ocr.utils import get_path_project_dir
    from comic_ocr.utils.pytorch_model import get_total_parameters_count

    model = CRNN.create_resnet()
    print('Total parameters count : ', get_total_parameters_count(model))

    dataset = RecognitionDatasetWithAugmentation.load_annotated_dataset(get_path_project_dir('example/manga_annotated'))
    dataloader = dataset.loader(batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    print('loss', loss)

    line_image = dataset.get_line_image(0)
    line_text_expected = dataset.get_line_text(0)
    line_text_recognized = model.recognize(line_image)
    # line_image.show()

    print('Expected : ', line_text_expected)
    print('Recognized : ', line_text_recognized)
