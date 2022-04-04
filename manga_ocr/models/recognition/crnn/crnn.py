import os

import torch
from torch import nn

import torch.nn.functional as F
from torchvision.models import resnet18

from manga_ocr.models.recognition.recognition_module import TextRecognizeModule, image_to_single_input_tensor

# TODO: Remove this
# Ref: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CRNN(TextRecognizeModule):
    """
    A Text-Recognition Module based-on CRNN (Convolutional Recurrent Neural Network) framework:
    https://arxiv.org/abs/1507.05717

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        resnet = resnet18(pretrained=True)
        resnet_modules = list(resnet.children())[:-3]
        self.feature_extraction_model = nn.Sequential(
            #nn.Sequential(*resnet_modules), # Todo: use resnet
            nn.Sequential(
                nn.Conv2d(self.input_channel, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        )

        self.sequential_model = nn.Sequential(
            nn.Linear(256 * 24, 256),
            BidirectionalRNNBlock(input_size=256, hidden_size=256, output_size=self.prediction_hidden_size)
        )


class BidirectionalRNNBlock(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BidirectionalRNNBlock, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


if __name__ == '__main__':
    from manga_ocr.utils import load_image, get_path_example_dir
    recognizer = CRNN()
    image = load_image(get_path_example_dir('annotated_manga/normal_01.jpg'))
    input = image_to_single_input_tensor(recognizer.input_height, image)
    recognizer(input.unsqueeze(0))
