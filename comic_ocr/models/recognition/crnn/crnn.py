import os

import torch
from torch import nn

import torch.nn.functional as F
from torchvision.models import resnet18

from comic_ocr.models.recognition.recognition_model import RecognitionModel, image_to_single_input_tensor

# TODO: Remove this
# Ref: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CRNN(RecognitionModel):
    """
    A Text-Recognition Module based-on CRNN (Convolutional Recurrent Neural Network) framework:
    https://arxiv.org/abs/1507.05717
    """

    def __init__(
            self,
            feature_extraction_num_features=128,
            sequential_input_size=64,
            sequential_hidden_size=64,
            sequential_num_layers=3,
            prediction_hidden_size=128,
            **kwargs,
    ):
        super().__init__(**kwargs, prediction_hidden_size=prediction_hidden_size)

        # Todo: Try resnet
        # resnet = resnet18(pretrained=True)
        # resnet_modules = list(resnet.children())[:-3]

        self.feature_extraction_model = nn.Sequential(
            # nn.Sequential(*resnet_modules),
            nn.Sequential(
                nn.Conv2d(self.input_channel, feature_extraction_num_features, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feature_extraction_num_features),
                nn.ReLU(inplace=True)
            )
        )

        self.sequential_model = nn.Sequential(
            nn.Linear(feature_extraction_num_features * self.input_height, sequential_input_size),
            BidirectionalRNNBlock(
                input_size=sequential_input_size,
                hidden_size=sequential_hidden_size,
                num_layers=sequential_num_layers,
                output_size=self.prediction_hidden_size)
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
    from comic_ocr.utils.files import get_path_example_dir, load_image
    from comic_ocr.utils.pytorch_model import get_total_parameters_count
    recognizer = CRNN()
    print(get_total_parameters_count(recognizer))
    image = load_image(get_path_example_dir('manga_annotated/normal_01.jpg'))
    input = image_to_single_input_tensor(recognizer.input_height, image)
    recognizer(input.unsqueeze(0))
