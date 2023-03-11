import os

import torch
from torch import nn

import torch.nn.functional as F
from torchvision.models import resnet18

from comic_ocr.models.recognition.recognition_model import RecognitionModel, image_to_single_input_tensor

# TODO: Remove this
# Ref: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CRNN(RecognitionModel):
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
            **kwargs,
    ):
        super().__init__(**kwargs, prediction_hidden_size=prediction_hidden_size)

        # Todo: Try resnet
        # resnet = resnet18(pretrained=True)
        # resnet_modules = list(resnet.children())[:-3]
        self.feature_extraction_model = nn.Sequential()

        input_channel = self.input_channel
        input_height = self.input_height
        for i, num_features in enumerate(feature_extraction_num_features):
            self.feature_extraction_model.add_module(
                'ConvolutionLayer_%d' % i,
                nn.Sequential(
                    nn.Conv2d(input_channel, num_features, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            input_channel = num_features
            input_height = input_height // 2

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

    @staticmethod
    def create():
        return CRNN()

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
    from comic_ocr.utils.files import get_path_project_dir
    from comic_ocr.utils.pytorch_model import get_total_parameters_count
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset
    from torch.utils.data import DataLoader

    recognizer = CRNN.create_small_model()
    print(get_total_parameters_count(recognizer))

    # recognizer = CRNN.create()
    # print(get_total_parameters_count(recognizer))

    dataset = RecognitionDataset.load_annotated_dataset(recognizer, get_path_project_dir('data/manga_line_annotated'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(recognizer.recognize(dataset.get_line_image(0)), dataset.get_line_text(0))

    batch = next(iter(dataloader))
    loss = recognizer.compute_loss(batch)
    print('loss', loss)
