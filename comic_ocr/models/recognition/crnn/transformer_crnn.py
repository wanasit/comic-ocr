import os

import torch
from torch import nn
from comic_ocr.models.recognition import recognition_model
from comic_ocr.models.recognition.crnn import crnn


class TransformerCRNN(crnn.CRNN):
    """
    A Text-Recognition Module based-on CRNN (Convolutional Recurrent Neural Network) framework:
    https://arxiv.org/abs/1507.05717
    """

    def __init__(
            self,
            feature_extraction_num_features=(64, 128, 256),
            sequential_input_size=256,
            sequential_transformer_encoder_nhead=8,
            sequential_transformer_encoder_layers=3,
            prediction_hidden_size=1024,
            resnet=False,
            **kwargs,
    ):
        super().__init__(**kwargs, prediction_hidden_size=prediction_hidden_size)

        # Todo: Try resnet
        # resnet = resnet18(pretrained=True)
        # resnet_modules = list(resnet.children())[:-3]
        input_channel, input_height = self._init_feature_extraction_model(
            num_features=feature_extraction_num_features, resnet=resnet)

        self.sequential_model = nn.Sequential(
            nn.Linear(input_channel * input_height, sequential_input_size),
            nn.TransformerEncoder(num_layers=sequential_transformer_encoder_layers,
                                  encoder_layer=nn.TransformerEncoderLayer(d_model=sequential_input_size,
                                                                           nhead=sequential_transformer_encoder_nhead)),
            nn.Linear(sequential_input_size, prediction_hidden_size)
        )

    @staticmethod
    def create():
        return TransformerCRNN()

    @staticmethod
    def create_resnet():
        return TransformerCRNN(resnet=True)


if __name__ == '__main__':
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDatasetWithAugmentation
    from comic_ocr.utils import get_path_project_dir
    from comic_ocr.utils.pytorch_model import get_total_parameters_count

    model = TransformerCRNN.create()
    print('Total parameters count : ', get_total_parameters_count(model))

    dataset = RecognitionDatasetWithAugmentation.load_annotated_dataset(get_path_project_dir('example/manga_annotated'))
    dataloader = dataset.loader(batch_size=2, shuffle=False, num_workers=0)
    #
    batch = next(iter(dataloader))
    loss = model.compute_loss(batch)
    print('loss', loss)

    line_image = dataset.get_line_image(0)
    line_text_expected = dataset.get_line_text(0)
    line_text_recognized = model.recognize(line_image)
    # line_image.show()

    print('Expected : ', line_text_expected)
    print('Recognized : ', line_text_recognized)
