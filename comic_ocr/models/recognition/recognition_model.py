import os
from typing import Union, List, Optional

import torch
from torch import nn
from torchvision import transforms
from PIL.Image import Image

from comic_ocr.models.recognition.encoding import SUPPORT_DICT_SIZE, decode

TRANSFORM_TO_TENSOR = transforms.ToTensor()
TRANSFORM_TO_GRAY_SCALE = transforms.Grayscale()

DEFAULT_INPUT_HEIGHT = 24
DEFAULT_INPUT_CHANNEL = 3


def image_to_single_input_tensor(input_height: int, image: Image) -> torch.Tensor:
    """
    Transform image to the input_tensor
    :param input_height:
    :param image:
    :return:
        Tensor [C, H=input_height, W]
    """
    w, h = image.size
    input_width = int((w / h) * input_height)
    image = image.resize((input_width, input_height))
    input = TRANSFORM_TO_TENSOR(image)
    # add noise???
    return input


def compute_ctc_loss(
        model_output: torch.Tensor,
        expected_output: torch.Tensor,
        expected_output_length: torch.Tensor
):
    """
    :param model_output: Tensor output from TextRecognizeModule shape [<batch>, <input_width>, SUPPORT_DICT_SIZE]
    :param expected_output: shape = [<batch>, <S>]
    :param expected_output_length: shape = [<batch>]
    :return:
    """
    loss_func = nn.CTCLoss(zero_infinity=True, blank=0)
    batch_size = model_output.shape[0]
    input_width = model_output.shape[1]

    # ctc_input.shape = [<input_width>, <batch>, SUPPORT_DICT_SIZE]
    # ctc_input_length.shape = [<batch>]
    ctc_input = model_output.log_softmax(2).permute(1, 0, 2)
    ctc_input_length = torch.tensor([input_width] * batch_size)
    return loss_func(ctc_input, expected_output, ctc_input_length, expected_output_length)


class RecognitionModel(nn.Module):
    r"""
    An abstract for Text-Recognition Module following the framework mentioned in deep-text-recognition-benchmark
    https://github.com/clovaai/deep-text-recognition-benchmark

    The implementation needs to provide:
    - feature_extraction_model: To extract features from input image (e.g. CNN)
    - sequential_model: To transform sequence of features into text (e.g. LSTM, RNN)

    This class:
    - passes the input image into *feature_extraction* to get the feature vector (3D)
    - flatten the feature vector (horizontally, according to text 'line' direction). Passes it to *sequential_model*
    - make final sequence prediction according to CTC framework
    """

    # Input Shape: (<batch>, self.input_channel, <any_input_width>, self.input_height)
    # Output Shape: (<batch>, <out_channel>, <out_width>, <out_height>)
    feature_extraction_model: nn.Module

    # Input Shape: (<batch>, <out_width>, <out_height> * self.input_height)
    # Output Shape: (<batch>, <out_width>, self.prediction_hidden_size)
    sequential_model: nn.Module

    def __init__(self,
                 input_channel: int = DEFAULT_INPUT_CHANNEL,
                 input_height: int = DEFAULT_INPUT_HEIGHT,
                 prediction_hidden_size: int = 256
                 ):
        super().__init__()
        self.input_channel = input_channel
        self.input_height = input_height
        self.prediction_hidden_size = prediction_hidden_size
        self.prediction = nn.Linear(self.prediction_hidden_size, SUPPORT_DICT_SIZE)

    # Input Shape: [<batch>, self.input_channel, self.input_height, <input_width>]
    # Output Shape: [<batch>, <input_width>, SUPPORT_DICT_SIZE]
    def forward(self, input):
        visual_feature = self.feature_extraction_model(input)
        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.flatten(2)

        contextual_feature = self.sequential_model(visual_feature)
        prediction = self.prediction(contextual_feature.contiguous())
        return prediction

    def compute_loss(self, dataset_batch) -> torch.Tensor:
        input = dataset_batch['input']
        expected_output = dataset_batch['output']
        expected_output_length = dataset_batch['output_length']
        return compute_ctc_loss(self(input), expected_output, expected_output_length)

    def recognize(self, tensor_or_image: Union[Image, torch.Tensor]) -> str:
        input_tensor = tensor_or_image
        if isinstance(tensor_or_image, Image):
            input_tensor = image_to_single_input_tensor(self.input_height, tensor_or_image)

        assert len(input_tensor.shape) == 3, 'The method expect only a single input image'

        prediction = self.predict_encoded_chars(input_tensor.unsqueeze(0))
        return decode(prediction[0].numpy())

    def predict_encoded_chars(self, tensor_or_image: Union[Image, torch.Tensor]) -> torch.Tensor:
        input_tensor = tensor_or_image
        if isinstance(tensor_or_image, Image):
            input_tensor = image_to_single_input_tensor(self.input_height, tensor_or_image)

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        output = self(input_tensor)  # output.shape = [<batch>, <input_width>, SUPPORT_DICT_SIZE]
        _, prediction = output.max(2)  # prediction = [<batch>, <input_width>]
        return prediction


if __name__ == '__main__':
    from comic_ocr.models.recognition.crnn.crnn import CRNN
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDataset
    from comic_ocr.utils import get_path_project_dir

    path_to_model = get_path_project_dir('data/output/models/recognition.bin')
    if os.path.exists(path_to_model):
        print('Loading an existing model...')
        model = torch.load(path_to_model)
    else:
        print('Creating a new model...')
        model = CRNN()

    dataset = RecognitionDataset.load_annotated_dataset(get_path_project_dir('example/manga_annotated'))

    line_image = dataset.get_line_image(0)
    line_text_expected = dataset.get_line_text(0)
    line_image.show()

    line_text_recognized = model.recognize(line_image)
    print('Expected : ', line_text_expected)
    print('Recognized : ', line_text_recognized)

    line_image = dataset.get_line_image(2)
    line_text_expected = dataset.get_line_text(2)
    line_image.show()

    line_text_recognized = model.recognize(line_image)
    print('Expected : ', line_text_expected)
    print('Recognized : ', line_text_recognized)
