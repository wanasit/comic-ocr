import torch
from torch import nn
from torchvision import transforms
from PIL.Image import Image

from manga_ocr.models.recognition import SUPPORT_DICT_SIZE

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


class TextRecognizeModule(nn.Module):
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
        batch_size = len(input)

        loss_func = nn.CTCLoss(zero_infinity=True)

        ctc_input = self(input)
        ctc_input = ctc_input.log_softmax(2).permute(1, 0, 2)
        ctc_input_length = torch.tensor([ctc_input.shape[0]] * batch_size)
        return loss_func(ctc_input, expected_output, ctc_input_length, expected_output_length)
