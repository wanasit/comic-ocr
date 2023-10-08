import os
from typing import Union, Optional, Callable

import torch
from torch import nn
from PIL.Image import Image

from comic_ocr.models import transforms
from comic_ocr.models.recognition.encoding import SUPPORT_DICT_SIZE, decode

DEFAULT_INPUT_HEIGHT = 24
DEFAULT_INPUT_CHANNEL = 3

TransformImageToTensor = Callable[[Union[Image, torch.Tensor]], torch.Tensor]


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
    """An abstraction for recognition models (nn.Module).

    A recognition model takes an image of line (arbitrary width) and outputs text.
    All recognition model subclasses need to provide/implement:
    - recognize() that takes an image of line and outputs text
    - compute_loss() that computes the loss given a batch of data from RecognitionDataset

    The model's forward() or __call__() method is NOT expected to be called directly by the user.
    """

    def __init__(self, preferred_image_height: Optional[int] = None, **kwargs):
        super().__init__()
        self._preferred_image_height = preferred_image_height

    @property
    def preferred_image_height(self) -> Optional[int]:
        return self._preferred_image_height

    def forward(self, **kwargs) -> torch.Tensor:
        raise AttributeError("The model does not support forward operation")

    def recognize(self, image: Image, device: Optional[torch.device] = None) -> str:
        """Recognize text from an image of line.

        Args:
            image: An image of line.
            device: The device to run the model on.

        Returns:
            The recognized text.
        """
        raise NotImplementedError

    def compute_loss(self, dataset_batch, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute the loss given a batch of data from RecognitionDataset.

        Args:
            dataset_batch: A batch of data from RecognitionDataset.
            device: The device to run the model on.

        Returns:
            The computed loss.
        """
        raise NotImplementedError


class CharBaseRecognitionModel(RecognitionModel):
    """A recognition model based on characters encoding and deep-text-recognition-benchmark framework.

    Ref: https://github.com/clovaai/deep-text-recognition-benchmark

    In this framework, the model is composed of two parts/sub-models:
    - feature_extraction_model: To extract features from input image (e.g. CNN)
    - sequential_model: To transform sequence of features into text (e.g. LSTM, RNN)

    The recognition works as follows:
    - Pass the input image into `feature_extraction_model` to get the feature vector (3D)
    - Flatten the feature vector (horizontally, according to text 'line' direction).
    - Passes the flatten vector to `sequential_model`.
    - Passes `sequential_model` output to `prediction` layer to get the final prediction of the encoding dictionary.
    - Make final sequence prediction according to CTC framework.
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
                 prediction_hidden_size: int = 256,
                 **kwargs):
        super().__init__(preferred_image_height=input_height, **kwargs)
        self.input_channel = input_channel
        self.input_height = input_height
        self.prediction_hidden_size = prediction_hidden_size
        self.prediction = nn.Linear(self.prediction_hidden_size, SUPPORT_DICT_SIZE)

    # Input Shape: [<batch>, self.input_channel, self.preferred_input_height, <input_width>]
    # Output Shape: [<batch>, <input_width>, SUPPORT_DICT_SIZE]
    def forward(self, input_tensor):
        visual_feature = self.feature_extraction_model(input_tensor)
        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.flatten(2)

        contextual_feature = self.sequential_model(visual_feature)
        if isinstance(contextual_feature, tuple):
            contextual_feature = contextual_feature[0]

        prediction = self.prediction(contextual_feature.contiguous())
        return prediction

    def compute_loss(self, dataset_batch, device: Optional[torch.device] = None) -> torch.Tensor:
        expected_output = dataset_batch['text_encoded']
        expected_output_length = dataset_batch['text_length']
        image_tensor = dataset_batch['image']
        image_tensor = self._resize_tensor_to_input_height(image_tensor)
        return compute_ctc_loss(self(image_tensor), expected_output, expected_output_length)

    def recognize(self, tensor_or_image: Union[Image, torch.Tensor], device: Optional[torch.device] = None) -> str:
        prediction = self.predict_encoded_chars(tensor_or_image)
        return decode(prediction[0].numpy())

    def predict_encoded_chars(
            self,
            tensor_or_image: Union[Image, torch.Tensor],
            device: Optional[torch.device] = None) -> torch.Tensor:
        input_tensor = tensor_or_image
        if isinstance(tensor_or_image, Image):
            input_tensor = transforms.image_to_tensor(tensor_or_image)

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = self._resize_tensor_to_input_height(input_tensor)
        output = self(input_tensor)  # output.shape = [<batch>, <input_width>, SUPPORT_DICT_SIZE]
        _, prediction = output.max(2)  # prediction = [<batch>, <input_width>]
        return prediction

    def _resize_tensor_to_input_height(self, tensor: torch.Tensor):
        assert len(tensor.shape) == 4
        tensor_height = tensor.shape[-2]
        if tensor_height == self.preferred_image_height:
            return tensor

        scale_factor = self.preferred_image_height / tensor_height
        return nn.functional.interpolate(tensor, scale_factor=scale_factor, recompute_scale_factor=True,
                                         mode='bilinear', align_corners=False)


class BasicCharBaseRecognitionModel(CharBaseRecognitionModel):
    """A basic implementation for the CharBaseRecognitionModel to be used for testing.

    The model:
    - uses a single-layer conv2d as feature_extraction_model
    - uses a single-layer LSTM as sequential_model
    """

    def __init__(self,
                 feature_hidden_size: int = 32,
                 prediction_hidden_size: int = 64, **kwargs):
        super().__init__(prediction_hidden_size=prediction_hidden_size, **kwargs)
        self.feature_extraction_model = nn.Sequential(
            nn.Conv2d(self.input_channel, feature_hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.sequential_model = nn.Sequential(
            nn.LSTM(feature_hidden_size * self.input_height, self.prediction_hidden_size),
        )


#
# class RecognitionInputTransform(nn.Module):
#     def __init__(self, input_height):
#         super().__init__()
#         self.input_height = input_height
#         self._to_tensor = transforms.ToTensor()
#         self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#
#     def __call__(self, tensor_or_image: Union[Image, torch.Tensor]):
#         input_tensor = tensor_or_image
#         if isinstance(tensor_or_image, Image):
#             tensor_or_image = self._check_and_resize(tensor_or_image)
#             input_tensor = self._to_tensor(tensor_or_image)
#         return self._normalize(input_tensor)
#
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"
#
#     def _check_and_resize(self, image: Image) -> Image:
#         w, h = image.size
#         if h == self.input_height:
#             return image
#         input_width = int((w / h) * self.input_height)
#         return image.resize((input_width, self.input_height))


if __name__ == '__main__':
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDatasetWithAugmentation
    from comic_ocr.utils import get_path_project_dir

    model = BasicCharBaseRecognitionModel()
    # path_to_model = get_path_project_dir('data/output/models/recognition.bin')
    # if os.path.exists(path_to_model):
    #     print('Loading an existing model...')
    #     model = torch.load(path_to_model)

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
