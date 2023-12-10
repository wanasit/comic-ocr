from torch import nn

from comic_ocr.models.recognition import recognition_model


class FCN(recognition_model.CharBaseRecognitionModel):
    """
    A Fully Convolutional Network (FCN) for Text-Recognition.
    """
    def __init__(
            self,
            num_features=(64, 128, 256),
            prediction_hidden_size=256,
            **kwargs,
    ):
        super().__init__(**kwargs, prediction_hidden_size=prediction_hidden_size)

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

        self.sequential_model = nn.Sequential(
            nn.Linear(input_channel * input_height, prediction_hidden_size))


if __name__ == '__main__':
    from comic_ocr.models.recognition.recognition_dataset import RecognitionDatasetWithAugmentation
    from comic_ocr.utils import get_path_project_dir
    from comic_ocr.utils.pytorch_model import get_total_parameters_count

    model = FCN()
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
