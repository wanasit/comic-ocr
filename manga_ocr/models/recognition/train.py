import logging
import os
from typing import Optional, Callable, List

import torch
from PIL.Image import Image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from manga_ocr.models.recognition.recognition_dataset import RecognitionDataset
from manga_ocr.models.recognition.recognition_module import TextRecognizeModule, image_to_single_input_tensor
from manga_ocr.utils import get_path_project_dir

logger = logging.getLogger(__name__)


def train(
        model: TextRecognizeModule,
        training_dataset: RecognitionDataset,
        validation_dataset: Optional[RecognitionDataset] = None,
        epoch_count: int = 1,
        epoch_callback: Optional[Callable] = None,
        tqdm_disable=False,
        optimizer=None
):
    # todo: support different batch szie
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=0)

    lr = 0.02
    weight_decay = 1e-5
    momentum=0.9
    clip_norm = 5
    optimizer = optimizer if optimizer else torch.optim.SGD(model.parameters(), lr=lr, nesterov=True,
                            weight_decay=weight_decay, momentum=momentum)

    training_losses = []
    validation_losses = []
    logger.info(f'Training {epoch_count} epochs, on {len(training_dataset)} samples ' +
                f'({len(validation_dataset)} validation samples)' if validation_dataset else '')

    for i_epoch in range(epoch_count):
        with tqdm(total=len(training_dataset), disable=tqdm_disable) as tepoch:
            tepoch.set_description(f"Epoch {i_epoch}")

            training_loss = 0.0
            for i_batch, batch in enumerate(training_dataloader):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

                batch_loss = loss.item()
                batch_size = batch['input'].size(0)

                tepoch.set_postfix(training_batch_loss=batch_loss)
                tepoch.update(batch_size)
                training_loss += batch_loss * batch_size

            training_loss = training_loss / len(training_dataset)
            training_losses.append(training_loss)

            if validation_dataset:
                validation_loss = 0.0
                valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                for i_batch, batch in enumerate(valid_dataloader):
                    batch_size = batch['input'].size(0)
                    logger.debug(f'> Validating with {batch_size} samples')
                    with torch.no_grad():
                        loss = model.compute_loss(batch)
                        validation_loss += loss.item() * batch_size
                validation_losses.append(validation_loss / len(validation_dataset))
            else:
                validation_loss = None

            tepoch.set_postfix(training_loss=training_loss, validation_loss=validation_loss)
            logger.info(f'> Finished training with training_loss={training_loss}, validation_loss={validation_loss}')

            if epoch_callback:
                epoch_callback()

    return (training_losses, validation_losses) if validation_dataset else (training_losses, None)


if __name__ == '__main__':
    from manga_ocr.models.recognition.crnn.crnn import CRNN

    path_dataset = get_path_project_dir('data/output/generate_manga_dataset')
    path_output_model = get_path_project_dir('data/output/models/recognizer.bin')

    if os.path.exists(path_output_model):
        print('Loading an existing model...')
        model = torch.load(path_output_model)
    else:
        print('Creating a new model...')
        model = CRNN()

    dataset = RecognitionDataset.load_generated_dataset(path_dataset)
    print(f'Loaded generated manga dataset of size {len(dataset)}...')

    validation_dataset = dataset.subset(to_dix=100)
    training_dataset = dataset.subset(from_idx=100, to_dix=200)

    training_dataset.get_line_image(0).show()
    training_dataset.get_line_image(1).show()
    validation_dataset.get_line_image(0).show()
    validation_dataset.get_line_image(1).show()

    train(model,
          training_dataset=training_dataset,
          validation_dataset=validation_dataset,
          epoch_callback=lambda: torch.save(model, path_output_model),
          epoch_count=10)
