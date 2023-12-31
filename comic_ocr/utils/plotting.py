from typing import Optional, Union, List, Tuple, Collection

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_images(
        images: Union[Image.Image, List],
        texts: Optional[Union[str, List]] = None,
        num_col: int = 3,
        figsize: Tuple[int, int] = (15, 10)
):
    if not isinstance(images, list):
        images = [images]

    if texts and not isinstance(texts, list):
        texts = [texts]

    plt.figure(figsize=figsize)
    col_count = min(num_col, len(images))
    row_count = (len(images) + num_col - 1) // num_col
    for i in range(len(images)):
        # noinspection PyTypeChecker
        conv_img = np.asarray(images[i])

        plt.subplot(row_count, col_count, i + 1)
        plt.imshow(conv_img)

        if texts:
            plt.title(str(texts[i]), fontsize=14, color=(1, 0, 0))
    plt.show()


def show_image(
        image: Image.Image,
        text: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
):
    show_images(
        images=[image],
        texts=[text] if text else None,
        num_col=1,
        figsize=figsize
    )


def show_image_recognition(
        image: Image.Image,
        text_recognized: str,
        text_expected: Optional[str]
):
    # noinspection PyTypeChecker
    image = np.asarray(image)
    title = f'Pred: {text_recognized}'
    if text_expected:
        title += f' | Truth: {text_expected}'

    plt.imshow(image)
    plt.title(title)
    plt.axis('off')


def plot_metrics(
        metrics: Collection[Tuple[str, List[float]]],
        num_col: int = 3,
        figsize: Tuple[int, int] = (12, 6),
):
    plt.figure(figsize=figsize)
    col_count = min(num_col, len(metrics))
    row_count = (len(metrics) + num_col - 1) // num_col
    fig, ax = plt.subplots(row_count, col_count, figsize=figsize)
    for i, (name, values) in enumerate(metrics):
        sub_plt = ax[i // col_count][i % col_count] if row_count > 1 else ax[i]
        sub_plt.set_title(name)
        sub_plt.plot(values)
        sub_plt.grid()
    plt.tight_layout(h_pad=2)
    plt.show()

def plot_losses(
        train_losses: List[float],
        val_losses: List[float],
        figsize: Tuple[int, int] = (12, 6)
):
    """
    Plots train and validation losses
    """

    # making titles
    train_title = f'Train loss average:{np.mean(train_losses[:]):.6f}'
    val_title = f'Val loss average:{np.mean(val_losses[:]):.6f}'

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(train_losses)
    ax[1].plot(val_losses)

    ax[0].set_title(train_title)
    ax[1].set_title(val_title)
    plt.show()
