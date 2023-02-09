from typing import Generator, Tuple, Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import get_predictions


def save_predictions(image_generator: Generator[Tuple[torch.Tensor, str], None, None],
                     model: Any, threshold: float, save_folder: Path) -> None:
    """Save prediction images for test data that does not have labels
    Args:
        image_generator: yields the image and image name to be used for prediction
        model: model to be used for prediction
        threshold: threshold to be used for saving the segmentation label
    """
    for image, image_name in image_generator:
        print(f"Saving prediction for image {image_name.split('.')[0]}")
        pred = get_predictions(image, model, threshold)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[0].set_axis_off()
        axs[0].set_title('image')
        axs[1].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], cmap='gray')
        axs[1].set_axis_off()
        axs[1].set_title('label')
        axs[2].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[2].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], alpha=0.5)
        axs[2].set_axis_off()
        axs[2].set_title('label on image')

        fig.suptitle(f"Prediction for image {image_name.split('.')[0]}")
        fig.tight_layout()
        fig.savefig(save_folder / f'prediction_{image_name}', dpi=200)
