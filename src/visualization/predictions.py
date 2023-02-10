from typing import Generator, Tuple, Any, List, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils

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
        pred = utils.get_predictions(image, model, threshold)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[0].set_axis_off()
        axs[0].set_title('image')
        axs[1].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], cmap='gray')
        axs[1].set_axis_off()
        axs[1].set_title('prediction')

        axs[2].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[2].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], alpha=0.5)
        axs[2].set_axis_off()
        axs[2].set_title('prediction on image')

        fig.suptitle(f"Prediction for image {image_name.split('.')[0]}")
        fig.tight_layout()
        fig.savefig(save_folder / f'prediction_{image_name}', dpi=200)
        plt.close('all')


def save_predictions_trval(data_loader: Any, model: Any,
                           threshold: float, save_folder: Path, metric_names: List) -> Dict[str, Any]:
    """Save prediction images for train or val data that have labels and return metrics
    Args:
        image_generator: yields the image and image name to be used for prediction
        model: model to be used for prediction
        threshold: threshold to be used for saving the segmentation label
    Returns:
        individual prediction metric and mean prediction metric
    """
    metrics = {metric_name: [] for metric_name in metric_names}
    for i, sample in enumerate(iter(data_loader)):
        image, label = sample['image'], sample['label']
        print(f"Saving prediction for image {i+1}")
        pred = utils.get_predictions(image, model, threshold)
        per_sample_metrics = utils.get_metrics(metric_names, label[0].numpy(), pred, threshold)
        for metric_name in metric_names:
            metrics[metric_name].append(per_sample_metrics[metric_name])

        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[0].set_axis_off()
        axs[0].set_title('image')
        axs[1].imshow(np.transpose(label[0].numpy(), (1, 2, 0))[:,:,0], cmap='gray')
        axs[1].set_axis_off()
        axs[1].set_title('label')

        axs[2].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], cmap='gray')
        axs[2].set_axis_off()
        axs[2].set_title('prediction')

        axs[3].imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        axs[3].imshow(np.transpose(pred, (1, 2, 0))[:,:,0], alpha=0.5)
        axs[3].set_axis_off()
        axs[3].set_title('prediction on image')

        fig.suptitle(f"Prediction for image {i + 1}")
        fig.tight_layout()
        fig.savefig(save_folder / f'prediction_{i + 1}', dpi=200)
        plt.close('all')

    for metric_name in metric_names:
        metrics[f'mean_{metric_name}'] = np.asarray(metrics[metric_name]).mean()

    return metrics
