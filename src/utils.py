from typing import Dict, Any, Tuple, List, Optional
import json
from pathlib import Path
import csv
import random
import os
import numpy as np

import torch

from metrics import dice_coefficient


def save_dict(data: Dict[str, Any], filename: Path) -> None:
    """Save dictionary as json file
    Args:
        d: data to be dumped into a json file
        filename: filename path
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file)


def verify_exists_else_create(folder: Path) -> None:
    """Verify if the folder exists in the given path
    otherwise, create it
    Args:
        folder: path of the folder
    """
    folder.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    """set all the random seeds for reproducibility
    Args:
        seed: value from the config file
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device() -> None:
    """Get device based on GPU availability
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prep_training_log(log_folder: Path, metrics: List) -> Tuple:
    """Prepare log variable and log file for training and validation loss
    to be updated during training
    Args:
        log_folder: folder where log file has to be saved
    Returns:
        Tuple of fieldnames to be logged and empty dictionary with fieldnames
    """

    # Initialize the log file for training and val loss
    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
                 [f'train_{m}' for m in metrics] + \
                 [f'val_{m}' for m in metrics]
    epochsummary = {a: [0] for a in fieldnames}
    save_train_log(log_folder, fieldnames, epochsummary, mode='w')

    return fieldnames, epochsummary


def update_epochsummary(epochsummary: Dict[str, Any], epoch: int,
                        batch_loss: Dict[str, float],
                        batch_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Update summary of epochs from batch data
    Args:
        epochsummary: dictionary to be updated with mean of batch loss
        epoch: epoch number in the train loop
        batch_loss: dictionary of the losses in the batches in the current epoch

    Returns:
        Dictionary containing updated epochsummary
    """
    epochsummary['epoch'] = epoch
    phases = ['train', 'val']
    metric_names = list(batch_metrics['train'].keys())
    for phase in phases:
        epochsummary[f'{phase}_loss'] = np.mean(batch_loss[phase])
        for metric_name in metric_names:
            epochsummary[f'{phase}_{metric_name}'] = np.mean(
                                                             batch_metrics[phase][metric_name]
                                                            )

    return epochsummary


def save_train_log(log_folder: Path, fieldnames: List,
                   epochsummary: Optional[Dict[str, Any]], mode: str='a') -> None:
    """Save train log to log.csv in log_folder with fieldnames as header
    and epochsummary as data
    Args:
        log_folder: folder where log file has to be saved
        fieldnames: fieldnames to be used as header
        epochsummary: summary of epoch losses to be written into log.csv
    """
    with open(log_folder / 'log.csv', mode, newline='',
              encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        elif mode == 'a':
            writer.writerow(epochsummary)


def save_model(model_weights: Any, log_folder: Path):
    """Save weights of best model
    Args:
        model_weights: state_dict of model
    """
    torch.save(model_weights, log_folder / 'weights.pt')


def get_metrics(metric_names: List, y_true: torch.tensor,
                y_pred: torch.tensor, threshold):
    """Get evaluation metrics for the pair of label and prediction
    Args:
        metric_names: list of metrics to be computed
        y_true: ground truth label
        y_pred: prediction from the model
        threshold: to be applied to the prediction
    Returns: 
    """
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = []
        for sample_y_true, sample_y_pred in zip(y_true, y_pred):
            if metric_name == 'dice':
                metrics[metric_name] = dice_coefficient(sample_y_true > 0,
                                                    sample_y_pred > threshold)
            else:
                raise f"{metric_name} has not been implemented"
    return metrics
