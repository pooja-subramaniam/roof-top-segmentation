import json
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import csv
import random
import os

import torch


def save_dict(d: Dict[str, Any], filename: Path) -> None:
    """Save dictionary as json file
    Args:
        d: data to be dumped into a json file
        filename: filename path
    """
    with open(filename, 'w') as f:
        json.dump(d, f)


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


def prep_training_log(log_folder: Path) -> Tuple:
    """Prepare log variable and log file for training and validation loss
    to be updated during training
    Args:
        log_folder: folder where log file has to be saved
    Returns:
        Tuple of fieldnames to be logged and empty dictionary with fieldnames
    """

    # Initialize the log file for training and val loss
    fieldnames = ['epoch', 'train_loss', 'val_loss']
    epochsummary = {a: [0] for a in fieldnames}
    save_train_log(log_folder, fieldnames, epochsummary, mode='w')

    return fieldnames, epochsummary


def update_epochsummary(epochsummary: Dict[str, Any], epoch: int,
                        batch_loss: Dict[str, float]) -> Dict[str, Any]:
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
    for phase in phases:
        epochsummary[f'{phase}_loss'] = np.mean(batch_loss[phase])

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
    with open(log_folder / 'log.csv', mode, newline='') as csvfile:
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
