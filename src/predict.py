from typing import Any
from pathlib import Path
import os

from yaml import safe_load
import torch

import utils
from data import get_dataloader
from models import create_deeplabv3
from visualization import save_predictions, save_predictions_trval


CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'


def predict(model: Any=None):
    """Prediction module used to predict on test, train or val sets
    Args:
        model: passed when called from main.py
    """

    with open(CONFIG_FILENAME, encoding='utf-8') as config_file:
        config = safe_load(config_file)

    data_directory = TOP_DIRECTORY / config['data dir']

    if model is None:
        model = create_deeplabv3()
        model.load_state_dict(torch.load(LOG_FOLDER / 'weights.pt'))

    model.eval().to(utils.get_device())

    # reproducibility
    utils.set_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if 'test' in config['predict_on']:

        test_data_directory = data_directory / config['test dir']
        image_generator = utils.get_test_images(test_data_directory)

        test_save_folder = LOG_FOLDER / 'test_predictions'

        utils.verify_exists_else_create(test_save_folder)

        save_predictions(image_generator, model, config['threshold'], test_save_folder)

    if 'val' in config['predict_on'] or 'train' in config['predict_on']:

        dataloaders = get_dataloader(data_directory, config['images dir'], config['labels dir'],
                                seed=config['seed'], batch_size=1)

    if 'val' in config['predict_on']:
        val_save_folder = LOG_FOLDER / 'val_predictions'
        utils.verify_exists_else_create(val_save_folder)
        metrics = save_predictions_trval(dataloaders['val'], model, config['threshold'], val_save_folder, config['metrics'])
        utils.save_dict(metrics, val_save_folder / 'metrics.json')

    if 'train' in config['predict_on']:
        train_save_folder = LOG_FOLDER / 'train_predictions'
        utils.verify_exists_else_create(train_save_folder)
        metrics = save_predictions_trval(dataloaders['train'], model, config['threshold'], train_save_folder, config['metrics'])
        utils.save_dict(metrics, train_save_folder / 'metrics.json')



if __name__ == '__main__':
    predict()
