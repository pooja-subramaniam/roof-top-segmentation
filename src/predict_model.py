from typing import Any
from pathlib import Path
import os

from yaml import safe_load
import torch

import utils
from models import create_deeplabv3
from visualization import save_predictions


CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'


def predict(model: Any=None):
    """Prediction module used to predict on test, train or val sets
    Args:
        log_folder: where predictions can be saved
        predict_on: list of sets for predictions to be made and saved as images
    """

    with open(CONFIG_FILENAME, encoding='utf-8') as config_file:
        config = safe_load(config_file)

    data_directory = TOP_DIRECTORY / config['data dir']

    if model is None:
        model = create_deeplabv3()
        model.load_state_dict(torch.load(LOG_FOLDER / 'weights.pt'))

    model.eval().to(utils.get_device())

    if 'test' in config['predict_on']:

        test_data_directory = data_directory / config['test dir']
        image_generator = utils.get_test_images(test_data_directory)

        folder_to_save = LOG_FOLDER / 'test_predictions'

        utils.verify_exists_else_create(folder_to_save)

        save_predictions(image_generator, model, config['threshold'], folder_to_save)


if __name__ == '__main__':
    predict()
