from yaml import safe_load
import os
from pathlib import Path

from data import DidaSegmentationDataset
from utils import save_dict

CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'

def main():

    with open(CONFIG_FILENAME) as config_file:
        config = safe_load(config_file) 
    save_dict(config, LOG_FOLDER / 'config.json')


if __name__ == '__main__':
    main()