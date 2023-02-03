from yaml import safe_load
import os
from pathlib import Path

from data import DidaSegmentationDataset, get_dataloader
from utils import save_dict

CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'

def main():

    with open(CONFIG_FILENAME) as config_file:
        config = safe_load(config_file) 
    save_dict(config, LOG_FOLDER / 'config.json')

    data_directory = TOP_DIRECTORY / config['data dir']

    # Create the dataloader
    dataloaders = get_dataloader(data_directory, config['images dir'], config['labels dir'],
                                 seed=config['seed'], batch_size=config['batch size'])


if __name__ == '__main__':
    main()