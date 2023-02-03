from yaml import safe_load
import os

from data import DidaSegmentationDataset

CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():

    with open(CONFIG_FILENAME) as config_file:
        config = safe_load(config_file)


if __name__ == '__main__':
    main()