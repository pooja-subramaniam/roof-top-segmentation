import os
from pathlib import Path
from yaml import safe_load

import torch

from data import get_dataloader
from models import create_deeplabv3, train
from utils import save_dict, verify_exists_else_create, set_seed


CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'


def main():
    """Pipeline including dataloading and training
    """

    with open(CONFIG_FILENAME, encoding='utf-8') as config_file:
        config = safe_load(config_file)

    verify_exists_else_create(LOG_FOLDER)

    save_dict(config, LOG_FOLDER / 'config.json')

    data_directory = TOP_DIRECTORY / config['data dir']

    # reproducibility
    set_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataloaders = get_dataloader(data_directory, config['images dir'], config['labels dir'],
                                 seed=config['seed'], batch_size=config['batch size'])

    model = create_deeplabv3()

    # Specify the loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Specify the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    _ = train(model,
                criterion,
                dataloaders,
                optimizer,
                log_folder=LOG_FOLDER,
                num_epochs=config['num epochs'])



if __name__ == '__main__':
    main()
