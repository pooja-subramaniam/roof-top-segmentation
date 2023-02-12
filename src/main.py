import os
from pathlib import Path
from yaml import safe_load

import torch

from data import get_dataloader
from models import create_deeplabv3, train
from utils import save_dict, verify_exists_else_create, set_seed, get_loss_function
from predict import predict


CONFIG_FILENAME = 'config.yaml'
TOP_DIRECTORY = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FOLDER = TOP_DIRECTORY / 'models'


def main():
    """Pipeline including dataloading and training
    """

    with open(CONFIG_FILENAME, encoding='utf-8') as config_file:
        config = safe_load(config_file)
    log_folder = LOG_FOLDER / f"experiment_{config['experiment_number']}"
    verify_exists_else_create(log_folder)

    save_dict(config, log_folder / 'config.json')

    data_directory = TOP_DIRECTORY / config['data dir']

    # reproducibility
    set_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataloaders = get_dataloader(data_directory, config['images dir'], config['labels dir'],
                                 seed=config['seed'], batch_size=config['batch size'])

    model = create_deeplabv3()

    # Specify the loss function
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = get_loss_function(config['loss'], config['weight'])

    # Specify the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    model = train(model,
                criterion,
                dataloaders,
                optimizer,
                log_folder=log_folder,
                num_epochs=config['num epochs'],
                metrics=config['metrics'],
                threshold=config['threshold'])

    predict(model)


if __name__ == '__main__':
    main()
