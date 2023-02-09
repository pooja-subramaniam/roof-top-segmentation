from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(log_folder: Path, loss_name: str) -> None:
    """Create and save loss curve in log_folder
    Args:
        log_folder: folder to save the plot
    """
    logs = pd.read_csv(log_folder / 'log.csv')
    epochs = range(1, len(logs.train_loss)+1)

    min_train_loss = logs.train_loss.min()
    epoch_min_train_loss = logs.train_loss.argmin() + 1
    plt.plot(epochs, logs.train_loss, label='train')
    plt.scatter(epoch_min_train_loss, min_train_loss, c='r')

    min_val_loss = logs.val_loss.min()
    epoch_min_val_loss = logs.val_loss.argmin() + 1
    plt.plot(epochs, logs.val_loss, label='val')
    plt.scatter(epoch_min_val_loss, min_val_loss, c='r', label='minimum')

    plt.title('Loss curves of train and validation sets during training')
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_name}')
    plt.legend()
    plt.savefig(log_folder / 'loss_curves.png', dpi=200)
    plt.close()
