from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_curves(log_folder: Path, metric_names: List) -> None:
    """Create and save metric curves plot in log_folder
    Args:
        log_folder: folder to save the plot
    """
    logs = pd.read_csv(log_folder / 'log.csv')
    epochs = range(1, len(logs)+1)
    for i, metric_name in enumerate(metric_names):
        max_metric = logs[metric_name].max()
        epoch_max_metric = logs[metric_name].argmax() + 1
        plt.plot(epochs, logs[metric_name], label=metric_name)
        if i != len(metric_names)-1:
            plt.scatter(epoch_max_metric, max_metric, c='g')
        else:
            plt.scatter(epoch_max_metric, max_metric, c='g', label='maximum')
        plt.title('Metric curves of train and validation sets during training')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()

    plt.savefig(log_folder / 'metric_curves.png', dpi=200)
    plt.close()
