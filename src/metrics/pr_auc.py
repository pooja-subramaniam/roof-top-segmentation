from sklearn.metrics import precision_recall_curve, auc
import numpy as np


def precision_recall_auc(y_true: np.array, y_pred: np.array) -> float:
    """Calculate precision recall auc
    Args:
        y_true: true labels
        y_pred: predictions
    Returns:
        Precision recall area under the curve (PR-AUC)
    """
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    return auc(recall, precision)
