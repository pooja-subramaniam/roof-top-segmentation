import numpy as np


def dice_coefficient(y_true: np.array, y_pred: np.array) -> float:
    """Calculate dice coefficient 
    Args:
        y_true: true labels
        y_pred: predictions
    Returns:
        Dice coefficient
    """
    smooth = 1.    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) +
                    np.sum(y_pred_f) + smooth)
