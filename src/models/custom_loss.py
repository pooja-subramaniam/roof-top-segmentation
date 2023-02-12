from torch import nn
import torch


class DiceLoss(nn.Module):
    """Custom loss function emulating dice coefficient.
    """
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.FloatTensor:
        """Actual dice loss implementation to be run during training. Combines sigmoid layer on
        the final layer output.
        Args:
            inputs: logits from the model final layer
            targets: ground truth labels
        Returns:
            diceloss
        """

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedDiceBCEWithLogitsLoss(nn.Module):
    """Combine BCE and Dice loss.
    """
    def __init__(self, weight=0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.FloatTensor:
        """Add dice loss and bce loss.
        Args:
            inputs: logits from the model final layer
            targets: ground truth labels
        Returns:
            combined dice and bce loss
        """

        return self.bce_loss(inputs, targets) + self.weight * self.dice_loss(inputs,targets)
