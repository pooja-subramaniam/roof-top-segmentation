import time
import copy
from typing import Any, List
from pathlib import Path

import torch
import utils

from metrics import dice_coefficient


def train(model: Any, criterion: Any, dataloaders: Any,
          optimizer: Any, log_folder: Path, num_epochs: int,
          metrics: List, threshold: float) -> Any:
    """Training module with pre-trained weights
    Args:
        model: pretrained model that needs fine-tuning
        criterion: loss function to be used
        dataloaders: train and val dataloader objects for training
        optimizer: torch optimizer object
        log_folder: folder to log training performance
        num_epochs: total number of epochs to train
        metrics: list of metrics to be computed and reported along with loss
        threshold: threshold used for calculating metrics
    Returns:
        trained model with best weights based on val performance

    """

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Use gpu if available
    device = utils.get_device()
    model.to(device)

    # Initialize fieldnames and epoch summary
    fieldnames, epochsummary = utils.prep_training_log(log_folder, metrics)

    start = time.time()
    print(f"Starting training with {criterion.__class__.__name__} "
          f"and {optimizer.__class__.__name__} "
          f"optimizer with learning rate {optimizer.param_groups[0]['lr']}")
    for epoch in range(1, num_epochs + 1):

        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        batch_loss = {'train': [], 'val': []}
        batch_metrics = {'train': {}, 'val': {}}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in iter(dataloaders[phase]):

                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    y_true = labels.detach().data.cpu().numpy()
                    y_pred = outputs['out'].detach().data.cpu()  # logits
                    y_pred_prob = torch.sigmoid(y_pred).numpy()  # probabilities

                batch_loss[phase].append(loss.item())
                batch_metrics[phase] = utils.get_metrics(metrics, y_true, y_pred_prob, threshold)

        epochsummary = utils.update_epochsummary(epochsummary, epoch, batch_loss, batch_metrics)
        print(epochsummary)
        utils.save_train_log(log_folder, fieldnames, epochsummary)

        # update best_loss and best_model_wts
        if epochsummary['val_loss'] < best_loss:
            best_loss = epochsummary['val_loss']
            best_model_wts = copy.deepcopy(model.state_dict())
            utils.save_model(best_model_wts, log_folder)

    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
