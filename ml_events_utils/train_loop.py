import time
import torch
import torch.nn as nn
import numpy as np

def train_loop(epoch: int, dataloader, model, loss_fn, optimizer, device, print_freq=100, penalties: dict = {}, *args, **kwargs):
    """
    Executes the training loop for a given model, dataloader, loss function, and optimizer.

    This function iterates over the dataloader, computes the loss, applies optional penalty terms,
    performs backpropagation, and updates the model parameters using the optimizer.

    :param epoch: int
        The current epoch number (used for logging and penalty purposes).
    :param dataloader: torch.utils.data.DataLoader
        The dataloader providing batches of input data and corresponding labels.
    :param model: torch.nn.Module
        The model to be trained.
    :param loss_fn: callable
        The loss function used to compute the training loss.
    :param optimizer: torch.optim.Optimizer
        The optimizer used to update the model parameters.
    :param device: torch.device
        The device (CPU or GPU) where the model and data will be moved for computation.
    :param print_freq: int, optional (default=100)
        Frequency (in batches) at which the training loss is printed to the console.
    :param penalties: dict, optional (default={})
        A dictionary specifying penalty terms to be applied to the loss.
        Currently supported penalties:
        - "cross_section": Adds a penalty to enforce closeness between the mean of predictions
            and the mean of true labels. The penalty is computed as:

            .. math::

                \text{penalty} = \text{importance} \cdot \max\left(0,
                \frac{|\sigma_{\text{learned}} - \sigma_{\text{true}}|}{\max(|\sigma_{\text{true}}|, \epsilon)} - \text{threshold}\right)

            where:
            - :math:`\sigma_{\text{true}}` is the mean of the true labels.
            - :math:`\sigma_{\text{learned}}` is the mean of the predictions.
            - :math:`\text{threshold}` is the allowed deviation (default: 0.01).
            - :math:`\text{importance}` is the weight of the penalty term (default: 1.0).
            - :math:`\epsilon` is a small constant to avoid division by zero (default: 1e-9).

    :param args: tuple
        Additional positional arguments (not used in this implementation).
    :param kwargs: dict
        Additional keyword arguments (not used in this implementation).

    :return: float
        The average training loss over all batches.
    """

    size        = len(dataloader.dataset)  # Total number of samples in the dataset.
    num_batches = len(dataloader)          # Number of batches in the dataloader.

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if batch == 0:
            batch_size = X.shape[0]
        # Compute prediction and loss
        pred = model(X)
        try:
            # Note: y[:,0].unsqueeze(-1) is used to bring y to shape (batch_size, 1) to match pred shape for loss computation.
            #       This can be necessary if the labels also contain additional information in other columns that are not used for the loss computation.
            #       For example: y[:,0] = r_LL + LL/UU and y[:,1] = UU, but only r_LL is used for the loss.
            loss = loss_fn(pred, y[:,0].unsqueeze(-1))
        except RuntimeError as e:
            print(f"RuntimeError during loss computation: {e}")
            print(f"pred shape: {pred.shape}, y shape: {y.shape}")
            print(f"pred: {pred}")
            print(f"y: {y}")
            raise e

        # Compute possible penalty terms
        penalty = torch.zeros_like(loss)

        if penalties.get("cross_section", False):
            sigma_true = y[:,0].mean()  # Average over all true labels in the training set
            sigma_learned = pred.mean()  # Average over the predicted values
            threshold = 0.01  # Threshold for closeness (in %)
            importance = 1.0  # Weight of the penalty term in the total loss
            epsilon = 1e-9  # Small constant to avoid division by zero
            xsec_penalty = torch.abs(sigma_learned - sigma_true) / torch.clamp(torch.abs(sigma_true), min=epsilon) - threshold
            xsec_penalty = importance * torch.clamp(xsec_penalty, min=0)
            penalty += xsec_penalty

        # Compute the total loss
        loss += penalty

        train_loss += loss.item()

        # Zero gradients before backpropagation
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if (batch + 1) % print_freq == 0 and batch > 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss / num_batches


def valid_loop(dataloader, model, loss_fn, device):
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)  # Total number of samples in the dataset.
    num_batches = len(dataloader)   # Number of batches in the dataloader.
    valid_loss, l1loss = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y[:,0].unsqueeze(-1)).item()  # For y[:,0].unsqueeze(-1) see comment in train_loop regarding the shape of y and pred for loss computation.
            l1loss += nn.L1Loss()(pred, y[:,0].unsqueeze(-1)).item()

    valid_loss /= num_batches
    l1loss /= num_batches

    print(f"Validation Error: \n Avg (per batch) valid loss: {valid_loss:>8f}, Avg L1 Loss: {l1loss:>8f}\n")

    return valid_loss