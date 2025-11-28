import time
import torch
import torch.nn as nn
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer, device, print_freq=100):
    size        = len(dataloader.dataset)  # Total number of samples in the dataset.
    num_batches = len(dataloader)          # Number of batches in the dataloader.

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
            loss = loss_fn(pred, y)
        except RuntimeError as e:
            print(f"RuntimeError during loss computation: {e}")
            print(f"pred shape: {pred.shape}, y shape: {y.shape}")
            print(f"pred: {pred}")
            print(f"y: {y}")
            raise e

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
            valid_loss += loss_fn(pred, y).item()
            l1loss += nn.L1Loss()(pred, y).item()

    valid_loss /= num_batches
    l1loss /= num_batches

    print(f"Validation Error: \n Avg (per batch) valid loss: {valid_loss:>8f}, Avg L1 Loss: {l1loss:>8f}\n")

    return valid_loss