import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging

from .transforms import exp_target_transform
from .analysis import costhetastar

logger = logging.getLogger(__name__)

def train_loop(
    epoch: int,
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    print_freq=100,
    penalties: dict = {},
    eps: float = 1e-12,
    target_col: int = 0,
    weight_col: int = 1,
    *args,
    **kwargs,
):
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
    :param eps: float, optional (default=1e-12)
        A small constant used to prevent division by zero in penalty calculations.
    :param target_col: int, optional (default=0)
        The column index in the labels (y) that contains the target values used for loss computation (for example the LL/ UU cross section ratio).
    :param weight_col: int, optional (default=1)
        The column index in the labels (y) that contains the denominator of the target_col (for example the UU cross section) used for weighted loss computation.
        If the specified column does not exist, the loss will be computed as an unweighted mean over the batch.
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
        target = y[:, target_col].unsqueeze(-1)
        try:
            # Note: y[:,target_col].unsqueeze(-1) is used to bring y to shape (batch_size, 1) to match pred shape for loss computation.
            #       This can be necessary if the labels also contain additional information in other columns that are not used for the loss computation.
            #       For example: y[:,0] = r_LL = LL/UU and y[:,1] = UU, but only r_LL is used for the loss.

            # loss = loss_fn(pred, target)

            # Choose event weights (usually UU). If the column doesn't exist, fall back to unweighted mean.
            w = None
            if (y.ndim >= 2) and (y.shape[1] > weight_col):
                w = torch.abs(y[:, weight_col]).detach()  # shape (B,)

            # compute per-sample loss (example for MSE; adapt similarly for L1/SmoothL1/etc.)
            if isinstance(loss_fn, nn.MSELoss):
                per_sample = F.mse_loss(pred, target, reduction="none")
            elif isinstance(loss_fn, nn.L1Loss):
                per_sample = F.l1_loss(pred, target, reduction="none")
            else:
                # fallback: require user to pass a loss_fn that already returns per-sample loss
                per_sample = loss_fn(pred, target)

            # Reduce loss to one scalar per event: (B, 1) -> (B,)
            if torch.is_tensor(per_sample) and per_sample.ndim > 1:
                per_sample = per_sample.view(per_sample.size(0), -1).mean(dim=1)
        except RuntimeError as e:
            logger.error(f"RuntimeError during loss computation: {e}")
            logger.error(f"pred shape: {pred.shape}, y shape: {y.shape}")
            logger.error(f"pred: {pred}")
            logger.error(f"y: {y}")
            raise e

        # Compute possible penalty terms
        penalty_scalar = 0.0

        if penalties.get("ZdecayAngles", False):
            threshold  = 0.01  # Threshold for closeness (in %)
            importance = 0.001 * (1 + (epoch // 10)**2)  # Weight of the penalty term in the total loss
            # The functional form of the normalised cthep distribution is
            #     (3/4)*sin(theta)^2 = (3/4)*(1-cthep^2).
            if y.ndim < 2 or y.shape[1] <= 1:
                raise ValueError("ZdecayAngles penalty expects y[:,1] to contain UU weights.")

            xsec_LL = pred[:,0] * y[:,1]
            # xsec_LL = y[:,0] * y[:,event_weight_col]
            xsec_LL_norm = xsec_LL / xsec_LL.sum()


            momenta = X.reshape(X.shape[0], -1, 4)
            # cthep, _, cthmup, _ = torch.stack(costhetastar(momenta), dim=-1)
            cthep, _, cthmup, _ = costhetastar(momenta)

            expected_cthep = (3/4) * (1 - cthep**2)
            expected_cthep /= expected_cthep.sum()  # Normalize the expected distribution

            expected_cthmup = (3/4) * (1 - cthmup**2)
            expected_cthmup /= expected_cthmup.sum()
            # Note that event though the normalization over only the batch size is going to be bad,
            # there is no way around that, since (3/4) * (1 - cthep**2) is only valid for the normalized distribution.

            # Elementwise penalty per event (shape (B,))
            diff_cthep  = torch.abs(xsec_LL_norm) / torch.clamp(torch.abs(expected_cthep),  min=eps)
            diff_cthmup = torch.abs(xsec_LL_norm) / torch.clamp(torch.abs(expected_cthmup), min=eps)

            angle_penalty = importance * (torch.clamp(diff_cthep - threshold, min=0) + torch.clamp(diff_cthmup - threshold, min=0))
            per_sample = per_sample + angle_penalty

            # # Plot the stuff as a sanity check.
            # bins = np.linspace(-1, +1, 50 + 1)
            # bin_widths = bins[1:] - bins[:-1]
            # bin_midths = (bins[:-1] + bins[1:]) / 2

            # # Sum predicted labels in each invariant mass bin
            # pred_sums_cthep, _ = np.histogram(cthep.cpu().detach().numpy(), bins=bins, weights=xsec_LL_norm.cpu().detach().numpy())
            # # Sum expected values in each invariant mass bin
            # expected_sums_cthep, _ = np.histogram(cthep.cpu().detach().numpy(), bins=bins, weights=expected_cthep.cpu().detach().numpy())

            # pred_sums_cthmup, _ = np.histogram(cthmup.cpu().detach().numpy(), bins=bins, weights=xsec_LL_norm.cpu().detach().numpy())
            # expected_sums_cthmup, _ = np.histogram(cthmup.cpu().detach().numpy(), bins=bins, weights=expected_cthmup.cpu().detach().numpy())

            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 12))

            # # Top plot: Expected vs Predicted distributions
            # plt.subplot(2, 1, 1)
            # plt.plot(bin_midths, expected_sums_cthep, label="Expected", linestyle="--", color="blue")
            # plt.plot(bin_midths, pred_sums_cthep, label="Predicted", linestyle="-", color="orange")
            # plt.plot(bin_midths, expected_sums_cthmup, label="Expected cthmup", linestyle="--", color="green")
            # plt.plot(bin_midths, pred_sums_cthmup, label="Predicted cthmup", linestyle="-", color="red")
            # plt.xlabel("cos(theta)")
            # plt.ylabel("Normalized Distribution")
            # plt.title("Comparison of Expected and Predicted cos(theta) Distribution")
            # plt.legend()
            # plt.grid(True)

            # # Bottom plot: Ratios of Predicted to Expected
            # plt.subplot(2, 1, 2)
            # ratio_cthep  = pred_sums_cthep  / np.clip(np.abs(expected_sums_cthep), a_min=1e-9, a_max=None)
            # ratio_cthmup = pred_sums_cthmup / np.clip(np.abs(expected_sums_cthmup), a_min=1e-9, a_max=None)
            # plt.plot(bin_midths, ratio_cthep, label="Ratio cthep", linestyle="-", color="orange")
            # plt.plot(bin_midths, ratio_cthmup, label="Ratio cthmup", linestyle="-", color="red")
            # plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal Ratio")
            # plt.xlabel("cos(theta)")
            # plt.ylabel("Ratio (Predicted / Expected)")
            # plt.title("Ratio of Predicted to Expected cos(theta) Distribution")
            # plt.legend()
            # plt.grid(True)

            # plt.tight_layout()
            # plt.savefig(f"cthep_distribution_epoch_{epoch}.pdf")

        if penalties.get("cross_section", False):
            sigma_true    = torch.mean(exp_target_transform(   y[:,0]) * exp_target_transform(y[:,1]))  # Average over all true labels in the training set.
            sigma_learned = torch.mean(exp_target_transform(pred[:,0]) * exp_target_transform(y[:,1]))  # Average over the predicted values.
            threshold  = 0.005  # Threshold for closeness (in %)
            importance = 0.001  # Weight of the penalty term in the total loss
            importance *= (1 + (epoch // 12)**2)  # Optionally increase the importance of the penalty term as training progresses.
            xsec_penalty = torch.abs(sigma_learned - sigma_true) / torch.clamp(torch.abs(sigma_true), min=eps) - threshold
            xsec_penalty = importance * torch.clamp(xsec_penalty, min=0)
            penalty_scalar += xsec_penalty

        # Weighted mean (if weights available), else plain mean.
        if w is None:
            loss = per_sample.mean()
        else:
            loss = (per_sample * w).sum() / torch.clamp(w.sum(), min=eps)

        # Compute the total loss
        loss += penalty_scalar

        train_loss += loss.item()

        # Zero gradients before backpropagation
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if (batch + 1) % print_freq == 0 and batch > 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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

    logger.info(f"Validation Error:")
    logger.info(f"  Avg (per batch) valid loss: {valid_loss:>8f}, Avg L1 Loss: {l1loss:>8f}")

    return valid_loss
