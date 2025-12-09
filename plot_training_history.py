"""
Utility function for plotting training history from CSV files.
"""

from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(model_dir, model_name, use_log_scale=True):
    """
    Generate and save a training history plot from CSV files.

    Parameters:
    -----------
    model_dir : Path
        Directory containing the CSV files
    model_name : str
        Name of the model (used for file naming)
    use_log_scale : bool
        Whether to use logarithmic scale for y-axes (default: True)

    Returns:
    --------
    tuple : (hist_loss, hist_val_loss, hist_lr)
        Numpy arrays containing the training history
    """

    # Define file paths
    train_loss_file = model_dir / f"{model_name}_train_loss.csv"
    val_loss_file   = model_dir / f"{model_name}_val_loss.csv"
    lr_file         = model_dir / f"{model_name}_learning_rates.csv"

    # Check if files exist
    if not all([train_loss_file.exists(), val_loss_file.exists(), lr_file.exists()]):
        raise FileNotFoundError(
            f"One or more CSV files not found in {model_dir}\n"
            f"  Train loss file: {train_loss_file.exists()}\n"
            f"  Val loss file: {val_loss_file.exists()}\n"
            f"  LR file: {lr_file.exists()}"
        )

    # Load data from CSV files
    print(f"Loading training history from {model_dir}...")
    train_data = pd.read_csv(train_loss_file)
    val_data   = pd.read_csv(val_loss_file)
    lr_data    = pd.read_csv(lr_file)

    epochs_train  = train_data['epoch'].values
    hist_loss     = train_data['train_loss'].values
    hist_val_loss = val_data['val_loss'].values
    epochs_val    = val_data['epoch'].values
    hist_lr       = lr_data['learning_rate'].values
    epochs_lr     = lr_data['epoch'].values

    print(f"Loaded {len(hist_loss)} epochs of training data")

    # Generate plot
    print(f"Generating training history plot...")
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Primary y-axis for loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='black')
    ax1.plot(epochs_train, hist_loss,     label="Avg training loss", color='blue')
    ax1.plot(epochs_val,   hist_val_loss, label="Avg validation loss", color='orange')
    ax1.tick_params(axis='y', labelcolor='black')
    # ax1.set_ylim(ymin=0)
    ax1.grid()
    ax1.legend(loc='upper left')
    if use_log_scale:
        # Scale y axis logarithmically
        ax1.set_yscale('log')

        # Adjust tick s for y-axis. Not that if also the ticks for the x-axis want to be adjusted, a new loglocator needs to be used.
        locmajy1 = LogLocator(base=10, numticks=100)
        locminy1 = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=1000) # subs=(0.2,0.4,0.6,0.8)

        ax1.yaxis.set_major_locator(locmajy1)
        ax1.yaxis.set_minor_locator(locminy1)
        # ax1.set_yscale('log')

    # Secondary y-axis for learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color='red')
    ax2.plot(epochs_lr, hist_lr, label="Learning rate", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    # if use_log_scale:
    #     ax2.set_yscale('log')

    plt.title(f"Training History for {model_name}")
    plt.tight_layout()

    # Save plot
    plot_file = model_dir / f"{model_name}_training_history.pdf"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Plot saved as: {plot_file}")
    plt.close()

    return hist_loss, hist_val_loss, hist_lr
