"""
Utility function for plotting training history from CSV files.
"""

from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List

from ml_events_utils import stylesheet_default
# from ml_events_utils import color_gio as color_dict
from ml_events_utils import color_deep as color_dict

plt.style.use(stylesheet_default)

def move_offset_factor(ax, ylabel):
    # Move the y-axis offset text (the "x 1e-3" part) into the y-axis label and hide the original offset text to avoid overlap with the title.
    ax.figure.draw_without_rendering()
    offset = ax.yaxis.get_major_formatter().get_offset()
    offset = r" / $" + offset[7:] if offset else ""
    ax.yaxis.set_label_text(ylabel + offset)
    ax.yaxis.offsetText.set_visible(False)


def plot_training_history(model_dir: Path,
                          model_name: str,
                          use_log_scale:bool=True,
                          plot_file: Path = None,
                          plot_settings: Dict[str, Any] = {},
                         ) -> tuple:
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
    model_dir = Path(model_dir)

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

    with plt.style.context(stylesheet_default):
        # Generate plot
        print(f"Generating training history plot...")
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = ax1.twinx()
        axs = [ax1, ax2]

        # Primary y-axis for loss
        axs[0].set_xlabel("epoch")
        axs[0].set_ylabel("loss", color=color_dict['black'])
        axs[0].plot(epochs_train, hist_loss,     label="average training loss",   color=color_dict['blue'])
        axs[0].plot(epochs_val,   hist_val_loss, label="average validation loss", color=color_dict['green'])
        axs[0].tick_params(axis='y', labelcolor=color_dict['black'])
        # ax1.set_ylim(ymin=0)
        axs[0].grid()
        axs[0].legend(loc='upper left')
        if use_log_scale:
            # Scale y axis logarithmically
            axs[0].set_yscale('log')

            # Adjust tick s for y-axis. Not that if also the ticks for the x-axis want to be adjusted, a new loglocator needs to be used.
            locmajy1 = LogLocator(base=10, numticks=100)
            locminy1 = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=1000) # subs=(0.2,0.4,0.6,0.8)

            axs[0].yaxis.set_major_locator(locmajy1)
            axs[0].yaxis.set_minor_locator(locminy1)
            # axs[0].set_yscale('log')

        # axs[0].set_xscale('log')

        # Secondary y-axis for learning rate
        axs[1].set_ylabel("learning Rate", color=color_dict['red'])
        axs[1].plot(epochs_lr, hist_lr, label="learning rate", color=color_dict['red'])
        axs[1].tick_params(axis='y', labelcolor=color_dict['red'])
        axs[1].legend(loc='upper right')

        for ax in axs:
            move_offset_factor(ax, ax.get_ylabel())
        # if use_log_scale:
        #     axs[1].set_yscale('log')


        plt.title(plot_settings.get("title", f"Training History for {model_name}"))
        plt.tight_layout()

        yranges = plot_settings.get("yranges", [])
        for ax, yrange in zip(axs, yranges):
            if yrange is not None:
                if isinstance(yrange, (list, tuple)) and len(yrange) == 2:
                    if yrange[0] is not None and yrange[1] is not None:
                        ax.set_ylim(yrange)
                    elif yrange[0] is not None:
                        ax.set_ylim(bottom=yrange[0])
                    elif yrange[1] is not None:
                        ax.set_ylim(top=yrange[1])
                elif isinstance(yrange, (list, tuple)) and len(yrange) == 1:
                    ax.set_ylim(bottom=yrange[0])

        xrange = plot_settings.get("xrange", [])
        if xrange is not None:
            if isinstance(xrange, (list, tuple)) and len(xrange) == 2:
                if xrange[0] is not None and xrange[1] is not None:
                    ax.set_xlim(xrange)
                elif xrange[0] is not None:
                    ax.set_xlim(left=xrange[0])
                elif xrange[1] is not None:
                    ax.set_xlim(right=xrange[1])
            elif isinstance(xrange, (list, tuple)) and len(xrange) == 1:
                ax.set_xlim(left=xrange[0])

        # Save plot
        if plot_file is None:
            plot_file = model_dir / f"{model_name}_training_history.pdf"
        else:
            plot_file = Path(plot_file)

        plt.savefig(plot_file, bbox_inches='tight')
        print(f"Plot saved as: {plot_file}")
        plt.close()

    return hist_loss, hist_val_loss, hist_lr
