from datetime import datetime
import time
import torch
import argparse
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.ticker as mticker


from pathlib import Path
from torchinfo import summary
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

from ml_events_utils.write_top_file import TopFileWriter
from ml_events_utils.transforms import januar2026_input_choice
from ml_events_utils import MLEventsDataset, scale_target, boost_into_four_lepton_cm_frame, log_target_transform, exp_target_transform  #, test_loop
from ml_events_utils import ZJetDataset
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
from ml_events_utils.analysis import costhetastar, get_pt, get_rapidity, cosmujet, getdphi
from ml_events_utils import log_file, setup_file_logger
from ml_events_utils import Settings
from ml_events_utils import stylesheet_default
from ml_events_utils import color_deep as color_dict

color_dict = {key: hexwithhash for key, (hex, hexwithhash, floats) in color_dict.items()}

# Apply the package default style globally so all plots in this module are consistent.
plt.style.use(stylesheet_default)

# %% Helper functions for plotting and histogram handling
def read_top_file_histograms(top_file_paths: dict) -> dict:
    """
    Read histogram data from a .top file.

    Args:
        top_file_paths (dict): Dictionary with keys as histogram names and values as paths to .top files

    Returns:
        dict: Dictionary with histogram names as keys, each containing:
            - 'bin_left': numpy array of left bin edges
            - 'bin_right': numpy array of right bin edges
            - 'values': numpy array of histogram values
            - 'uncertainties': numpy array of uncertainties
    """
    histogram_data = {}
    observables    = {}
    for run, top_file_path in top_file_paths.items():
        histogram = {}
        observables[run] = []
        with open(top_file_path, 'r') as f:
            current_histogram = None

            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check if this is a histogram header
                if line.startswith('#') and 'index' in line:
                    # Extract histogram name (everything before 'index')
                    hist_name = line.split('index')[0].strip('# ').strip()
                    current_histogram = hist_name
                    observables[run].append(current_histogram)
                    histogram[current_histogram] = {
                        'bin_left': [],
                        'bin_right': [],
                        'values': [],
                        'uncertainties': [],
                        'edges': []
                    }
                    continue

                # Skip other comment lines
                if line.startswith('#'):
                    continue

                # Parse data lines
                if current_histogram is not None:
                    try:
                        # Split by whitespace and convert scientific notation
                        parts = line.split()
                        if len(parts) >= 4:
                            bin_left = float(parts[0].replace('D', 'E'))
                            bin_right = float(parts[1].replace('D', 'E'))
                            value = float(parts[2].replace('D', 'E'))
                            uncertainty = float(parts[3].replace('D', 'E'))

                            histogram[current_histogram]['bin_left'].append(bin_left)
                            histogram[current_histogram]['bin_right'].append(bin_right)
                            histogram[current_histogram]['values'].append(value)
                            histogram[current_histogram]['uncertainties'].append(uncertainty)
                    except ValueError:
                        # Skip lines that can't be parsed as numbers
                        continue

        # Convert lists to numpy arrays for easier manipulation
        for hist_name in histogram:
            for key in histogram[hist_name]:
                histogram[hist_name][key] = np.array(histogram[hist_name][key])
                histogram[hist_name]['edges'] = np.concatenate((
                    histogram[hist_name]['bin_left'],
                    histogram[hist_name]['bin_right'][-1:]
                ))

        histogram_data[run] = histogram

    observables_intersection = set.intersection(*[set(obs) for obs in observables.values()])
    # So far the structure of histogram_data is {run: {histogram_name: {bin_left, bin_right, values, uncertainties, edges}}}.
    # Change it to be {histogram_name: {run: {bin_left, bin_right, values, uncertainties, edges}}} for easier access in the plotting functions.
    histogram_data_restructured = {observable: {run : histogram_data[run][observable] for run in top_file_paths.keys()} for observable in observables_intersection}

    return histogram_data_restructured

def print_integration_statistics(observable_dict, histogram_data: dict = {}, model: str = "", powheg_histogram_runs: list = ["LL",], fitted_polarisation: str = "LL", hist_writer: TopFileWriter = None):
    pred_integral = np.sum(observable_dict["weights_ypred"][0])
    true_integral = np.sum(observable_dict["weights_y"][0])

    # Note that we can in principle also integrate over the histograms here for cross-checks:
    # pred_integral = np.sum(pred_sums * bin_widths)
    # true_integral = np.sum(true_sums * bin_widths)

    if model:
        title = f"Integrated cross-sections for model\n{model}"
    else:
        title = "Integrated cross-sections"
    logger.info(f"{title}:")
    logger.info(f"  True r_{fitted_polarisation}:          {true_integral:.6e}")
    logger.info(f"  Predicted r_{fitted_polarisation}:     {pred_integral:.6e}")
    if histogram_data:
        for run in powheg_histogram_runs:
            logger.info(f"  POWHEG reweighting [{run}]: {histogram_data['totxsec'][run]['values'][0]:.6e}")
        logger.info(f"  Ratio (pred/PWG-[{powheg_histogram_runs[0]}]):   {pred_integral/histogram_data['totxsec'][powheg_histogram_runs[0]]['values'][0]:.6f}")
    logger.info(f"  Ratio (pred/true):  {pred_integral/true_integral:.6f}")

    if hist_writer:
        hist_writer.write_histogram(f"{fitted_polarisation}_pred.top", "totxsec", bin_edges=[[0., 1.],], values=[pred_integral,], uncertainties=[0., ])
        hist_writer.write_histogram(f"{fitted_polarisation}_true.top", "totxsec", bin_edges=[[0., 1.],], values=[true_integral,], uncertainties=[0., ])
        for run in powheg_histogram_runs:
            hist_writer.write_histogram(f"POWHEG_{run}.top", "totxsec", bin_edges=[[0., 1.],], values=[histogram_data['totxsec'][run]['values'][0],], uncertainties=[0., ])

    # Create a text-only plot for integration results
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')  # Remove axes

    text_content  = f"    True r_{fitted_polarisation}:               {true_integral:.6e}\n"
    text_content += f"    Predicted r_{fitted_polarisation}:          {pred_integral:.6e}\n"
    if histogram_data:
        for run in powheg_histogram_runs:
            text_content += f"    POWHEG reweighting [{run}]: {histogram_data['totxsec'][run]['values'][0]:.6e}\n"

    text_content += f"Ratio (pred/PWG-[{powheg_histogram_runs[0]}]):   {pred_integral/histogram_data['totxsec'][powheg_histogram_runs[0]]['values'][0]:.6f}\n"
    text_content += f"    Ratio (pred/true):       {pred_integral/true_integral:.6f}"

    ax.text(0.1, 0.5, text_content, fontsize=14, verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    ax.set_title(f'{title}', fontsize=16, fontweight='bold')

    return fig, ax

def plot_r_distribution(r_pred, r_true, model_name="Model", fitted_polarisation:str="LL", hist_writer: TopFileWriter = None):
    fig, axs = plt.subplots(1, 1)
    r_min, r_max = min(r_pred.min() ,r_true.min()), max(r_pred.max(), r_true.max())
    bins = np.linspace(r_min, r_max, 101)

    axs.set_title(f"{model_name}")
    axs.hist(r_pred, bins=bins, alpha=0.7, label=f"Predicted")
    axs.hist(r_true, bins=bins, alpha=0.7, label="True")

    if hist_writer:
        pred_hist, _ = np.histogram(r_pred, bins=bins)
        true_hist, _ = np.histogram(r_true, bins=bins)
        hist_writer.write_histogram(f"{fitted_polarisation}_pred.top", f"r_{fitted_polarisation}_count", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=pred_hist, uncertainties=np.zeros_like(pred_hist))
        hist_writer.write_histogram(f"{fitted_polarisation}_true.top", f"r_{fitted_polarisation}_count", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=true_hist, uncertainties=np.zeros_like(true_hist))

    axs.set_xlabel(r"$r_{\mathrm{" + fitted_polarisation + r"}}$")

    ylabel = "Events"
    # axs.set_ylabel(ylabel)
    # Move the y-axis offset text (the "x 1e-3" part) into the y-axis label and hide the original offset text to avoid overlap with the title.
    axs.figure.draw_without_rendering()
    offset = axs.yaxis.get_major_formatter().get_offset()
    offset = r" / $" + offset[7:] if offset else ""
    axs.yaxis.set_label_text(ylabel + offset)
    axs.yaxis.offsetText.set_visible(False)

    axs.legend()
    return fig, axs

def r_plot(r_pred, r_true, weights, model_name="Model", fitted_polarisation:str="LL", hist_writer: TopFileWriter = None):
    fig, axs = plt.subplots(1, 1)

    r_min, r_max = -0.02, 1.00

    bin_width = 0.01
    bins = np.arange(r_min, r_max + bin_width, bin_width)
    # np.linspace(r_min, r_max, 81)

    # Calculate bin widths for proper integration
    bin_widths = bins[1:] - bins[:-1]

    # Sum predicted labels in each r bin
    # Handle case where weights is a single float
    if np.isscalar(weights):
        weight_array = np.full_like(r_pred, weights)
    else:
        weight_array = weights

    pred_sums, _ = np.histogram(r_pred, bins=bins, weights=weight_array)
    pred_sums /= bin_widths
    true_sums, _ = np.histogram(r_true, bins=bins, weights=weight_array)
    true_sums /= bin_widths

    # Plot as step histograms
    bin_centers = (bins[:-1] + bins[1:]) / 2

    axs.step(bin_centers, true_sums, where='mid', label='True r',      color=color_dict['green'], linewidth=2, alpha=0.7, marker='')
    axs.step(bin_centers, pred_sums, where='mid', label='Predicted r', color=color_dict['red'],   linewidth=2, alpha=0.7, marker='')

    if hist_writer:
        hist_writer.write_histogram(f"{fitted_polarisation}_pred.top", f"r_{fitted_polarisation}", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=pred_sums, uncertainties=np.zeros_like(pred_sums))
        hist_writer.write_histogram(f"{fitted_polarisation}_true.top", f"r_{fitted_polarisation}", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=true_sums, uncertainties=np.zeros_like(true_sums))


    # Scale y axis logarithmically
    # axs[0].set_yscale('log')

    axs.set_ylabel(r"$\frac{\mathrm{d} \sigma^{\mathrm{UU}}}{\mathrm{d} r_{\mathrm{" + fitted_polarisation + r"}}}$ [pb]")
    axs.set_title(f"{model_name}")
    axs.legend()
    axs.grid(True, alpha=0.3)

    try:
        r_min, r_max = min(r_pred.min() ,r_true.min()), max(r_pred.max(), r_true.max())
        if abs(r_min) != float("inf") and abs(r_max) != float("inf"):
            axs.set_xlim(xmin=r_min * 0.99, xmax=r_max * 1.01)
    except ValueError as e:
        logger.error(f"Could not set x limits for r plot: {e}")

    axs.set_xlabel(r"$r_{\mathrm{" + fitted_polarisation + r"}}$")

    fig.tight_layout()
    powheg_label = "rLL"
    return fig, axs, powheg_label

def comparison_plots(observable_dict:dict, observable_key:str, powheg_histogram:dict = None, log_scale=True, nbins=50, model_name="Model", powheg_histogram_runs: list = ["LL",], hist_writer: TopFileWriter = None, *args, **kwargs):
    """
    Create a comparison plot of predicted vs true labels for a given observable.
    This function generates a step histogram plot comparing predicted labels, true labels,
    and (if given) POWHEG reference data for a specified observable. The histograms are normalized
    by bin width and displayed on a logarithmic y-scale.
    Note the slight difference of the true labels and POWHEG histograms due to the smaller statistics
    of the true labels (which are only from the test set) compared to the POWHEG histograms (which are from the full dataset).

    Args:
        observable_dict (dict): Dictionary containing observable data with keys:
            - observable_key: The observable values to plot
            - "weights_ypred": Predicted label weights
            - "weights_y": True label weights
            - "invmass_Z1": Invariant mass values (used for x-axis limits)
        observable_key (str): Key specifying which observable to plot from observable_dict
        powheg_histogram (dict, optional): Optional dictionary used as an additional comparison
            containing POWHEG reference data with keys:
            - 'edges': Bin edges for the histogram
            - 'values': Histogram values for comparison
        log_scale (bool, optional): Whether to use logarithmic scaling for the y-axis. Default is True.
        nbins (int, optional): Number of bins to use if powheg_histogram is not provided. Default is 50.
        plotrLL (bool, optional): Whether to plot for directly r_LL distributions instead of the reweighted cross section. Default is False.
    Returns:
        tuple: Figure and axes objects (fig, axs) for the created plot
    Note:
        The function creates step histograms normalized by bin width, plots them with
        different colors (green for true, red for predicted, blue for POWHEG), and
        applies logarithmic scaling to the y-axis. The plot includes a legend, grid,
        and appropriate labels.
    """

    # Create a single comparison plot
    size = mpl.rcParams['figure.figsize']
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(size[0], 1.5*size[1]), height_ratios=[3, 1])

    if powheg_histogram:
        bins = powheg_histogram[powheg_histogram_runs[0]]['edges']
    else:
        bins = np.linspace(observable_dict[observable_key][0].min(), observable_dict[observable_key][0].max(), nbins+1)

    # Calculate bin widths for proper integration
    bin_widths = bins[1:] - bins[:-1]

    # Sum predicted labels in each invariant mass bin
    if not kwargs.get("plotrLL", False):
        pred_sums, _ = np.histogram(observable_dict[observable_key][0], bins=bins, weights=observable_dict["weights_ypred"][0])
        # Sum true labels in each invariant mass bin
        true_sums, _ = np.histogram(observable_dict[observable_key][0], bins=bins, weights=observable_dict["weights_y"][0])
    else:
        pred_sums, _ = np.histogram(observable_dict[observable_key][0], bins=bins, weights=observable_dict["r_pred"][0])
        # Sum true labels in each invariant mass bin
        true_sums, _ = np.histogram(observable_dict[observable_key][0], bins=bins, weights=observable_dict["r_true"][0])
    pred_sums /= bin_widths
    true_sums /= bin_widths

    if powheg_histogram:
        # POWHEG histograms for comparison
        powheg_sums = {}
        for run in powheg_histogram_runs:
            powheg_sums[run] = powheg_histogram[run]['values']

    # Plot as step histograms
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axs[0].step(bin_centers, true_sums,   where='mid', label='True Labels', color=color_dict['green'], linewidth=2, alpha=1.0, marker='')
    axs[0].plot(bin_centers, true_sums, 'x', color=color_dict['green'], markersize=8, alpha=1.0)

    def powheg_color(reset_index=False):
        if reset_index or "counter" not in powheg_color.__dict__:
            powheg_color.__dict__["counter"] = 0
        colors = [color_dict['blue'], color_dict['cyan'], color_dict['yellow']]
        while True:
            i = powheg_color.__dict__["counter"] % len(colors)
            powheg_color.__dict__["counter"] += 1
            yield colors[i]

    a = powheg_color(reset_index=True)

    if powheg_histogram:
        for run, color in zip(powheg_histogram_runs, powheg_color(reset_index=True)):
            axs[0].step(bin_centers, powheg_sums[run], where='mid', label=f'POWHEG Labels {run}',    color=color,  linewidth=2, alpha=1.0, marker='')
    axs[0].step(bin_centers, pred_sums,   where='mid', label='Predicted Labels', color=color_dict['red'],   linewidth=2, alpha=1.0, marker='')

    if hist_writer:
        fitted_polarisation = powheg_histogram_runs[0]
        hist_writer.write_histogram(f"{fitted_polarisation}_pred.top", f"{observable_key}", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=pred_sums, uncertainties=np.zeros_like(pred_sums))
        hist_writer.write_histogram(f"{fitted_polarisation}_true.top", f"{observable_key}", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=true_sums, uncertainties=np.zeros_like(true_sums))
        if powheg_histogram:
            for run in powheg_histogram_runs:
                hist_writer.write_histogram(f"POWHEG_{run}.top", f"{observable_key}", bin_edges=np.column_stack((bins[:-1], bins[1:])), values=powheg_sums[run], uncertainties=np.zeros_like(powheg_sums[run]))

    axs[1].step(bin_centers, pred_sums / np.maximum(true_sums, 1e-10), where='mid', color=color_dict['red'], linewidth=2, alpha=1.0, marker='')
    if powheg_histogram:
        for run, color in zip(powheg_histogram_runs, powheg_color(reset_index=True)):
            axs[1].step(bin_centers, powheg_sums[run] / np.maximum(true_sums, 1e-10), where='mid', color=color, linewidth=2, alpha=1.0, linestyle='--', marker='')

    axs[1].axhline(1.0, color=color_dict['gray'], linestyle='--', linewidth=1)
    axs[1].set_ylabel("X / True")

    x_label_tex = observable_dict[observable_key][1]

    # Scale y axis logarithmically
    if log_scale:
        try:
            if np.all(true_sums > 0) and np.all(pred_sums > 0) and (not powheg_histogram or np.all([np.all(sums > 0) for sums in powheg_sums.values()])):
                axs[0].set_yscale('log')
            else:
                logger.info(f"Not all histogram values are positive for {observable_key} plot; skipping log scale.")
        except ValueError as e:
            logger.error(f"Could not set y scale to log for {observable_key} plot: {e}")

    if not kwargs.get("plotrLL", False):
        ylabel_axs0 = r"$\frac{\mathrm{d} \sigma}{\mathrm{d} \mathrm{" + x_label_tex[1:-1] + r"}}$ [pb]"
    else:
        ylabel_axs0 = r"$r$"
    # axs[0].set_ylabel(ylabel_axs0)

    axs[0].set_title(f"{model_name}")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)

    # Move the y-axis offset text (the "x 1e-3" part) into the y-axis label and hide the original offset text to avoid overlap with the title.
    axs[0].figure.draw_without_rendering()
    offset = axs[0].yaxis.get_major_formatter().get_offset()
    offset = r" / $" + offset[7:] if offset else ""
    axs[0].yaxis.set_label_text(ylabel_axs0 + offset)
    axs[0].yaxis.offsetText.set_visible(False)

    # Move the y-axis offset text (the "x 1e-3" part) down a bit to avoid overlap with the x-axis label.
    # axs[0].yaxis.get_offset_text().set_y(0.5)

    try:
        # Determin percentage of small values to set x limits accordingly
        # Percentage of bins with values smaller than the mean value of the histogram. If this percentage is large, we set the y limits to be smaller to better visualize the distribution. Otherwise, we set the y limits to be larger to include all values.
        ratios = []
        ratios.append(np.sum(true_sums < true_sums.mean()) / len(true_sums))
        ratios.append(np.sum(pred_sums < pred_sums.mean()) / len(pred_sums))
        if powheg_histogram:
            for run in powheg_histogram_runs:
                ratios.append(np.sum(powheg_sums[run] < powheg_sums[run].mean()) / len(powheg_sums[run]))
        small_percent = max(ratios)
        if small_percent > 0.9:
            # Find non-zero bins and set y limits to be between the 1st and 99th percentile of these bins to better visualize the distribution.
            # Find the indices of the bins that are smaller than the mean value of the histogram and set the x limits to be between the minimum and maximum of these bins to better visualize the distribution.

            big_indices = []
            big_indices.append(np.flatnonzero(~(true_sums < true_sums.mean())))
            big_indices.append(np.flatnonzero(~(pred_sums < pred_sums.mean())))
            if powheg_histogram:
                for run in powheg_histogram_runs:
                    big_indices.append(np.flatnonzero(~(powheg_sums[run] < powheg_sums[run].mean())))

            min_index, max_index = min([indices.min() for indices in big_indices if len(indices) > 0]), max([indices.max() for indices in big_indices if len(indices) > 0])

            extra_bins = int(0.05 * len(bins))  # Add 5% of the total number of bins as extra range on both sides
            min_index = max(min_index - extra_bins, 0) if min_index is not None else None
            max_index = min(max_index + extra_bins, len(bins) - 2) if max_index is not None else None
            axs[0].set_xlim(xmin=bins[min_index] if min_index is not None else bins[0], xmax=bins[max_index+1] if max_index is not None else bins[-1])
    except ValueError as e:
        logger.error(f"Could not set x limits for {observable_key} plot: {e}")

    axs[1].set_xlabel(f"{x_label_tex}")

    fig.tight_layout()

    return fig, axs

# %% Testing loop
def test_model_ZZ(model,
                  model_dir,
                  histogram_dir,
                  dataloader,
                  dataloader_untransformed,
                  loss_fn,
                  device,
                  n_generated_events:int=0.2*1e7,
                  model_name="Model",
                  fitted_polarisation="LL",
                  showered:bool=False,
                  *args,
                  **kwargs):
    """
    Test a trained machine learning model and generate comparison plots with POWHEG reference data.
    This function evaluates the model on test data, computes observables (invariant masses and cos(theta*)),
    creates histograms comparing predicted vs. true labels vs. POWHEG results, and saves the plots to a PDF.
    Args:
        model: PyTorch model to be tested
        model_dir: Directory containing the model files
        histogram_dir (Path): Directory containing reference histogram files (.top format)
        dataloader: PyTorch DataLoader containing test data with features (X) and targets (y)
        dataloader_untransformed: PyTorch DataLoader containing untransformed test data with features (X_untransformed) and targets (y_untransformed)
        loss_fn: Loss function used for evaluation
        device: PyTorch device (CPU or GPU) for computation
        n_generated_events (float, optional): Number of events generated in POWHEG-BOX-RES for normalization multiplied by
                                              relative size of the test dataset.
                                              Defaults to 0.2*1e7.
    Returns:
        float: Average test loss per batch
    Side Effects:
        - Prints testing progress and integration statistics.
        - Saves comparison plots to "test_histograms.pdf" in the model run directory.
        - Creates histograms for invariant mass (Z1), cos(theta*), ptep and yep distributions.
        - Generates integration statistics comparing predicted, true, and POWHEG results.
    Note:
        The function expects:
        - Input features X with shape (batch_size, n_particles, 4) representing 4-momenta.
        - Target y with shape (batch_size, 2) where y[:,0] are the rLL weights and y[:,1] are the UU weights.
        - Reference histograms in .top format containing "mee", "cthep", and "totxsec" observables.
        - Input features X_untransformed with shape (batch_size, n_particles, 4) representing untransformed 4-momenta.
        - Target y_untransformed with shape (batch_size, 1) where y_untransformed[:,0] are the rL weights.
    """

    logger.info("Starting testing for ZZ model...")
    size        = len(dataloader.dataset)  # Total number of samples in the dataset (= n_events).
    num_batches = len(dataloader)          # Number of batches in the dataloader.

    n_generated_events = int(n_generated_events)
    logger.info(f"Analysing {n_generated_events} generated events which result in {size} events after applying cuts.")

    # Load the LL histogram for comparison plots
    if not showered:
        powheg_histograms_paths = {"UU": histogram_dir / "pwgLHEF_analysis-mean-W8.top",
                                   "LL": histogram_dir / "pwgLHEF_analysis-mean-W9.top",
                                   "LT": histogram_dir / "pwgLHEF_analysis-mean-W10.top",
                                   "TL": histogram_dir / "pwgLHEF_analysis-mean-W11.top",
                                   "TT": histogram_dir / "pwgLHEF_analysis-mean-W12.top",
                                   "LU": histogram_dir / "pwgLHEF_analysis-mean-W13.top",
                                   "UL": histogram_dir / "pwgLHEF_analysis-mean-W14.top",
                                  }
    else:
        powheg_histograms_paths = {"UU": histogram_dir / "pwgoutput_py8_histos-mean-W8.top",
                                   "LL": histogram_dir / "pwgoutput_py8_histos-mean-W9.top",
                                   "LT": histogram_dir / "pwgoutput_py8_histos-mean-W10.top",
                                   "TL": histogram_dir / "pwgoutput_py8_histos-mean-W11.top",
                                   "TT": histogram_dir / "pwgoutput_py8_histos-mean-W12.top",
                                   "LU": histogram_dir / "pwgoutput_py8_histos-mean-W13.top",
                                   "UL": histogram_dir / "pwgoutput_py8_histos-mean-W14.top",
                                  }
    histogram_data = read_top_file_histograms(powheg_histograms_paths)
    # Normalise all runs to have the total cross section as the fitted polarisation run.
    runs_wo_fitted_polarisation = list(next(iter(histogram_data.values()), {}).keys()).copy()
    runs_wo_fitted_polarisation.remove(fitted_polarisation)
    xsec_fitted_polarisation = histogram_data['totxsec'][fitted_polarisation]['values'][0]
    for run in runs_wo_fitted_polarisation:
        observables_in_run = list(histogram_data.keys())
        observables_in_run.remove('totxsec')  # Don't rescale the total cross-section histogram itself, only the observable histograms.
        for observable in observables_in_run:
            histogram_data[observable][run]['values'] *= xsec_fitted_polarisation / histogram_data['totxsec'][run]['values'][0]

    logger.info(f"Loaded {len(histogram_data)} histograms from .top file")

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()


    observable_dict = {
                       "weights_unpolarised": [np.zeros(size), r"$\sigma_{\mathrm{UU}}$"],
                        "weights_y":          [np.zeros(size), r"$\sigma_{\mathrm{true}}$"],
                        "weights_ypred":      [np.zeros(size), r"$\sigma_{\mathrm{pred}}$"],
                       # Start observable arrays
                       "invmass_Z1":          [np.zeros(size), r"$m_{e^{+} \, e^{-}}$"],       # mee in POWHEG in analysis.
                       "invmass_Z2":          [np.zeros(size), r"$m_{\mathrm{Z} 2}$"],
                       "cthep":               [np.zeros(size), r"$\cos \theta^{*}_{e^{+}}$"], # cos(theta*) in POWHEG in analysis.
                    #    "cthep_mll_cut5":  np.zeros(size),
                    #    "cthep_mll_cut10": np.zeros(size),
                       "ptee":                [np.zeros(size), r"$p_{\mathrm{T}, \, e^{+} \, e^{-}}$"], # Transverse momentum of the Z(e+ e-) boson
                       "pt4l":                [np.zeros(size), r"$p_{\mathrm{T}, \, 4\ell}$"],          # Transverse momentum of the 4-lepton system
                       "ptep":                [np.zeros(size), r"$p_{\mathrm{T}, \, e^{+}}$"],          # Transverse momentum of the positron
                       "yep":                 [np.zeros(size), r"$y_{e^{+}}$"],                         # Rapidity of the positron
                       "dphiee":              [np.zeros(size), r"$\Delta \phi(e^{+} \, e^{-})$"],       # Delta phi between the two leptons from the Z(e+ e-) boson
                       "r_pred":      [np.zeros(size), r"$\mathrm{r}_{\mathrm{pred}}$"],
                       "r_true":      [np.zeros(size), r"$\mathrm{r}_{\mathrm{true}}$"],
                       }

    test_loss = 0
    with torch.no_grad():
        for batch, ((X, y), (X_untransformed, y_untransformed)) in enumerate(zip(dataloader, dataloader_untransformed)):
            X, y = X.to(device), y.to(device)

            if kwargs.get("input_choice", "") in ["jan2026",]:
                X_untransformed, y_untransformed = X_untransformed.to(device), y_untransformed.to(device)
            else:
                X_untransformed, y_untransformed = X.to(device), y.to(device)

            if batch == 0:
                batch_size = X.shape[0]
            y_first_weight_only = y[...,0].unsqueeze(-1)

            # Compute prediction and loss
            pred = model(X)

            test_loss += loss_fn(pred, y_first_weight_only).item()

            # Store weights for integration
            observable_dict["weights_ypred"][0][batch * batch_size : batch * batch_size + X.shape[0]] = (exp_target_transform(pred[:,0]) * exp_target_transform(y[:,1])).cpu().numpy()
            observable_dict["weights_y"][0][batch * batch_size : batch * batch_size + X.shape[0]]     = (exp_target_transform(y[:,0])    * exp_target_transform(y[:,1])).cpu().numpy()
            observable_dict["weights_unpolarised"][0][batch * batch_size : batch * batch_size + X.shape[0]] = exp_target_transform(y[:,1]).cpu().numpy()
            # The weights are calculated as an average over the number of genereated events in POWHEG-BOX-RES:
            observable_dict["weights_ypred"][0][batch * batch_size : batch * batch_size + X.shape[0]] /= n_generated_events
            observable_dict["weights_y"][0][batch * batch_size : batch * batch_size + X.shape[0]]     /= n_generated_events
            observable_dict["weights_unpolarised"][0][batch * batch_size : batch * batch_size + X.shape[0]] /= n_generated_events


            # Compute observables
            observable_dict["r_pred"][0][batch * batch_size : batch * batch_size + X.shape[0]] = exp_target_transform(pred[:,0]).cpu().numpy()
            observable_dict["r_true"][0][batch * batch_size : batch * batch_size + X.shape[0]] = exp_target_transform(y[:,0]).cpu().numpy()


            # zl1, zl2, zl3, zl4 = e+, e-, mu+, mu-
            momenta = X_untransformed.reshape(X_untransformed.shape[0], -1, 4)

            # Invariant masses of Z1 and Z2 candidates:
            invmass_Z1 = torch.sqrt((momenta[:,0,3] + momenta[:,1,3])**2 - ((momenta[:,0,0:3] + momenta[:,1,0:3])**2).sum(dim=-1) + 1e-9)
            # invmass_Z2 = torch.sqrt((momenta[:,2,3] + momenta[:,3,3])**2 - ((momenta[:,2,0:3] + momenta[:,3,0:3])**2).sum(dim=-1) + 1e-9)

            observable_dict["invmass_Z1"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = invmass_Z1.cpu().numpy()
            # observable_dict["invmass_Z2"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = invmass_Z2.cpu().numpy()
            # FIXME: Check weather the really the pt of the Z boson is calculated as the pt of the sum of the two lepton momenta.
            #        It seems to be order of magnitudes to big compared to the POWHEG histograms.
            #        Probably something is summed w.r.t. the wrong axis.
            observable_dict["ptee"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = get_pt(momenta[:,0,0:4] + momenta[:,1,0:4]).cpu().numpy()


            ct1, ct2, ct3, ct4 = costhetastar(momenta)
            observable_dict["cthep"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = ct1.cpu().numpy()

            observable_dict["ptep"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = get_pt(momenta[:,0,:]).cpu().numpy()

            observable_dict["yep"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]]  = get_rapidity(momenta[:,0,:]).cpu().numpy()
            # Note that pt4l is zero in the 4-lepton CM frame
            observable_dict["pt4l"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = get_pt(momenta.sum(dim=1)).cpu().numpy()

            # Delta phi e+ e- (just in labframe)
            observable_dict["dphiee"][0][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = getdphi(momenta[:,0,:], momenta[:,1,:]).cpu().numpy()

    # observable_dict["cthep_mll_cut10"] = np.where(np.abs(observable_dict["invmass_Z1"] - 91.19) < 10, observable_dict["cthep"], 0.0)
    # observable_dict["cthep_mll_cut5"] = np.where(np.abs(observable_dict["invmass_Z1"] - 91.19) < 5, observable_dict["cthep"], 0.0)

    # Count number of zero values in the array
    test_loss /= num_batches

    logger.info(f"Testing Error: \n Avg (per batch) test loss: {test_loss:>8f}\n")

    hist_writer = TopFileWriter(basepath = Path(f"{model_dir}"))

    with PdfPages(f"{model_dir}/test_histograms.pdf") as pdf:
        d = pdf.infodict()
        d['Title']        = f"Test results for model {model_name}"
        d['Author']       = 'You'
        d['Subject']      = 'Some comparison plots'
        d['Keywords']     = 'Machine Learning POWHEG POWHEGBOX POWHEG-BOX-RES'
        d['CreationDate'] = datetime.today()
        d['ModDate']      = datetime.today()

        show_polarisation = [fitted_polarisation, "UU"]

        # Integration statistics
        fig, _ = print_integration_statistics(observable_dict, histogram_data, model=model_name, powheg_histogram_runs = [fitted_polarisation, ], fitted_polarisation=fitted_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # Invariant mass Z1 comparison plot
        fig, _ = comparison_plots(observable_dict, "invmass_Z1", histogram_data["mee"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # pT of Z1 comparison plot
        fig, _ = comparison_plots(observable_dict, "ptee", histogram_data["ptee"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # pT of Z1 with rLL plot
        fig, _ = comparison_plots(observable_dict, "ptee", histogram_data["ptee"], model_name=model_name, powheg_histogram_runs = show_polarisation, plotrLL=True, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # Cos(theta*) comparison plot
        fig, _ = comparison_plots(observable_dict, "cthep", histogram_data["cthep"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # # Cos(theta*) with mll cut comparison plot
        # fig, axs = comparison_plots(observable_dict, "cthep_mll_cut10", histogram_data["cthep"], model_name=model_name, powheg_histogram_runs = [fitted_polarisation, ], hist_writer=hist_writer)
        # axs[0].set_title("Cos(theta*) with mll cut |mll - mZ| < 10 GeV")
        # pdf.savefig(fig)
        # plt.close(fig)

        # fig, axs = comparison_plots(observable_dict, "cthep_mll_cut5", histogram_data["cthep"], model_name=model_name, powheg_histogram_runs = [fitted_polarisation, ], hist_writer=hist_writer)
        # axs[0].set_title("Cos(theta*) with mll cut |mll - mZ| < 5 GeV")
        # pdf.savefig(fig)
        # plt.close(fig)

        # Transverse momentum of positron
        fig, _ = comparison_plots(observable_dict, "ptep", histogram_data["ptep"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # Rapidity of positron
        fig, _ = comparison_plots(observable_dict, "yep", histogram_data["yep"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # Transverse momentum of 4-lepton system
        fig, _ = comparison_plots(observable_dict, "pt4l", histogram_data["pt4l"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        # Delta phi e+ e- (just in labframe)
        fig, _ = comparison_plots(observable_dict, "dphiee", histogram_data["dphiee"], model_name=model_name, powheg_histogram_runs = show_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        fig, _, powheg_label = r_plot(observable_dict["r_pred"][0], observable_dict["r_true"][0], observable_dict["weights_unpolarised"][0], model_name=model_name, fitted_polarisation=fitted_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

        fig, _ = plot_r_distribution(observable_dict["r_pred"][0], observable_dict["r_true"][0], model_name=model_name, fitted_polarisation=fitted_polarisation, hist_writer=hist_writer)
        pdf.savefig(fig)
        plt.close(fig)

    hist_writer.save_all()

    return test_loss

# %% Define testing function for Z+jet model
def test_model_Zjet(model, model_dir, histogram_dir, dataloader, dataloader_untransformed, loss_fn, device, model_name="Model"):
    """
    Test a trained machine learning model for Z+jet events.
    This function evaluates the model on test data and computes the average test loss.
    Args:
        model: PyTorch model to be tested
        model_dir: Directory containing the model files
        histogram_dir (Path): Directory containing the total unpolarised cross-section (Events are unweighted in the Z+jet case).
        dataloader: PyTorch DataLoader containing test data with features (X) and targets (y)
        dataloader_untransformed: PyTorch DataLoader containing untransformed test data with features (X_untransformed) and targets (y_untransformed)
        loss_fn: Loss function used for evaluation
        device: PyTorch device (CPU or GPU) for computation
        n_generated_events (int): Number of generated events for normalisation
    Returns:
        float: Average test loss per batch
    Side Effects:
        - Saves comparison plots to "test_histograms.pdf" in the model run directory.
        - Creates histogram for the rLL observable.
    Note:
        The function expects:
        - Input features X with shape (batch_size, n_particles, 4) representing 4-momenta.
        - Target y with shape (batch_size, 1) where y[:,0] are the rL weights.
        - Input features X_untransformed with shape (batch_size, n_particles, 4) representing untransformed 4-momenta.
        - Target y_untransformed with shape (batch_size, 1) where y_untransformed[:,0] are the rL weights.
    """

    logger.info("Starting testing for Z+jet model...")
    size        = len(dataloader.dataset)  # Total number of samples in the dataset (= n_events).
    num_batches = len(dataloader)          # Number of batches in the dataloader.

    # In the Z+jet case, the events are unweighted, so we can directly use the number of generated events for normalisation.
    n_generated_events = size

    logger.info(f"Analysing {size} events in total.")

    # Get the total unpolarised cross-section from the xsec.txt file
    with open(histogram_dir / "xsec.txt", 'r') as f:
        lines = f.readlines()
        # Skip header lines and extract the cross-section value from the third line
        xsec_line = lines[2].strip()  # "817.3(2) pb"
        # Extract the numerical value before the parentheses
        total_xsec = float(xsec_line.split('(')[0])
    logger.info(f"Total unpolarised cross-section from MG5: {total_xsec:.6e} pb")

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    observable_dict = {"weights_y":     np.zeros(size),
                       "weights_ypred": np.zeros(size),
                       "rL_pred":       np.zeros(size),
                       "rL_true":       np.zeros(size),
                       "ptjet":         np.zeros(size),
                       "cosmupjet":     np.zeros(size),
                     }

    test_loss = 0
    with torch.no_grad():
        for batch, ((X, y), (X_untransformed, y_untransformed)) in enumerate(zip(dataloader, dataloader_untransformed)):
            X, y = X.to(device), y.to(device)
            X_untransformed, y_untransformed = X_untransformed.to(device), y_untransformed.to(device)
            if batch == 0:
                batch_size = X.shape[0]

            # Compute prediction and loss
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            # Store weights for integration
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] = (pred[:,0] * total_xsec).cpu().numpy()
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     = (y[:,0]    * total_xsec).cpu().numpy()
            # The weights are calculated as an average over the number of genereated events in POWHEG-BOX-RES:
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] /= n_generated_events
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     /= n_generated_events

            # Compute observables
            observable_dict["rL_pred"][batch * batch_size : batch * batch_size + X.shape[0]] = pred[:,0].cpu().numpy()
            observable_dict["rL_true"][batch * batch_size : batch * batch_size + X.shape[0]] = y[:,0].cpu().numpy()

            # zl1, zl2, jet =  mu+, mu-, jet
            momenta = X_untransformed.reshape(X_untransformed.shape[0], -1, 4)

            observable_dict["ptjet"][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = get_pt(momenta[:,2,:]).cpu().numpy()

            ct1, ct2 = cosmujet(momenta)
            observable_dict["cosmupjet"][batch * batch_size : batch * batch_size + X_untransformed.shape[0]] = ct1.cpu().numpy()

    test_loss /= num_batches

    logger.info(f"Testing Error: \n Avg (per batch) test loss: {test_loss:>8f}\n")

    with PdfPages(f"{model_dir}/test_histograms.pdf") as pdf:
        d = pdf.infodict()
        d['Title']        = f"Test results for model {model_name}"
        d['Author']       = 'You'
        d['Subject']      = 'Some comparison plots'
        d['Keywords']     = 'Machine Learning POWHEG POWHEGBOX POWHEG-BOX-RES'
        d['CreationDate'] = datetime.today()
        d['ModDate']      = datetime.today()

        fig, _ = print_integration_statistics(observable_dict, model=model_name)
        pdf.savefig(fig)
        plt.close(fig)

        # Invariant pT of the jet
        fig, _ = comparison_plots(observable_dict, "ptjet", model_name=model_name)
        pdf.savefig(fig)
        plt.close(fig)

        # Cos(theta*) of the mu+ and jet in Z+jet CM frame
        fig, _ = comparison_plots(observable_dict, "cosmupjet", log_scale=False, nbins=30, model_name=model_name)
        pdf.savefig(fig)
        plt.close(fig)

        fig, _, powheg_label = r_plot(observable_dict["rL_pred"], observable_dict["rL_true"], total_xsec, model_name=model_name)
        pdf.savefig(fig)
        plt.close(fig)

    return test_loss

def do_test_run(run_settings: Settings, model, test_dataset):
    start_time = time.time()

    device = run_settings.device.value
    use_zjet = run_settings.use_zjet.value
    model_name = run_settings.model.value
    model_dir = run_settings.model_dir.value
    histogram_dir = run_settings.histogram_dir.value
    mlfiles = run_settings.mlfiles.value
    seed = run_settings.seed.value
    split_ratios = run_settings.split_ratios.value
    polarisation = run_settings.polarisation.value
    batch_size = run_settings.batch_size.value
    n_workers = run_settings.nworkers.value
    n_generated_events = run_settings.n_generated_events.value
    input_choice = run_settings.input_choice.value
    showered = run_settings.showered.value

    torch.manual_seed(seed)
    np.random.seed(seed)

    if not use_zjet:
        # ZZ case
        labels = [f"{polarisation}/UU", "UU"]
        dataset_untransformed = MLEventsDataset(mlfiles,
                                                # target_transform=log_target_transform,  # Apply log transform to reduce outlier impact
                                                # inv_target_transform=exp_target_transform,  # Inverse transform to revert log transformation
                                                labels = labels,
                                                cache_events=True,
                                                standardise=False)
    else:
        # Z+jet case
        dataset_untransformed = ZJetDataset(mlfiles[0],
                                max_events=None,  # Maximum number of events to load (useful for testing). Max = 10^6.
                                standardise=False)

    # Identical random number generator for the untransformed dataset to get the same test/ train split as used during training.
    test_generator = torch.Generator().manual_seed(seed)
    _, _, test_dataset_untransformed = torch.utils.data.random_split(dataset_untransformed, split_ratios, generator=test_generator)

    # Get the test dataloaders for the transformed and untransformed datasets
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # Larger batch size for efficiency
        shuffle=False,
        num_workers=n_workers,  # Use multiple workers for large files
        pin_memory=True  # Faster GPU transfer
    )
    test_dataloader_untransformed = DataLoader(
        test_dataset_untransformed,
        batch_size=batch_size,  # Larger batch size for efficiency
        shuffle=False,
        num_workers=n_workers,  # Use multiple workers for large files
        pin_memory=True  # Faster GPU transfer
    )

    # Test iteration (only first batch to avoid long output)
    for batch_idx, (batch_features, batch_labels) in enumerate(test_dataloader):
        logger.info(f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")
        input_dim = batch_features.shape[1]
        logger.info(f"{input_dim = }")
        break  # Only show first batch


    test_loss_fn = torch.nn.MSELoss()

    if not use_zjet:
        test_loss = test_model_ZZ(model, model_dir, histogram_dir, test_dataloader, test_dataloader_untransformed, test_loss_fn, device, split_ratios[2] * n_generated_events, model_name=model_name, fitted_polarisation=polarisation, input_choice=input_choice, showered=showered)
    else:
        test_loss = test_model_Zjet(model, model_dir, histogram_dir, test_dataloader, test_dataloader_untransformed, test_loss_fn, device, model_name=model_name)

    end_time = time.time()
    logger.info(f"Testing completed in {end_time - start_time:.2f} seconds.")

    return test_loss


def parse_args() -> argparse.Namespace:
    # %%
    parser = argparse.ArgumentParser(
        description='Test the already trained neural network for polarisation tagging.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mlfiles", nargs='+', type=Path,        action="store",               help=".ml files to be used for training.")
    parser.add_argument("model",              type=str,         action="store",               help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
    parser.add_argument("model_weight_file",  type=Path,        action="store",               help="Path to the .pt(y) file containing the trained model weights.")
    parser.add_argument("-g", "--gpu",        type=int,         action="store", default=-1,   help="Specify manually which of the available gpus is supposed to be used.")
    parser.add_argument("-b", "--batch_size", type=int,         action="store", default=512,  help="Batch size for training.")
    parser.add_argument("-n", "--nworkers",   type=int,         action="store", default=0,    help="Number of workers for DataLoader.")
    parser.add_argument("-t", "--test_mode",  dest="test_mode", action="store_true",          help="Run in test mode (only one data point to test implementation of the model).")
    parser.add_argument("--inputdir",         type=Path,        action="store", default=None, help='Specify name of input directory.')
    parser.add_argument("--histogram_dir",    type=Path,        action="store", default=None, help='Directory containing the .top histogram files for comparison (They are in the folder where also the events are.).')
    parser.add_argument("--n_generated_events", type=lambda x: int(float(x)),       action="store", default=int(1e7), help="Number of generated events for comparison (1e7 for LO and LOwS and 5e6 for NLO).")
    parser.add_argument("--useZjet",          dest="use_zjet",  action="store_true",          help="Use Z+jet dataset instead of default.")
    parser.add_argument("--standardise",      dest="standardise", action="store_true",        help="Enable standardisation of features over the whole dataset (default).")
    parser.add_argument("--input_choice",     type=str,         action="store", default=None, help="Choice of input features. Options: Momenta, jan2026.")
    parser.add_argument("--polarisation",     type=str,         action="store", default="LL", help="Specify which polarisation to train on (Only relevant for ZZ). Options: LL, LT, TL, TT, UL, LU.")

    # Create a mutually exclusive group for specifying the reference frame
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--labframe",       dest="labframe", default=True, action="store_true",    help="Use lab frame instead of partonic CMS.")
    frame_group.add_argument("--cmframe",        dest="labframe", default=True, action="store_false",   help="Use partonic CMS instead of lab frame.")

    args = parser.parse_args()

    return args


def namespace_from_settings(run_settings: Settings) -> argparse.Namespace:
    return argparse.Namespace(**{key: parameter.value for key, parameter in run_settings.items()})


def select_device(gpu: int) -> str:
    # %% Specify the computation device (cpu or gpu).
    # In torch/pytorch data and models need to be moved in the specific processing unit
    # this code snippet allows to set the variable "device" according to available resource (cpu or cuda gpu)
    if torch.cuda.is_available():
        logger.info(f"Number of devices: {torch.cuda.device_count()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

    if torch.cuda.is_available():
        if gpu >= 0:
            device = f"cuda:{gpu}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Computation device: {device}")

    # Set CUDA device globally
    if torch.cuda.is_available():
        if gpu >= 0:
            torch.cuda.set_device(gpu)
            logger.info(f"Set CUDA device to: {gpu}")
        else:
            torch.cuda.set_device(0)
            logger.info(f"Set CUDA device to: 0")
    return device


def run_testing(run_settings: Settings):
    arg = namespace_from_settings(run_settings)

    run_settings.log_to_logger(logger, header="Arguments:")

    device = select_device(arg.gpu)
    run_settings.set("device", device, overwrite=True)

    # %% Model selection
    model_name = arg.model
    if model_name not in model_dict:
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(model_dict.keys())}")
    else:
        logger.info(f"Using model architecture: {model_name}")

    if arg.inputdir is not None:
        model_dir = arg.inputdir
    else:
        model_dir = Path().cwd()

    if arg.histogram_dir is None:
        arg.histogram_dir = arg.mlfiles[0].parent

    run_settings.set("model_dir", model_dir, overwrite=True)
    run_settings.set("histogram_dir", arg.histogram_dir, overwrite=True)

    # %% Data Handling
    # Example usage for large files:
    # files = Path("event_files/pwgevents-*.ml")
    files = arg.mlfiles

    if not arg.use_zjet:
        if arg.labframe:
            trafo = None
        else:
            trafo = boost_into_four_lepton_cm_frame

        if arg.input_choice == "jan2026":
            if trafo:
                trafo = lambda x: januar2026_input_choice(trafo(x))
            else:
                trafo = januar2026_input_choice

        labels = [f"{arg.polarisation}/UU", "UU"]
        dataset = MLEventsDataset(files,
                                  labels = labels,
                                  transform=trafo,
                                  target_transform=log_target_transform,  # Apply log transform to reduce outlier impact
                                  inv_target_transform=exp_target_transform,  # Inverse transform to revert log transformation
                                  cache_events=True,  # Caching enabled
                                  standardise=False)  # Standardisation is add by now as an additional layer in the model, whose weights are loaded from the state dict of the trained model.

    else:
        if arg.labframe:
            trafo = None
        else:
            trafo = boost_into_Zjet_cm_frame

        if arg.input_choice == "jan2026":
            if trafo:
                trafo = lambda x: januar2026_input_choice(trafo(x))
            else:
                trafo = januar2026_input_choice

        dataset = ZJetDataset(files[0],
                            transform=trafo,
                            target_transform=None,
                            max_events=None,  # Maximum number of events to load (useful for testing). Max = 10^6.
                            standardise=False)  # Standardisation is add by now as an additional layer in the model, whose weights are loaded from the state dict of the trained model.

    logger.info(f"Dataset info: {dataset.get_file_info()}")

    # Set fixed random number seed to get the same test/ train split as used during training
    logger.info(Path.cwd())
    with open(model_dir / "training_seed.txt", 'r') as f:
        seed = int(f.readline().strip())

    run_settings.set("seed", seed, overwrite=True)
    generator = torch.Generator().manual_seed(seed)

    split_ratios = run_settings.split_ratios.value if hasattr(run_settings, "split_ratios") else [0.6, 0.2, 0.2]
    run_settings.set("split_ratios", split_ratios, overwrite=True)

    _, _, test_dataset = torch.utils.data.random_split(dataset, split_ratios, generator=generator)
    logger.info(f"Test dataset size:       {len(test_dataset)}")

    # Initialize the model and load the trained weights
    input_dim = dataset.input_shape[0]
    if arg.standardise:
        model = model_dict[arg.model](input_dim=input_dim, external_stat=True)
    else:
        model = model_dict[arg.model](input_dim=input_dim)

    if arg.model_weight_file.is_absolute():
        model_weight_file = arg.model_weight_file
    else:
        model_weight_file = model_dir / arg.model_weight_file

    # if torch.cuda.is_available():
    #     model_summary = str(summary(model.cuda(), input_size=(input_dim,), verbose=0))
    # else:
    #     model_summary = str(summary(model, input_size=(input_dim,), verbose=0))

    # logger.info(f"\n{model_summary}")

    model.load_state_dict(torch.load(model_weight_file, map_location=device, weights_only=True))
    model.to(device)

    return do_test_run(run_settings, model, test_dataset)


def prepare_run_settings(arg: argparse.Namespace) -> Settings:
    run_settings = Settings(argparse=arg)
    run_settings.set("split_ratios", [0.6, 0.2, 0.2])
    run_settings.set("showered", False)
    return run_settings

# %% Run the test
if __name__ == "__main__":
    logger = setup_file_logger(log_file=log_file, level="DEBUG", mode="a", console=False, force=True)

    logger.info(f"numpy:  {np.__version__}")
    logger.info(f"pandas: {pd.__version__}")
    logger.info(f"torch:  {torch.__version__}")

    arg          = parse_args()
    run_settings = prepare_run_settings(arg)

    try:
        run_testing(run_settings)
    except Exception as exc:
        logger.error(f"Testing failed: {exc}")
        sys.exit(1)

else:
    logger = logging.getLogger(__name__)

