from datetime import datetime
import time
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from torchsummary import summary
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

from ml_events_utils import MLEventsDataset, scale_target, boost_into_four_lepton_cm_frame  #, test_loop
from ml_events_utils import ZJetDataset
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
from ml_events_utils.analysis import costhetastar, get_pt, get_rapidity, cosmujet
import argparse

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('torch', torch.__version__)



# %%
parser = argparse.ArgumentParser(
    description='Test the already trained neural network for polarisation tagging.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("mlfiles", nargs='+', type=Path,        action="store",               help=".ml files to be used for training.")
parser.add_argument("model",              type=str,         action="store",               help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
parser.add_argument("model_weight_file",  type=Path,        action="store",               help="Path to the .pt(y) file containing the trained model weights.")
parser.add_argument("-g", "--gpu",        type=int,         action="store", default=-1,   help="Specify manually which of the available gpus is supposed to be used.")
parser.add_argument("-b", "--batch_size", type=int,         action="store", default=128,  help="Batch size for training.")
parser.add_argument("-n", "--nworkers",   type=int,         action="store", default=0,    help="Number of workers for DataLoader.")
parser.add_argument("-t", "--test_mode",  dest="test_mode", action="store_true",          help="Run in test mode (only one data point to test implementation of the model).")
parser.add_argument("--inputdir",         type=Path,        action="store", default=None, help='Specify name of input directory.')
parser.add_argument("--histogram_dir",    type=Path,        action="store", default=None, help='Directory containing the .top histogram files for comparison (They are in the folder where also the events are.).')
parser.add_argument("-e", "--n_generated_events", type=lambda x: int(float(x)),       action="store", default=int(1e7), help="Number of generated events for comparison (1e7 for LO and LOwS and 5e6 for NLO).")
parser.add_argument("--useZjet",          dest="use_zjet",  action="store_true",          help="Use Z+jet dataset instead of default.")
parser.add_argument("--standardise",      dest="standardise", action="store_true",        help="Enable standardisation of features over the whole dataset (default).")

# Create a mutually exclusive group for specifying the reference frame
frame_group = parser.add_mutually_exclusive_group()
frame_group.add_argument("--labframe",       dest="labframe", default=True, action="store_true",    help="Use lab frame instead of partonic CMS.")
frame_group.add_argument("--cmframe",        dest="labframe", default=True, action="store_false",   help="Use partonic CMS instead of lab frame.")

arg = parser.parse_args()

print("Arguments:")
for attr, value in vars(arg).items():
    print(f"  {attr}: {value}")

# %% Model selection
model_name = arg.model
if model_name not in model_dict:
    raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(model_dict.keys())}")
else:
    print(f"Using model architecture: {model_name}")

if arg.inputdir is not None:
    model_dir = arg.inputdir
else:
    model_dir = Path(model_name)

if arg.histogram_dir is None:
    arg.histogram_dir = arg.mlfiles[0].parent

# %% Specify the computation device (cpu or gpu).
# In torch/pytorch data and models need to be moved in the specific processing unit
# this code snippet allows to set the variable "device" according to available resource (cpu or cuda gpu)

if torch.cuda.is_available():
  print('Number of devices: ', torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    if arg.gpu >= 0:
        device = f"cuda:{arg.gpu}"
    else:
        device = "cuda"
else:
    device = "cpu"
print(f"Computation device: {device}\n")

# Set CUDA device globally
if torch.cuda.is_available():
    if arg.gpu >= 0:
        torch.cuda.set_device(arg.gpu)
        print(f"Set CUDA device to: {arg.gpu}")
    else:
        torch.cuda.set_device(0)
        print(f"Set CUDA device to: 0")

# %% Set fixed random number seed to get the same test/ train split as used during training
print(Path.cwd())
with open(model_dir / "training_seed.txt", 'r') as f:
    seed = int(f.readline().strip())

torch.manual_seed(seed)
np.random.seed(seed)
# %% Data Handling

# Example usage for large files:
# files = Path("event_files/pwgevents-*.ml")
files = arg.mlfiles

if not arg.use_zjet:
    if arg.labframe:
        trafo = None
    else:
        trafo = boost_into_four_lepton_cm_frame
    dataset = MLEventsDataset(files,
                            #   labels = ["LL/UU", ],
                            labels = ["LL/UU", "UU"],
                            transform=trafo,
                            #   target_transform=scale_target,  # Scale target by 1000
                            cache_events=True,  # Caching enabled
                            standardise=False)  # Standardisation is add by now as an additional layer in the model, whose weights are loaded from the state dict of the trained model.
else:
    if arg.labframe:
        trafo = None
    else:
        trafo = boost_into_Zjet_cm_frame
    dataset = ZJetDataset(files[0],
                          transform=trafo,
                          target_transform=None,
                          max_events=None,  # Maximum number of events to load (useful for testing). Max = 10^6.
                          standardise=False)  # Standardisation is add by now as an additional layer in the model, whose weights are loaded from the state dict of the trained model.
print(f"Dataset info: {dataset.get_file_info()}")

# %% Hyperparameters
batch_size    = arg.batch_size     # 128
n_workers     = arg.nworkers       # Use multiple (default 4) workers for DataLoader

# %% Get the test dataloader
generator = torch.Generator().manual_seed(seed)

split_ratios = [0.6, 0.2, 0.2]  # Train, Val, Test
_, _, test_dataset = torch.utils.data.random_split(dataset, split_ratios, generator=generator)
print(f"Test dataset size:       {len(test_dataset)}")


test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=False,
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True  # Faster GPU transfer
)

# Test iteration (only first batch to avoid long output)
for batch_idx, (batch_features, batch_labels) in enumerate(test_dataloader):
    print(f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")
    input_dim = batch_features.shape[1]
    print(f"{input_dim = }")
    break  # Only show first batch

# %% Initialize the model and load the trained weights
if arg.standardise:
    model = model_dict[arg.model](input_dim=input_dim, external_stat=True)
else:
    model = model_dict[arg.model](input_dim=input_dim)

if arg.model_weight_file.is_absolute():
    model_weight_file = arg.model_weight_file
    model_run_dir     = model_weight_file.parent
else:
    model_weight_file = model_dir / arg.model_weight_file
    model_run_dir     = model_dir


if torch.cuda.is_available():
  summary(model.cuda(), input_size=(input_dim,))
else:
  summary(model, input_size=(input_dim,))

model.load_state_dict(torch.load(model_weight_file, map_location=device, weights_only=True))
model.to(device)

# %% Histogram reading function
def read_top_file_histograms(top_file_path):
    """
    Read histogram data from a .top file.

    Args:
        top_file_path (Path): Path to the .top file

    Returns:
        dict: Dictionary with histogram names as keys, each containing:
            - 'bin_left': numpy array of left bin edges
            - 'bin_right': numpy array of right bin edges
            - 'values': numpy array of histogram values
            - 'uncertainties': numpy array of uncertainties
    """
    histogram_data = {}

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
                histogram_data[current_histogram] = {
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

                        histogram_data[current_histogram]['bin_left'].append(bin_left)
                        histogram_data[current_histogram]['bin_right'].append(bin_right)
                        histogram_data[current_histogram]['values'].append(value)
                        histogram_data[current_histogram]['uncertainties'].append(uncertainty)
                except ValueError:
                    # Skip lines that can't be parsed as numbers
                    continue

    # Convert lists to numpy arrays for easier manipulation
    for hist_name in histogram_data:
        for key in histogram_data[hist_name]:
            histogram_data[hist_name][key] = np.array(histogram_data[hist_name][key])
            histogram_data[hist_name]['edges'] = np.concatenate((
                histogram_data[hist_name]['bin_left'],
                histogram_data[hist_name]['bin_right'][-1:]
            ))

    return histogram_data

def print_integration_statistics(observable_dict, histogram_data: dict = {}):
    pred_integral = np.sum(observable_dict["weights_ypred"])
    true_integral = np.sum(observable_dict["weights_y"])

    # Note that we can in principle also integrate over the histograms here for cross-checks:
    # pred_integral = np.sum(pred_sums * bin_widths)
    # true_integral = np.sum(true_sums * bin_widths)

    print(f"Integrated cross-sections:")
    print(f"  True r_LL:          {true_integral:.6e}")
    print(f"  Predicted r_LL:     {pred_integral:.6e}")
    if histogram_data:
        print(f"  POWHEG reweighting: {histogram_data['totxsec']['values'][0]:.6e}")
        print(f"  Ratio (pred/PWG):   {pred_integral/histogram_data['totxsec']['values'][0]:.6f}")
    print(f"  Ratio (pred/true):  {pred_integral/true_integral:.6f}")

    # Create a text-only plot for integration results
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')  # Remove axes

    text_content = f"""    True r_LL:          {true_integral:.6e}
    Predicted r_LL:     {pred_integral:.6e}"""
    if histogram_data:
        text_content += f"""
    POWHEG reweighting: {histogram_data['totxsec']['values'][0]:.6e}
    Ratio (pred/PWG):   {pred_integral/histogram_data['totxsec']['values'][0]:.6f}"""

    text_content += f"""
    Ratio (pred/true):  {pred_integral/true_integral:.6f}"""

    ax.text(0.1, 0.5, text_content, fontsize=14, verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    ax.set_title('Integrated cross-sections', fontsize=16, fontweight='bold')

    return fig, ax

def r_plot(r_pred, r_true, weights):
    fig, axs = plt.subplots(1, 1)

    r_min, r_max = min(r_pred.min() ,r_true.min()), max(r_pred.max(), r_true.max())
    bins = np.linspace(r_min, r_max, 51)

    # Calculate bin widths for proper integration
    bin_widths = bins[1:] - bins[:-1]

    # Sum predicted labels in each r bin
    pred_sums, _ = np.histogram(r_pred, bins=bins, weights=weights)
    pred_sums /= bin_widths
    true_sums, _ = np.histogram(r_true, bins=bins, weights=weights)
    true_sums /= bin_widths

    # Plot as step histograms
    bin_centers = (bins[:-1] + bins[1:]) / 2

    axs.step(bin_centers, true_sums, where='mid', label='True r',      color='green', linewidth=2, alpha=0.7)
    axs.step(bin_centers, pred_sums, where='mid', label='Predicted r', color='red',   linewidth=2, alpha=0.7)

    # Scale y axis logarithmically
    # axs[0].set_yscale('log')

    axs.set_ylabel(r"$\frac{\mathrm{d} \sigma}{\mathrm{d} r}$ [pb / [r]]")
    axs.set_title("Predicted vs. True Labels")
    axs.legend()
    axs.grid(True, alpha=0.3)

    try:
        axs.set_xlim(xmin=r_min * 0.99, xmax=r_max * 1.01)
    except ValueError as e:
        print(f"Could not set x limits for r plot: {e}")

    axs.set_xlabel("r")

    fig.tight_layout()
    return fig, axs


def comparison_plots(observable_dict:dict, observable_key:str, powheg_histogram:dict = None, log_scale=True, nbins=50):
    """
    Create a comparison plot of predicted vs true labels for a given observable.
    This function generates a step histogram plot comparing predicted labels, true labels,
    and (if given) POWHEG reference data for a specified observable. The histograms are normalized
    by bin width and displayed on a logarithmic y-scale.
    Note the slight difference of the true labels and POWHEG histograms due to FIXME: Add explanation here.
    Args:
        observable_dict (dict): Dictionary containing observable data with keys:
            - observable_key: The observable values to plot
            - "weights_ypred": Predicted label weights
            - "weights_y": True label weights
            - "invmass_Z1": Invariant mass values (used for x-axis limits)
        observable_key (str): Key specifying which observable to plot from observable_dict
        powheg_histogram (dict, optional): Optional dictionary used as an additional compariosn
            containing POWHEG reference data with keys:
            - 'edges': Bin edges for the histogram
            - 'values': Histogram values for comparison
        log_scale (bool, optional): Whether to use logarithmic scaling for the y-axis. Default is True.
        nbins (int, optional): Number of bins to use if powheg_histogram is not provided. Default is 50.
    Returns:
        tuple: Figure and axes objects (fig, axs) for the created plot
    Note:
        The function creates step histograms normalized by bin width, plots them with
        different colors (green for true, red for predicted, blue for POWHEG), and
        applies logarithmic scaling to the y-axis. The plot includes a legend, grid,
        and appropriate labels.
    """

    # Create a single comparison plot
    fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1])

    if powheg_histogram:
        bins = powheg_histogram['edges']
    else:
        bins = np.linspace(observable_dict[observable_key].min(), observable_dict[observable_key].max(), nbins+1)

    # Calculate bin widths for proper integration
    bin_widths = bins[1:] - bins[:-1]

    # Sum predicted labels in each invariant mass bin
    pred_sums, _ = np.histogram(observable_dict[observable_key], bins=bins, weights=observable_dict["weights_ypred"])
    pred_sums /= bin_widths
    # Sum true labels in each invariant mass bin
    true_sums, _ = np.histogram(observable_dict[observable_key], bins=bins, weights=observable_dict["weights_y"])
    true_sums /= bin_widths
    if powheg_histogram:
        # POWHEG histograms for comparison
        powheg_sums = powheg_histogram['values']

    # Plot as step histograms
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axs[0].step(bin_centers, true_sums,   where='mid', label='True Labels',      color='green', linewidth=2, alpha=0.7)
    axs[0].plot(bin_centers, true_sums, 'x', color='green', markersize=8, alpha=0.7)
    if powheg_histogram:
        axs[0].step(bin_centers, powheg_sums, where='mid', label='POWHEG Labels',    color='blue',  linewidth=2, alpha=0.7)
    axs[0].step(bin_centers, pred_sums,   where='mid', label='Predicted Labels', color='red',   linewidth=2, alpha=0.7)

    axs[1].step(bin_centers, pred_sums / np.maximum(true_sums, 1e-10), where='mid', color='green', linewidth=2, alpha=0.7)
    if powheg_histogram:
        axs[1].step(bin_centers, pred_sums / np.maximum(powheg_sums, 1e-10), where='mid', color='blue', linewidth=2, alpha=0.7, linestyle='--')

    axs[1].axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axs[1].set_ylabel("Predicted / X")

    # Scale y axis logarithmically
    if log_scale:
        try:
            if np.all(true_sums > 0) and np.all(pred_sums > 0) and (not powheg_histogram or np.all(powheg_sums > 0)):
                axs[0].set_yscale('log')
            else:
                print(f"Not all histogram values are positive for {observable_key} plot; skipping log scale.")
        except ValueError as e:
            print(f"Could not set y scale to log for {observable_key} plot: {e}")

    axs[0].set_ylabel(r"$\frac{\mathrm{d} \sigma}{\mathrm{d} \mathrm{" + observable_key + r"}}$ [pb / [" + observable_key + "]]")
    axs[0].set_title("Predicted vs. True Labels")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)

    try:
        axs[1].set_xlim(xmin=observable_dict[observable_key].min() * 0.99, xmax=observable_dict[observable_key].max() * 1.01)
    except ValueError as e:
        print(f"Could not set x limits for {observable_key} plot: {e}")

    axs[1].set_xlabel(f"{observable_key}")

    fig.tight_layout()

    return fig, axs


# %% Testing loop
def test_model_ZZ(model, model_dir, histogram_dir, dataloader, loss_fn, device, n_generated_events=0.2*1e7):
    """
    Test a trained machine learning model and generate comparison plots with POWHEG reference data.
    This function evaluates the model on test data, computes observables (invariant masses and cos(theta*)),
    creates histograms comparing predicted vs. true labels vs. POWHEG results, and saves the plots to a PDF.
    Args:
        model: PyTorch model to be tested
        model_dir: Directory containing the model files
        histogram_dir (Path): Directory containing reference histogram files (.top format)
        dataloader: PyTorch DataLoader containing test data with features (X) and targets (y)
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
    """

    print("Starting testing for ZZ model...")
    size        = len(dataloader.dataset)  # Total number of samples in the dataset (= n_events).
    num_batches = len(dataloader)          # Number of batches in the dataloader.

    n_generated_events = int(n_generated_events)
    print(f"Analysing {n_generated_events} generated events which result in {size} events after applying cuts.")

    # Load the LL histogram for comparison plots
    histogram_data = read_top_file_histograms(histogram_dir / "pwgLHEF_analysis-mean-W8.top")
    print(f"Loaded {len(histogram_data)} histograms from .top file")

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()


    observable_dict = {"weights_y":     np.zeros(size),
                       "weights_ypred": np.zeros(size),
                       # Start observable arrays
                       "invmass_Z1":    np.zeros(size),
                       "invmass_Z2":    np.zeros(size),
                       "cthep":         np.zeros(size),
                       "cthep_mll_cut5":  np.zeros(size),
                       "cthep_mll_cut10": np.zeros(size),
                       "pt4l":          np.zeros(size),
                       "ptep":          np.zeros(size),
                       "yep":           np.zeros(size),
                       "rLL_pred":      np.zeros(size),
                       "rLL_true":      np.zeros(size),
                       }

    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            if batch == 0:
                batch_size = X.shape[0]
            y_first_weight_only = y[...,0].unsqueeze(-1)

            # Compute prediction and loss
            pred = model(X)

            test_loss += loss_fn(pred, y_first_weight_only).item()

            # Store weights for integration
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] = (pred[:,0] * y[:,1]).cpu().numpy()
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     = (y[:,0]    * y[:,1]).cpu().numpy()
            # The weights are calculated as an average over the number of genereated events in POWHEG-BOX-RES:
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] /= n_generated_events
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     /= n_generated_events


            # Compute observables
            observable_dict["rLL_pred"][batch * batch_size : batch * batch_size + X.shape[0]] = pred[:,0].cpu().numpy()
            observable_dict["rLL_true"][batch * batch_size : batch * batch_size + X.shape[0]] = y[:,0].cpu().numpy()


            # zl1, zl2, zl3, zl4 = e+, e-, mu+, mu-
            momenta = X.reshape(X.shape[0], -1, 4)

            # Invariant masses of Z1 and Z2 candidates:
            invmass_Z1 = torch.sqrt((momenta[:,0,3] + momenta[:,1,3])**2 - ((momenta[:,0,0:3] + momenta[:,1,0:3])**2).sum(dim=-1) + 1e-9)
            # invmass_Z2 = torch.sqrt((momenta[:,2,3] + momenta[:,3,3])**2 - ((momenta[:,2,0:3] + momenta[:,3,0:3])**2).sum(dim=-1) + 1e-9)

            observable_dict["invmass_Z1"][batch * batch_size : batch * batch_size + X.shape[0]] = invmass_Z1.cpu().numpy()
            # observable_dict["invmass_Z2"][batch * batch_size : batch * batch_size + X.shape[0]] = invmass_Z2.cpu().numpy()


            ct1, ct2, ct3, ct4 = costhetastar(momenta)
            observable_dict["cthep"][batch * batch_size : batch * batch_size + X.shape[0]] = ct1.cpu().numpy()

            observable_dict["ptep"][batch * batch_size : batch * batch_size + X.shape[0]] = get_pt(momenta[:,0,:]).cpu().numpy()

            observable_dict["yep"][batch * batch_size : batch * batch_size + X.shape[0]]  = get_rapidity(momenta[:,0,:]).cpu().numpy()
            # Note that pt4l is zero in the 4-lepton CM frame
            # observable_dict["pt4l"][batch * batch_size : batch * batch_size + X.shape[0]] = get_pt(momenta.sum(dim=1)).cpu().numpy()

    observable_dict["cthep_mll_cut10"] = np.where(np.abs(observable_dict["invmass_Z1"] - 91.19) < 10, observable_dict["cthep"], 0.0)
    observable_dict["cthep_mll_cut5"] = np.where(np.abs(observable_dict["invmass_Z1"] - 91.19) < 5, observable_dict["cthep"], 0.0)

    # Count number of zero values in the array
    test_loss /= num_batches

    print(f"Testing Error: \n Avg (per batch) test loss: {test_loss:>8f}\n")

    with PdfPages(f"{model_run_dir}/test_histograms.pdf") as pdf:
        d = pdf.infodict()
        d['Title']        = f"Test results for model {model_name}"
        d['Author']       = 'You'
        d['Subject']      = 'Some comparison plots'
        d['Keywords']     = 'Machine Learning POWHEG POWHEGBOX POWHEG-BOX-RES'
        d['CreationDate'] = datetime.today()
        d['ModDate']      = datetime.today()

        # Integration statistics
        fig, _ = print_integration_statistics(observable_dict, histogram_data)
        pdf.savefig(fig)
        plt.close(fig)

        # Invariant mass Z1 comparison plot
        fig, _ = comparison_plots(observable_dict, "invmass_Z1", histogram_data["mee"])
        pdf.savefig(fig)
        plt.close(fig)

        # Cos(theta*) comparison plot
        fig, _ = comparison_plots(observable_dict, "cthep", histogram_data["cthep"])
        pdf.savefig(fig)
        plt.close(fig)

        # Cos(theta*) with mll cut comparison plot
        fig, axs = comparison_plots(observable_dict, "cthep_mll_cut10", histogram_data["cthep"])
        axs[0].set_title("Cos(theta*) with mll cut |mll - mZ| < 10 GeV")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = comparison_plots(observable_dict, "cthep_mll_cut5", histogram_data["cthep"])
        axs[0].set_title("Cos(theta*) with mll cut |mll - mZ| < 5 GeV")
        pdf.savefig(fig)
        plt.close(fig)

        # Transverse momentum of positron
        fig, _ = comparison_plots(observable_dict, "ptep", histogram_data["ptep"])
        pdf.savefig(fig)
        plt.close(fig)

        # Rapidity of positron
        fig, _ = comparison_plots(observable_dict, "yep", histogram_data["yep"])
        pdf.savefig(fig)
        plt.close(fig)

        # # Transverse momentum of 4-lepton system
        # fig, _ = comparison_plots(observable_dict, "pt4l", histogram_data["pt4l"])
        # pdf.savefig(fig)
        # plt.close(fig)

        fig, _ = r_plot(observable_dict["rLL_pred"], observable_dict["rLL_true"], observable_dict["weights_y"])
        pdf.savefig(fig)
        plt.close(fig)

    return test_loss

# %% Define testing function for Z+jet model
def test_model_Zjet(model, model_dir, histogram_dir, dataloader, loss_fn, device):
    """
    Test a trained machine learning model for Z+jet events.
    This function evaluates the model on test data and computes the average test loss.
    Args:
        model: PyTorch model to be tested
        model_dir: Directory containing the model files
        histogram_dir (Path): Directory containing the total unpolarised cross-section (Events are unweighted in the Z+jet case).
        dataloader: PyTorch DataLoader containing test data with features (X) and targets (y)
        loss_fn: Loss function used for evaluation
        device: PyTorch device (CPU or GPU) for computation
    Returns:
        float: Average test loss per batch
    Side Effects:
        - Saves comparison plots to "test_histograms.pdf" in the model run directory.
        - Creates histogram for the rLL observable.
    Note:
        The function expects:
        - Input features X with shape (batch_size, n_particles, 4) representing 4-momenta.
        - Target y with shape (batch_size, 1) where y[:,0] are the rL weights.
    """

    print("Starting testing for Z+jet model...")
    size        = len(dataloader.dataset)  # Total number of samples in the dataset (= n_events).
    num_batches = len(dataloader)          # Number of batches in the dataloader.

    print(f"Analysing {size} events in total.")

    # Get the total unpolarised cross-section from the xsec.txt file
    with open(histogram_dir / "xsec.txt", 'r') as f:
        lines = f.readlines()
        # Skip header lines and extract the cross-section value from the third line
        xsec_line = lines[2].strip()  # "817.3(2) pb"
        # Extract the numerical value before the parentheses
        total_xsec = float(xsec_line.split('(')[0])
    print(f"Total unpolarised cross-section from MG5: {total_xsec:.6e} pb")

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
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            if batch == 0:
                batch_size = X.shape[0]

            # Compute prediction and loss
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            # Store weights for integration
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] = (pred[:,0] * total_xsec).cpu().numpy()
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     = (y[:,0]    * total_xsec).cpu().numpy()
            # The weights are calculated as an average over the number of genereated events in POWHEG-BOX-RES:
            n_generated_events = size  # TODO: Check if this is correct for Z+jet case
            observable_dict["weights_ypred"][batch * batch_size : batch * batch_size + X.shape[0]] /= n_generated_events
            observable_dict["weights_y"][batch * batch_size : batch * batch_size + X.shape[0]]     /= n_generated_events

            # Compute observables
            observable_dict["rL_pred"][batch * batch_size : batch * batch_size + X.shape[0]] = pred[:,0].cpu().numpy()
            observable_dict["rL_true"][batch * batch_size : batch * batch_size + X.shape[0]] = y[:,0].cpu().numpy()

            # zl1, zl2, jet =  mu+, mu-, jet
            momenta = X.reshape(X.shape[0], -1, 4)

            observable_dict["ptjet"][batch * batch_size : batch * batch_size + X.shape[0]] = get_pt(momenta[:,2,:]).cpu().numpy()

            ct1, ct2 = cosmujet(momenta)
            observable_dict["cosmupjet"][batch * batch_size : batch * batch_size + X.shape[0]] = ct1.cpu().numpy()

    test_loss /= num_batches

    print(f"Testing Error: \n Avg (per batch) test loss: {test_loss:>8f}\n")

    with PdfPages(f"{model_run_dir}/test_histograms.pdf") as pdf:
        d = pdf.infodict()
        d['Title']        = f"Test results for model {model_name}"
        d['Author']       = 'You'
        d['Subject']      = 'Some comparison plots'
        d['Keywords']     = 'Machine Learning POWHEG POWHEGBOX POWHEG-BOX-RES'
        d['CreationDate'] = datetime.today()
        d['ModDate']      = datetime.today()

        fig, _ = print_integration_statistics(observable_dict)
        pdf.savefig(fig)
        plt.close(fig)

        # Invariant pT of the jet
        fig, _ = comparison_plots(observable_dict, "ptjet")
        pdf.savefig(fig)
        plt.close(fig)

        # Cos(theta*) of the mu+ and jet in Z+jet CM frame
        fig, _ = comparison_plots(observable_dict, "cosmupjet", log_scale=False, nbins=30)
        pdf.savefig(fig)
        plt.close(fig)

        fig, _ = r_plot(observable_dict["rL_pred"], observable_dict["rL_true"], observable_dict["weights_y"])
        pdf.savefig(fig)
        plt.close(fig)

    return test_loss

# %% Run the test

if __name__ == "__main__":
    start_time = time.time()
    test_loss_fn = torch.nn.MSELoss()

    if not arg.use_zjet:
        test_loss = test_model_ZZ(model, model_dir, arg.histogram_dir, test_dataloader, test_loss_fn, device, split_ratios[2] * arg.n_generated_events)
    else:
        test_loss = test_model_Zjet(model, model_dir, arg.histogram_dir, test_dataloader, test_loss_fn, device)

    end_time = time.time()
    print(f"Testing completed in {end_time - start_time:.2f} seconds.")