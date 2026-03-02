# %% Imports

import os
import time
import torch
import copy
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_events_utils.transforms import januar2026_input_choice
from ml_events_utils import MLEventsDataset, get_statistics_from_dataset, scale_target, boost_into_four_lepton_cm_frame, log_target_transform, exp_target_transform
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils import train_loop, valid_loop
from ml_events_utils import ZJetDataset
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
from ml_events_utils import log_file, setup_file_logger
from polarisation_test import do_test_run
from plot_training_history import plot_training_history


# if __name__ == "__main__":
log_file = "output.log"
mllogger = setup_file_logger(log_file, level="DEBUG", console=False, force=True)

mllogger.info(f"numpy:  {np.__version__}")
mllogger.info(f"pandas: {pd.__version__}")
mllogger.info(f"torch:  {torch.__version__}")

# %%
parser = argparse.ArgumentParser(
    description='Train a neural network for polarisation tagging.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("mlfiles", nargs='*',     type=Path,  action="store", help=".ml files to be used for training. Not required when using --replot.")
parser.add_argument("-m", "--model",          type=str,   action="store", default="FFNN_paper_BatchNorm", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
parser.add_argument("-o", "--optimizer",      type=str,   action="store", default="paper", help="Optimizer to use. Options: SGD, Adam, RMSprop, paper, paper_momentum.")
parser.add_argument("-g", "--gpu",            type=int,   action="store", default=-1,      help="Specify manually which of the available gpus is supposed to be used.")
parser.add_argument("-e", "--epochs",         type=int,   action="store", default=1000,    help="Number of training epochs.")
parser.add_argument("-b", "--batch_size",     type=int,   action="store", default=512,     help="Batch size for training.")
parser.add_argument("-l", "--learning_rate",  type=float, action="store", default=1e-3,    help="Learning rate for the optimizer.")
parser.add_argument("-p", "--patience",       type=int,   action="store", default=25,      help="Early stopping patience.")
parser.add_argument("-s", "--seed",           type=int,   action="store", default=42,      help="Random seed for reproducibility.")
parser.add_argument("-n", "--nworkers",       type=int,   action="store", default=0,       help="Number of workers for DataLoader.")
parser.add_argument("-t", "--test_mode",      dest="test_mode",    action="store_true",    help="Run in test mode (only one data point to test implementation of the model).")
parser.add_argument("--no-cache-events",      dest="cache_events", action="store_false",   help="Disable caching of events in the dataset (defaul: Cache the events.).")
parser.add_argument("--outputdir",            type=Path,  action='store', default=None,    help='Specify name of output directory.')
parser.add_argument("--replot",               dest="replot_only",  action="store_true",    help="Only regenerate the training history plot from existing CSV files. The model and potentially the output directory need to be specified.")
parser.add_argument("--useZjet",              dest="use_zjet",     action="store_true",    help="Use Z+jet dataset instead of default.")
parser.add_argument("--standardise",          dest="standardise",  action="store_true",    help="Enable standardisation of features over the whole dataset (default).")
parser.add_argument("--input_choice",         type=str,   action="store", default=None,    help="Choice of input features. Options: Momenta, jan2026.")
parser.add_argument("--n_generated_events",   type=lambda x: int(float(x)),       action="store", default=int(1e7), help="Number of generated events for comparison (1e7 for LO and LOwS and 5e6 for NLO).")
parser.add_argument("--dont_test",            dest="do_test",      action="store_false",   help="Run the test script after training with the best model weights found during training.")
parser.add_argument("--penalties", nargs='*', type=str,   action="store", default=[],      help="Specify which penalty terms to include in the loss function. Options: cross_section, ZdecayAngles.")
parser.add_argument("--polarisation",         type=str,   action="store", default="LL",    help="Specify which polarisation to train on (Only relevant for ZZ). Options: LL, LT, TL, TT, UL, LU.")
parser.add_argument("--showered",             dest="showered",     action="store_true",    help="This run used showered events instead of parton level events (default: use parton level events). Important for plotting.")

# Create a mutually exclusive group for specifying the reference frame
frame_group = parser.add_mutually_exclusive_group()
frame_group.add_argument("--labframe",       dest="labframe", default=True, action="store_true",    help="Use lab frame instead of partonic CMS.")
frame_group.add_argument("--cmframe",        dest="labframe", default=True, action="store_false",   help="Use partonic CMS instead of lab frame.")

arg = parser.parse_args()

mllogger.info("Arguments:")
for attr, value in vars(arg).items():
    mllogger.info(f"  {attr}: {value}")


start_time = time.time()


# %%
# from torch import nn
# from torchsummary import summary
# from torch.utils.data import DataLoader
# %% Specify the computation device (cpu or gpu).
# In torch/pytorch data and models need to be moved in the specific processing unit
# this code snippet allows to set the variable "device" according to available resource (cpu or cuda gpu)

if torch.cuda.is_available():
  mllogger.info(f"Number of devices: {torch.cuda.device_count()}")
  mllogger.info(str(torch.cuda.get_device_name(0)))

if torch.cuda.is_available():
    if arg.gpu >= 0:
        device = f"cuda:{arg.gpu}"
    else:
        device = "cuda"
else:
    device = "cpu"
mllogger.info(f"Computation device: {device}")

# Set CUDA device globally
if torch.cuda.is_available():
    if arg.gpu >= 0:
        torch.cuda.set_device(arg.gpu)
        mllogger.info(f"Set CUDA device to: {arg.gpu}")
    else:
        torch.cuda.set_device(0)
        mllogger.info(f"Set CUDA device to: 0")

# Validate arguments
if not arg.replot_only and len(arg.mlfiles) == 0:
    parser.error("mlfiles are required when not using --replot")

# Set fixed random number seed
seed = arg.seed
torch.manual_seed(seed)
np.random.seed(seed)

# %% Replot mode - load existing data and regenerate plot
if arg.replot_only:
    mllogger.info("Running in replot mode - loading existing training history...")

    # Determine model directory
    if arg.outputdir is not None:
        model_dir = arg.outputdir
        if not model_dir.exists():
            mllogger.error(f"Error: Directory {model_dir} does not exist!")
            sys.exit(1)
    else:
        # Try to infer from model name
        model_name = arg.model
        model_dir = Path().cwd()

    # Generate plot using the plotting function
    try:
        plot_training_history(model_dir, arg.model, use_log_scale=True)
        mllogger.info("Replot completed!")
    except FileNotFoundError as e:
        mllogger.error(f"Error: {e}")
        sys.exit(1)

    sys.exit(0)

# %% Data Handling

# Example usage for large files:
# files = Path("event_files/pwgevents-*.ml")
files = arg.mlfiles

mllogger.info(f"Cache events: {arg.cache_events}")

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

    dataset = MLEventsDataset(files,
                            labels = [f"{arg.polarisation}/UU", "UU"],
                            transform=trafo,
                            target_transform=log_target_transform,  # Apply log transform to reduce outlier impact
                            inv_target_transform=exp_target_transform,  # Inverse transform to revert log transformation
                            cache_events=arg.cache_events,  # Caching enabled
                            standardise=False)  # Specify wether standardisation over the whole dataset is enabled (this changes the dataset).
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
                          standardise=False)  # Specify wether standardisation over the whole dataset is enabled (this changes the dataset).
mllogger.info(f"Dataset info: {dataset.get_file_info()}")

# %% Hyperparameters

learning_rate = arg.learning_rate  # 1e-2
batch_size    = arg.batch_size     # 512
epochs        = arg.epochs         # 1000
n_workers     = arg.nworkers       # Use multiple (default 4) workers for DataLoader

# %% Split the dataset into training, validation and test sets

generator = torch.Generator().manual_seed(seed)

split_ratios = [0.6, 0.2, 0.2]  # 60% train, 20% validation, 20% test
# split_ratios = [0.02, 0.02, 0.96]  # 2% train, 2% validation, 96% test
# split_ratios = [0.005, 0.005, 0.99]  # 0.5% train, 0.5% validation, 99% test
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, split_ratios, generator=generator)

mllogger.info(f"Train dataset size:      {len(train_dataset)}")
mllogger.info(f"Validation dataset size: {len(val_dataset)}")
mllogger.info(f"Test dataset size:       {len(test_dataset)}")


if arg.standardise:
    overall_mean, overall_stddev = get_statistics_from_dataset(train_dataset)
    stat_norm = {
        "mean": overall_mean,
        "stddev": overall_stddev
    }
    mllogger.info(f"Feature means over training set (verification):\n{overall_mean}")
    mllogger.info(f"Feature stddevs over training set (verification):\n{overall_stddev}")


# Create DataLoader with multiple workers for better performance

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=True,
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True         # Faster GPU transfer
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=False,          # No need to shuffle validation data
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True         # Faster GPU transfer
)

mllogger.info(f"\nDataLoader created with batch_size={batch_size}, num_workers={n_workers}")

# Test iteration (only first batch to avoid long output)
for batch_idx, (batch_features, batch_labels) in enumerate(train_dataloader):
    mllogger.info(f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")
    input_dim = batch_features.shape[1]
    mllogger.info(f"{input_dim = }")
    break  # Only show first batch

# %% Test standardisation statistics
test_standardisation = False
if test_standardisation:
    fulldataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    features, labels = next(iter(fulldataloader))
    features_overall_mean   = features.mean(dim=0)
    mllogger.info(f"\nFeature means over training set:\n{features_overall_mean}")
    features_overall_stddev = features.std(dim=0)
    mllogger.info(f"\nFeature stddevs over training set:\n{features_overall_stddev}")

    # TODO: Calculate correct mean by multiplying for ZZ with UU xsec before averaging.
    xsec_estimate  = labels.sum(dim=0)
    xsec_estimate /= (split_ratios[0] * arg.n_generated_events)
    mllogger.info(f"\nxSec estimate over training set:\n{xsec_estimate}")

    del fulldataloader

    sys.exit(0)

# %% Specify the loss function

# Mean Squared Error loss for regression tasks
loss_fn = nn.MSELoss()

# loss_fn = nn.SmoothL1Loss()

# %% Set the model and choose an optimizer.
input_dim = dataset.input_shape[0]
model_name = arg.model
if arg.standardise:
    model = model_dict[model_name](input_dim=input_dim, stat_norm=stat_norm)
else:
    model = model_dict[model_name](input_dim=input_dim)

# TODO: Change the class name to model_name.


# if torch.cuda.device_count() > 1:
#   mllogger.info("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

if torch.cuda.is_available():
#   summary(model.cuda(), input_size=(1,input_dim))
    model_summary = summary(model.cuda(), input_size=(input_dim,))
else:
#   summary(model, input_size=(1,input_dim))
    model_summary = summary(model, input_size=(input_dim,))

mllogger.info(model_summary)

model.to(device)
# mllogger.info(f"Model {model_name} is on GPU: {next(model.parameters()).is_cuda}")
mllogger.info(f"Model {model_name} device: {next(model.parameters()).device}")

# Create directory for this model's outputs
if arg.outputdir is not None:
    model_dir = arg.outputdir
    model_dir.mkdir(exist_ok=True)
else:
    model_dir = Path().cwd()

optimizers = {
    "SGD":     torch.optim.SGD(    model.parameters(), lr=learning_rate),
    "Adam":    torch.optim.Adam(   model.parameters(), lr=learning_rate),
    "RMSprop": torch.optim.RMSprop(model.parameters(), lr=learning_rate),
    "paper":   torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.0),
    "paper_momentum":   torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.9)
}

# Initialize the optimizer
optimizer = optimizers[arg.optimizer]
mllogger.info(f"Using optimizer:\n{optimizer}")


# %% Test implementation on one batch before to train
xb, yb = next(iter(train_dataloader))

# test_batch_size = 10
# xb = torch.randn(test_batch_size, input_dim)
# yb = torch.randn(test_batch_size)

# Put tensors on device
xb = xb.type(torch.float).to(device)
yb = yb.type(torch.float).to(device)

mllogger.info(f"{xb.shape = }")
mllogger.info(f"{yb.shape = }")

# Prediction
pred = model(xb)
mllogger.info(f'output shape: {pred.shape}')
mllogger.info(f"{pred.squeeze().shape = }")


# Loss and metric
loss = loss_fn(pred, yb[:,0].unsqueeze(-1))
# loss = loss_fn(pred, torch.unsqueeze(yb,1))  # Bring yb to shape (batch_size, 1) to match pred shape.
# metric = binary_accuracy(pred, torch.unsqueeze(yb,1))

mllogger.info(f'loss: {loss.item()}')
# mllogger.info('metric: ', metric.item())


if arg.test_mode:
    mllogger.info("Exiting script now after testing implementation of the model on one point.")
    sys.exit(0)


# %% Training loop

# Save the seed used for this training
with open(model_dir / "training_seed.txt", 'w') as f:
    f.write(f"{seed}\n")

hist_loss     = []
hist_val_loss = []
hist_lr       = []

# Early stopping parameters
best_val_loss = float('inf')
patience = arg.patience
patience_counter = 0
best_model_state = None

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3,
)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Scheduler for multi step decay lr schedule
# Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.1)

# Initialize CSV files for real-time saving
train_loss_file = model_dir / f"{model_name}_train_loss.csv"
val_loss_file   = model_dir / f"{model_name}_val_loss.csv"
lr_file         = model_dir / f"{model_name}_learning_rates.csv"

# Write headers (optional, just the first value will be written)
with open(train_loss_file, 'w') as f:
    f.write("epoch,train_loss\n")
with open(val_loss_file, 'w') as f:
    f.write("epoch,val_loss\n")
with open(lr_file, 'w') as f:
    f.write("epoch,learning_rate\n")

mllogger.info(f"Starting training for {epochs} epochs...")
mllogger.info(f"Early stopping patience: {patience}")

penalties = {penalty: True for penalty in arg.penalties}

# Ensure that Z decay angle penalty is disabled when using Z+jet dataset or the januar2026 input choice, as the relevant features are not included in these cases.
if arg.use_zjet or (arg.input_choice in ["jan2026",]):
    penalties["ZdecayAngles"] = False

for epoch in range(epochs):
    epoch_start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']

    mllogger.info(f"Epoch {epoch + 1}/{epochs}")
    mllogger.info(f"Learning Rate: {current_lr:.2e}")
    mllogger.info("-" * 50)

    # Training phase
    train_loss = train_loop(epoch, train_dataloader, model, loss_fn, optimizer, device, print_freq = 2500, penalties=penalties)
    hist_loss.append(train_loss)
    hist_lr.append(current_lr)

    # Validation phase
    valid_loss = valid_loop(val_dataloader, model, loss_fn, device)
    hist_val_loss.append(valid_loss)

    # Save current epoch results to CSV files immediately
    with open(train_loss_file, 'a') as f:
        f.write(f"{epoch+1},{train_loss:.10e}\n")
    with open(val_loss_file, 'a') as f:
        f.write(f"{epoch+1},{valid_loss:.10e}\n")
    with open(lr_file, 'a') as f:
        f.write(f"{epoch+1},{current_lr:.10e}\n")

    # Learning rate scheduling
    # scheduler.step()
    scheduler.step(valid_loss)

    # Early stopping check
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        patience_counter = 0
        # Save best model state
        best_model_state = copy.deepcopy(model.state_dict())
        mllogger.info(f"✓ New best validation loss: {best_val_loss:.6f}")
    else:
        patience_counter += 1
        mllogger.info(f"No improvement. Patience: {patience_counter}/{patience}")

    epoch_time = time.time() - epoch_start_time
    mllogger.info(f"Epoch time: {epoch_time:.2f} seconds")
    mllogger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f}")

    # Early stopping
    if patience_counter >= patience:
        mllogger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
        mllogger.info(f"Best validation loss: {best_val_loss:.6f}")
        break

# Save final model
model_filename = model_dir / f"{model_name}_model_weights_final.pt"
torch.save(model.state_dict(), model_filename)

# Load best model weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    mllogger.info(f"\nLoaded best model with validation loss: {best_val_loss:.6f}")


hist_loss     = np.array(hist_loss)
hist_val_loss = np.array(hist_val_loss)
hist_lr       = np.array(hist_lr)


# Save best model separately
if best_model_state is not None:
    best_model_filename = model_dir / f"{model_name}_model_weights_best.pt"
    torch.save(best_model_state, best_model_filename)
    mllogger.info(f"Best model saved as: {best_model_filename}")


mllogger.info(f"Final model saved as: {model_filename}")
mllogger.info("Training completed!")

# Print training summary
mllogger.info(f"Training Summary:")
mllogger.info(f"Total epochs: {len(hist_loss)}")
mllogger.info(f"Final train loss: {hist_loss[-1]:.6f}")
mllogger.info(f"Final validation loss: {hist_val_loss[-1]:.6f}")
mllogger.info(f"Best validation loss: {best_val_loss:.6f}")

# %% Plot loss

mllogger.info(f"Plotting training history, using best model weights: {best_model_state is not None}")

# Generate plot using the plotting function
try:
    plot_training_history(model_dir, model_name, use_log_scale=True)
except Exception as e:
    mllogger.warning(f"Could not generate plot: {e}")

# %% Check network on random event:
# ```
# <event>
#   1.445418701E+01 -2.611547450E+00  8.079240742E+01  8.211672667E+01
#   4.121475591E+00 -3.706903553E+01 -1.028783725E+01  3.869030307E+01
#   2.206750721E+01 -9.725720987E+00  1.263696046E+01  2.722604071E+01
#  -4.064316981E+01  4.940630397E+01 -3.490849930E+01  7.287971904E+01
# <rwgt>
# <weight id='UU'> 0.248160008E-01 </weight>
# <weight id='LL'> 0.885049987E-03 </weight>
# <weight id='LT'> 0.178980001E-02 </weight>
# <weight id='TL'> 0.346059998E-03 </weight>
# <weight id='TT'> 0.179340001E-01 </weight>
# </rwgt>
# </event>
# ```

# model = FFNN_BatchNorm(input_dim=input_dim, width= 100) # we do not specify ``weights``, i.e. create untrained model
# model.load_state_dict(torch.load(f"{model_name}_model_weights_best.pt", weights_only=True))

# model.eval()

# if not arg.use_zjet and not (arg.input_choice in ["jan2026",]):
#     test_tensor = torch.tensor([ 1.445418701E+01, -2.611547450E+00,  8.079240742E+01,  8.211672667E+01,
#                                 4.121475591E+00, -3.706903553E+01, -1.028783725E+01,  3.869030307E+01,
#                                 2.206750721E+01, -9.725720987E+00,  1.263696046E+01,  2.722604071E+01,
#                                 -4.064316981E+01,  4.940630397E+01, -3.490849930E+01,  7.287971904E+01])

#     res = model(test_tensor.unsqueeze(0).to(device))
#     mllogger.info(f"res = {res.item():.10e}")
#     mllogger.info(f"Expected LL/ UU weight: {0.885049987E-03 / 0.248160008E-01:.10e}")
# elif not (arg.input_choice in ["jan2026",]):
#     test_tensor = torch.tensor([-12.130391188000001,  34.443724807000002, 262.44532550000002, 264.97370709000000,
#                                  59.635322049999999, -22.605515205000000, 283.59799611000000, 290.68058819999999,
#                                 -47.504930862000002, -11.838209601000001, 171.64348498999999, 178.48906858000001])

#     res = model(test_tensor.unsqueeze(0).to(device))
#     mllogger.info(f"res = {res.item():.10e}")
#     mllogger.info(f"Expected LL/ UU weight: {0.91735652950215585:.10e}")

end_time = time.time()
elapsed_time = end_time - start_time
mllogger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if arg.do_test:
    do_test_run(device, arg.use_zjet, model, model_name, model_dir, files[0].parent, files, seed, test_dataset, split_ratios, arg.polarisation, batch_size=arg.batch_size, n_workers=arg.nworkers, n_generated_events=arg.n_generated_events, input_choice=arg.input_choice, showered=arg.showered)
