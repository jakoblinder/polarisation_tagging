# %% Imports

import os
import time
import torch
import copy
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_events_utils.transforms import januar2026_input_choice
from ml_events_utils import MLEventsDataset, get_statistics_from_dataset, scale_target, boost_into_four_lepton_cm_frame, log_target_transform
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils import train_loop, valid_loop
from ml_events_utils import ZJetDataset
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
from plot_training_history import plot_training_history
import argparse

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('torch', torch.__version__)



# %%
parser = argparse.ArgumentParser(
    description='Train a neural network for polarisation tagging.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("mlfiles", nargs='*',    type=Path,  action="store", help=".ml files to be used for training. Not required when using --replot.")
parser.add_argument("-m", "--model",         type=str,   action="store", default="FFNN_paper_BatchNorm", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
parser.add_argument("-o", "--optimizer",     type=str,   action="store", default="paper", help="Optimizer to use. Options: SGD, Adam, RMSprop, paper, paper_momentum.")
parser.add_argument("-g", "--gpu",           type=int,   action="store", default=-1,      help="Specify manually which of the available gpus is supposed to be used.")
parser.add_argument("-e", "--epochs",        type=int,   action="store", default=1000,    help="Number of training epochs.")
parser.add_argument("-b", "--batch_size",    type=int,   action="store", default=512,     help="Batch size for training.")
parser.add_argument("-l", "--learning_rate", type=float, action="store", default=1e-2,    help="Learning rate for the optimizer.")
parser.add_argument("-p", "--patience",      type=int,   action="store", default=25,      help="Early stopping patience.")
parser.add_argument("-s", "--seed",          type=int,   action="store", default=42,      help="Random seed for reproducibility.")
parser.add_argument("-n", "--nworkers",      type=int,   action="store", default=0,       help="Number of workers for DataLoader.")
parser.add_argument("-t", "--test_mode",     dest="test_mode",    action="store_true",    help="Run in test mode (only one data point to test implementation of the model).")
parser.add_argument("--no-cache-events",     dest="cache_events", action="store_false",   help="Disable caching of events in the dataset (defaul: Cache the events.).")
parser.add_argument("--outputdir",           type=Path,  action='store', default=None,    help='Specify name of output directory.')
parser.add_argument("--replot",              dest="replot_only",  action="store_true",    help="Only regenerate the training history plot from existing CSV files. The model and potentially the output directory need to be specified.")
parser.add_argument("--useZjet",             dest="use_zjet",     action="store_true",    help="Use Z+jet dataset instead of default.")
parser.add_argument("--standardise",         dest="standardise",  action="store_true",    help="Enable standardisation of features over the whole dataset (default).")
parser.add_argument("--input_choice",        type=str,   action="store", default=None,    help="Choice of input features. Options: Momenta, jan2026.")

# Create a mutually exclusive group for specifying the reference frame
frame_group = parser.add_mutually_exclusive_group()
frame_group.add_argument("--labframe",       dest="labframe", default=True, action="store_true",    help="Use lab frame instead of partonic CMS.")
frame_group.add_argument("--cmframe",        dest="labframe", default=True, action="store_false",   help="Use partonic CMS instead of lab frame.")

arg = parser.parse_args()

print("Arguments:")
for attr, value in vars(arg).items():
    print(f"  {attr}: {value}")

# %%
# from torch import nn
# from torchsummary import summary
# from torch.utils.data import DataLoader
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

# Validate arguments
if not arg.replot_only and len(arg.mlfiles) == 0:
    parser.error("mlfiles are required when not using --replot")

# Set fixed random number seed
seed = arg.seed
torch.manual_seed(seed)
np.random.seed(seed)

# %% Replot mode - load existing data and regenerate plot
if arg.replot_only:
    print("Running in replot mode - loading existing training history...")

    # Determine model directory
    if arg.outputdir is not None:
        model_dir = arg.outputdir
        if not model_dir.exists():
            print(f"Error: Directory {model_dir} does not exist!")
            sys.exit(1)
    else:
        # Try to infer from model name
        model_name = arg.model
        model_dir = Path().cwd()

    # Generate plot using the plotting function
    try:
        plot_training_history(model_dir, arg.model, use_log_scale=True)
        print("Replot completed!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sys.exit(0)

# %% Data Handling

# Example usage for large files:
# files = Path("event_files/pwgevents-*.ml")
files = arg.mlfiles

print(f"Cache events: {arg.cache_events}")

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
                            labels = ["LL/UU",],
                            transform=trafo,
                            #   target_transform=log_target_transform,  # Apply log transform to reduce outlier impact
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
print(f"Dataset info: {dataset.get_file_info()}")

# %% Hyperparameters

learning_rate = arg.learning_rate  # 1e-2
batch_size    = arg.batch_size     # 128
epochs        = arg.epochs         # 1000
n_workers     = arg.nworkers       # Use multiple (default 4) workers for DataLoader

# %% Split the dataset into training, validation and test sets

generator = torch.Generator().manual_seed(seed)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

print(f"Train dataset size:      {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size:       {len(test_dataset)}")


if arg.standardise:
    overall_mean, overall_stddev = get_statistics_from_dataset(train_dataset)
    stat_norm = {
        "mean": overall_mean,
        "stddev": overall_stddev
    }
    print(f"\nFeature means over training set (verification):\n{overall_mean}")
    print(f"\nFeature stddevs over training set (verification):\n{overall_stddev}")


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

print(f"\nDataLoader created with batch_size={batch_size}, num_workers={n_workers}")

# Test iteration (only first batch to avoid long output)
for batch_idx, (batch_features, batch_labels) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")
    input_dim = batch_features.shape[1]
    print(f"{input_dim = }")
    break  # Only show first batch

# %% Test standardisation statistics
test_standardisation = False
if test_standardisation:
    fulldataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    features, _ = next(iter(fulldataloader))
    overall_mean   = features.mean(dim=0)
    overall_stddev = features.std(dim=0)
    del fulldataloader

    print(f"\nFeature means over training set:\n{overall_mean}")
    print(f"\nFeature stddevs over training set:\n{overall_stddev}")

    sys.exit(0)

# %% Specify the loss function

# Mean Squared Error loss for regression tasks
loss_fn = nn.MSELoss()

# loss_fn = nn.SmoothL1Loss()

# %% Set the model and choose an optimizer.
model_name = arg.model
if arg.standardise:
    model = model_dict[model_name](input_dim=input_dim, stat_norm=stat_norm)
else:
    model = model_dict[model_name](input_dim=input_dim)

# TODO: Change the class name to model_name.


# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

if torch.cuda.is_available():
#   summary(model.cuda(), input_size=(1,input_dim))
  summary(model.cuda(), input_size=(input_dim,))
else:
#   summary(model, input_size=(1,input_dim))
  summary(model, input_size=(input_dim,))

model.to(device)
# print(f"Model {model_name} is on GPU: {next(model.parameters()).is_cuda}")
print(f"Model {model_name} device: {next(model.parameters()).device}")

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
    "paper":   torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.0),
    "paper_momentum":   torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.9)
}

# Initialize the optimizer
optimizer = optimizers[arg.optimizer]
print(f"Using optimizer: {optimizer}")


# %% Test implementation on one batch before to train
xb, yb = next(iter(train_dataloader))

# test_batch_size = 10
# xb = torch.randn(test_batch_size, input_dim)
# yb = torch.randn(test_batch_size)

# Put tensors on device
xb = xb.type(torch.float).to(device)
yb = yb.type(torch.float).to(device)

print(f"{xb.shape = }")
print(f"{yb.shape = }")

# Prediction
pred = model(xb)
print('output shape: ', pred.shape)
print(f"{pred.squeeze().shape = }")


# Loss and metric
loss = loss_fn(pred, yb)
# loss = loss_fn(pred, torch.unsqueeze(yb,1))  # Bring yb to shape (batch_size, 1) to match pred shape.
# metric = binary_accuracy(pred, torch.unsqueeze(yb,1))

print('loss: ', loss.item())
# print('metric: ', metric.item())


if arg.test_mode:
    print("Exiting script now after testing implementation of the model on one point.")
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

print(f"Starting training for {epochs} epochs...")
print(f"Early stopping patience: {patience}")

for epoch in range(epochs):
    epoch_start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Learning Rate: {current_lr:.2e}")
    print("-" * 50)

    # Training phase
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device, print_freq = 2500)
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
    scheduler.step(valid_loss)

    # Early stopping check
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        patience_counter = 0
        # Save best model state
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"✓ New best validation loss: {best_val_loss:.6f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch time: {epoch_time:.2f} seconds")
    print(f"Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f}")

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        print(f"Best validation loss: {best_val_loss:.6f}")
        break

# Save final model
model_filename = model_dir / f"{model_name}_model_weights_final.pt"
torch.save(model.state_dict(), model_filename)

# Load best model weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model with validation loss: {best_val_loss:.6f}")


hist_loss     = np.array(hist_loss)
hist_val_loss = np.array(hist_val_loss)
hist_lr       = np.array(hist_lr)

# np.savetxt(model_dir / f"{model_name}_train_loss.csv",     hist_loss,     delimiter=',')
# np.savetxt(model_dir / f"{model_name}_val_loss.csv",       hist_val_loss, delimiter=',')
# np.savetxt(model_dir / f"{model_name}_learning_rates.csv", hist_lr,       delimiter=',')

# Save best model separately
if best_model_state is not None:
    best_model_filename = model_dir / f"{model_name}_model_weights_best.pt"
    torch.save(best_model_state, best_model_filename)
    print(f"Best model saved as: {best_model_filename}")


print(f"Final model saved as: {model_filename}")
print("Training completed!")

# Print training summary
print(f"\nTraining Summary:")
print(f"Total epochs: {len(hist_loss)}")
print(f"Final train loss: {hist_loss[-1]:.6f}")
print(f"Final validation loss: {hist_val_loss[-1]:.6f}")
print(f"Best validation loss: {best_val_loss:.6f}")

# %% Plot loss

print(f"Plotting training history, using best model weights: {best_model_state is not None}")

# Generate plot using the plotting function
try:
    plot_training_history(model_dir, model_name, use_log_scale=True)
except Exception as e:
    print(f"Warning: Could not generate plot: {e}")

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

model.eval()

if not arg.use_zjet and not (arg.input_choice in ["jan2026",]):
    test_tensor = torch.tensor([ 1.445418701E+01, -2.611547450E+00,  8.079240742E+01,  8.211672667E+01,
                                4.121475591E+00, -3.706903553E+01, -1.028783725E+01,  3.869030307E+01,
                                2.206750721E+01, -9.725720987E+00,  1.263696046E+01,  2.722604071E+01,
                                -4.064316981E+01,  4.940630397E+01, -3.490849930E+01,  7.287971904E+01])

    res = model(test_tensor.unsqueeze(0).to(device))
    print(f"res = {res.item():.10e}")
    print(f"Expected LL/ UU weight: {0.885049987E-03 / 0.248160008E-01:.10e}")
elif not (arg.input_choice in ["jan2026",]):
    test_tensor = torch.tensor([-12.130391188000001,  34.443724807000002, 262.44532550000002, 264.97370709000000,
                                 59.635322049999999, -22.605515205000000, 283.59799611000000, 290.68058819999999,
                                -47.504930862000002, -11.838209601000001, 171.64348498999999, 178.48906858000001])

    res = model(test_tensor.unsqueeze(0).to(device))
    print(f"res = {res.item():.10e}")
    print(f"Expected LL/ UU weight: {0.91735652950215585:.10e}")

