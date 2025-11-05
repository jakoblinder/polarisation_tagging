# %% Imports

import os
import time
import torch
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_events_utils import MLEventsDataset, scale_target, train_loop, valid_loop
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
import argparse

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('torch', torch.__version__)



# %%
parser = argparse.ArgumentParser(
    description='Train a neural network for polarisation tagging.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("mlfiles", nargs='+',    type=Path,  action="store", help=".ml files to be used for training.")
parser.add_argument("-m", "--model",         type=str,   action="store", default="FFNN_BatchNorm_no_output", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
parser.add_argument("-o", "--optimizer",     type=str,   action="store", default="paper", help="Optimizer to use. Options: SGD, Adam, RMSprop, paper.")
parser.add_argument("-e", "--epochs",        type=int,   action="store", default=1000,    help="Number of training epochs.")
parser.add_argument("-b", "--batch_size",    type=int,   action="store", default=128,     help="Batch size for training.")
parser.add_argument("-l", "--learning_rate", type=float, action="store", default=1e-2,    help="Learning rate for the optimizer.")
parser.add_argument("-p", "--patience",      type=int,   action="store", default=25,      help="Early stopping patience.")
parser.add_argument("-s", "--seed",          type=int,   action="store", default=42,      help="Random seed for reproducibility.")
parser.add_argument("-n", "--nworkers",      type=int,   action="store", default=4,       help="Number of workers for DataLoader.")
parser.add_argument("--no-cache-events",     dest="cache_events", action="store_false",   help="Disable caching of events in the dataset.")
parser.add_argument("--outputdir",           type=Path,  action='store', default=None, help='Specify name of output directory.')

arg = parser.parse_args()

# Set fixed random number seed
seed = arg.seed
torch.manual_seed(seed)


# %% Data Handling

# Example usage for large files:
# files = Path("event_files/pwgevents-*.ml")
files = arg.mlfiles

print(f"Cache events: {arg.cache_events}")

dataset = MLEventsDataset(files,
                          labels = ["LL",],
                        #   transform=None,
                          target_transform=scale_target,  # Scale target by 1000
                          cache_events=arg.cache_events)  # Caching enabled
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

# Create DataLoader with multiple workers for better performance

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=True,
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True  # Faster GPU transfer
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=True,
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True  # Faster GPU transfer
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # Larger batch size for efficiency
    shuffle=True,
    num_workers=n_workers,  # Use multiple workers for large files
    pin_memory=True  # Faster GPU transfer
)


print(f"\nDataLoader created with batch_size={batch_size}, num_workers={n_workers}")

# Test iteration (only first batch to avoid long output)
for batch_idx, (batch_features, batch_labels) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")
    input_dim = batch_features.shape[1]
    print(f"{input_dim = }")
    break  # Only show first batch

# %% Specify the loss function

# Mean Squared Error loss for regression tasks
loss_fn = nn.MSELoss()

# loss_fn = nn.SmoothL1Loss()

# %% Specify the computation device (cpu or gpu).
# In torch/pytorch data and models need to be moved in the specific processing unit
# this code snippet allows to set the variable "device" according to available resource (cpu or cuda gpu)

if torch.cuda.is_available():
  print('Number of devices: ', torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# %% Set the model and choose an optimizer.

model = model_dict[arg.model](input_dim=input_dim)
model_name = model.__class__.__name__

model.to(device)
print(f"Model {model_name} is on GPU: {next(model.parameters()).is_cuda}")


# Create directory for this model's outputs
if arg.outputdir is not None:
    model_dir = arg.outputdir
else:
    model_dir = Path(model_name)
model_dir.mkdir(exist_ok=True)
print(f"Created directory: {model_dir}")

optimizers = {
    "SGD":     torch.optim.SGD( model.parameters(), lr=learning_rate),
    "Adam":    torch.optim.Adam(model.parameters(), lr=learning_rate),
    "RMSprop": torch.optim.RMSprop(model.parameters(), lr=learning_rate),
    "paper":   torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, weight_decay=0.0, momentum=0.0)
}
# Initialize the optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Alternative
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# RMSprop optimizer, set to the parameters Giovanni and Mathieu used in their studies
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, weight_decay=0.0, momentum=0.0)

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



# %% Training loop

if torch.cuda.is_available():
#   summary(model.cuda(), input_size=(1,input_dim))
  summary(model.cuda(), input_size=(input_dim,))
else:
#   summary(model, input_size=(1,input_dim))
  summary(model, input_size=(input_dim,))



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
    valid_loss = valid_loop(val_dataloader,   model, loss_fn, device)
    hist_val_loss.append(valid_loss)

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

np.savetxt(model_dir / f"{model_name}_train_loss.csv",     hist_loss,     delimiter=',')
np.savetxt(model_dir / f"{model_name}_val_loss.csv",       hist_val_loss, delimiter=',')
np.savetxt(model_dir / f"{model_name}_learning_rates.csv", hist_lr,       delimiter=',')

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
plt.figure(figsize=(10, 7))
plt.title(f"Training History for {model_name}")

plt.plot(range(1,len(hist_loss)+1),     np.array(hist_loss),     label="Avg training loss")
plt.plot(range(1,len(hist_val_loss)+1), np.array(hist_val_loss), label="Avg validation loss")
plt.plot(range(1,len(hist_lr)+1),       np.array(hist_lr)*1000,  label="Learning rate x 1000")

plt.ylim(ymin=0)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

# Save plot to model directory
plt.savefig(model_dir / f"{model_name}_training_history.pdf", bbox_inches='tight')
# plt.show()

# %% Check network on random event:
# ```
# <event>
#   3.016570943E+01  4.510460277E+01 -1.552387738E+02  1.644489954E+02
#  -5.944532656E+01  1.420361466E+02 -3.064427199E+02  3.429506588E+02
#  -3.520454553E+01  5.968701995E+01 -2.189955679E+01  7.267386713E+01
#  -7.860515623E+01 -7.434789122E-01 -9.197268046E+01  1.209888313E+02
# <rwgt>
# <weight id='LL'> 0.360386982E-02 </weight>
# <weight id='LT'> 0.914721633E-03 </weight>
# <weight id='TL'> 0.168988248E-02 </weight>
# <weight id='TT'> 0.160581823E-01 </weight>
# </rwgt>
# </event>
# ```

# model = FFNN_BatchNorm(input_dim=input_dim, width= 100) # we do not specify ``weights``, i.e. create untrained model
# model.load_state_dict(torch.load(f"{model_name}_model_weights_best.pt", weights_only=True))

model.eval()

test_tensor = torch.tensor([ 3.016570943E+01,  4.510460277E+01, -1.552387738E+02, 1.644489954E+02,
                            -5.944532656E+01,  1.420361466E+02, -3.064427199E+02, 3.429506588E+02,
                            -3.520454553E+01,  5.968701995E+01, -2.189955679E+01, 7.267386713E+01,
                            -7.860515623E+01, -7.434789122E-01, -9.197268046E+01, 1.209888313E+02])

res = model(test_tensor.unsqueeze(0).to(device))
print(f"res = {res.item()/ 1000:.2e}")
print(f"Expected LL weight: {0.360386982E-02:.2e}")

# %% Test the trained model

# Let's test it on cpu
# model.to(torch.device("cpu"))

# X_test_pt = X_test_pt.type(torch.float).to(torch.device("cpu"))
# res = model(X_test_pt)

