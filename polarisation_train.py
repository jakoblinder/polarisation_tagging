# %% Imports

import os
import time
import torch
import copy
import sys
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import optuna
except ImportError:
    optuna = None

from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_events_utils.transforms import januar2026_input_choice
from ml_events_utils import MLEventsDataset, get_statistics_from_dataset, scale_target, boost_into_four_lepton_cm_frame, log_target_transform, exp_target_transform
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils import train_loop, valid_loop
from ml_events_utils import ZJetDataset
from ml_events_utils.models import *  # FFNN_BatchNorm, FFNN_BatchNorm_no_output, FFNN_paper
from ml_events_utils import log_file, setup_file_logger
from ml_events_utils import Settings
from polarisation_test import do_test_run
from plot_training_history import plot_training_history


def parse_args() -> argparse.Namespace:
    # %%
    parser = argparse.ArgumentParser(
        description="Train a neural network for polarisation tagging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mlfiles", nargs="*", type=Path, action="store", help=".ml files to be used for training. Not required when using --replot.")
    parser.add_argument("-m", "--model", type=str, action="store", default="FFNN_paper_BatchNorm", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
    parser.add_argument("-o", "--optimizer", type=str, action="store", default="paper", help="Optimizer to use. Options: SGD, Adam, AdamW, RMSprop, paper, paper_momentum.")
    parser.add_argument("-g", "--gpu", type=int, action="store", default=-1, help="Specify manually which of the available gpus is supposed to be used.")
    parser.add_argument("-e", "--epochs", type=int, action="store", default=1000, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, action="store", default=512, help="Batch size for training.")
    parser.add_argument("-l", "--learning_rate", type=float, action="store", default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("-p", "--patience", type=int, action="store", default=25, help="Early stopping patience.")
    parser.add_argument("-s", "--seed", type=int, action="store", default=42, help="Random seed for reproducibility.")
    parser.add_argument("-n", "--nworkers", type=int, action="store", default=0, help="Number of workers for DataLoader.")
    parser.add_argument("-t", "--test_mode", dest="test_mode", action="store_true", help="Run in test mode (only one data point to test implementation of the model).")
    parser.add_argument("--no-cache-events", dest="cache_events", action="store_false", help="Disable caching of events in the dataset (default: cache the events).")
    parser.add_argument("--outputdir", type=Path, action="store", default=None, help="Specify name of output directory.")
    parser.add_argument("--replot", dest="replot_only", action="store_true", help="Only regenerate the training history plot from existing CSV files. The model and potentially the output directory need to be specified.")
    parser.add_argument("--useZjet", dest="use_zjet", action="store_true", help="Use Z+jet dataset instead of default.")
    parser.add_argument("--standardise", dest="standardise", action="store_true", help="Enable standardisation of features over the whole dataset (default).")
    parser.add_argument("--input_choice", type=str, action="store", default=None, help="Choice of input features. Options: Momenta, jan2026.")
    parser.add_argument("--n_generated_events", type=lambda x: int(float(x)), action="store", default=int(1e7), help="Number of generated events for comparison (1e7 for LO and LOwS and 5e6 for NLO).")
    parser.add_argument("--dont_test", dest="do_test", action="store_false", help="Run the test script after training with the best model weights found during training.")
    parser.add_argument("--penalties", nargs="*", type=str, action="store", default=[], help="Specify which penalty terms to include in the loss function. Options: cross_section, ZdecayAngles.")
    parser.add_argument("--polarisation", type=str, action="store", default="LL", help="Specify which polarisation to train on (Only relevant for ZZ). Options: LL, LT, TL, TT, UL, LU.")
    parser.add_argument("--showered", dest="showered", action="store_true", help="This run used showered events instead of parton level events (default: use parton level events). Important for plotting.")

    # Create a mutually exclusive group for specifying the reference frame
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--labframe", dest="labframe", default=True, action="store_true", help="Use lab frame instead of partonic CMS.")
    frame_group.add_argument("--cmframe",  dest="labframe", default=True, action="store_false", help="Use partonic CMS instead of lab frame.")

    args = parser.parse_args()

    return args


def namespace_from_settings(run_settings: Settings) -> argparse.Namespace:
    return argparse.Namespace(**{key: parameter.value for key, parameter in run_settings.items()})


def prepare_run_settings(arg: argparse.Namespace) -> Settings:
    run_settings = Settings(argparse=arg)
    run_settings.set("split_ratios", [0.6, 0.2, 0.2])
    run_settings.set("test_standardisation", False)
    run_settings.set("count_negative_weights", False)
    return run_settings


def select_device(gpu: int, logger) -> str:
    # Specify the computation device (cpu or gpu).
    # In torch/pytorch data and models need to be moved in the specific processing unit
    # this code snippet allows to set the variable "device" according to available resource (cpu or cuda gpu)
    if torch.cuda.is_available():
        logger.info(f"Number of devices: {torch.cuda.device_count()}")
        logger.info(str(torch.cuda.get_device_name(0)))

    if torch.cuda.is_available():
        if gpu >= 0:
            device = f"cuda:{gpu}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Computation device: {device}")

    if torch.cuda.is_available():
        if gpu >= 0:
            torch.cuda.set_device(gpu)
            logger.info(f"Set CUDA device to: {gpu}")
        else:
            torch.cuda.set_device(0)
            logger.info("Set CUDA device to: 0")

    return device


def load_statistics_from_file(output_root_parent: Path) -> dict:
    """Load pre-computed dataset statistics from JSON file.

    Args:
        output_root_parent: Parent directory containing dataset_statistics.json
                           (typically {output_root} from hyperparameter scan)

    Returns:
        dict with 'mean' and 'stddev' numpy arrays, or None if file doesn't exist/is invalid
    """
    stats_file = output_root_parent / "dataset_statistics.json"

    if not stats_file.exists():
        return None

    try:
        with open(stats_file, "r") as f:
            stats_data = json.load(f)

        # Convert lists back to numpy arrays
        stat_norm = {
            "mean": torch.tensor(stats_data["mean"]),
            "stddev": torch.tensor(stats_data["stddev"]),
        }
        return stat_norm

    except Exception as e:
        print(f"Warning: Could not load statistics from {stats_file}: {e}")
        return None


def run_training(run_settings: Settings, logger, trial=None):
    arg = namespace_from_settings(run_settings)
    start_time = time.time()

    run_settings.log_to_logger(logger, header="Arguments:")

    if not arg.replot_only and len(arg.mlfiles) == 0:
        raise ValueError("mlfiles are required when not using --replot")

    seed = arg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # %% Replot mode - load existing data and regenerate plot
    if arg.replot_only:
        logger.info("Running in replot mode - loading existing training history...")

        if arg.outputdir is not None:
            model_dir = arg.outputdir
            if not model_dir.exists():
                raise FileNotFoundError(f"Directory {model_dir} does not exist")
        else:
            model_dir = Path().cwd()

        plot_training_history(model_dir, arg.model, use_log_scale=True)
        logger.info("Replot completed!")
        return {"mode": "replot", "model_dir": model_dir, "model_name": arg.model}

    # Data Handling

    # Example usage for large files:
    # files = Path("event_files/pwgevents-*.ml")
    files = arg.mlfiles
    logger.info(f"Cache events: {arg.cache_events}")

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

        dataset = MLEventsDataset(
            files,
            labels=[f"{arg.polarisation}/UU", "UU"],
            transform=trafo,
            # Apply log transform to reduce outlier impact
            target_transform=log_target_transform,
            # Inverse transform to revert log transformation
            inv_target_transform=exp_target_transform,
            cache_events=arg.cache_events,
            # Specify wether standardisation over the whole dataset is enabled (this changes the dataset).
            standardise=False,
        )
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

        dataset = ZJetDataset(
            files[0],
            transform=trafo,
            target_transform=None,
            # Maximum number of events to load (useful for testing). Max = 10^6.
            max_events=None,
            # Specify wether standardisation over the whole dataset is enabled (this changes the dataset).
            standardise=False,
        )

    logger.info(f"Dataset info: {dataset.get_file_info()}")

    # %% Hyperparameters
    learning_rate = arg.learning_rate
    batch_size    = arg.batch_size
    epochs        = arg.epochs
    n_workers     = arg.nworkers
    split_ratios  = run_settings.split_ratios.value

    # %% Split the dataset into training, validation and test sets
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, split_ratios, generator=generator
    )

    logger.info(f"Train dataset size:      {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size:       {len(test_dataset)}")

    stat_norm = None
    if arg.standardise:
        # Try to load cached statistics from parent directory (hyperparameter scan cache)
        output_root_parent = Path(arg.outputdir).parent
        cached_stats = load_statistics_from_file(output_root_parent)

        if cached_stats is not None:
            stat_norm = cached_stats
            logger.info("Loaded pre-computed dataset statistics from cache.")
            logger.info(f"Feature means over training set (from cache):\n{stat_norm['mean']}")
            logger.info(f"Feature stddevs over training set (from cache):\n{stat_norm['stddev']}")
        else:
            # Compute statistics for this trial
            overall_mean, overall_stddev = get_statistics_from_dataset(train_dataset)
            stat_norm = {"mean": overall_mean, "stddev": overall_stddev}
            logger.info("Computing dataset statistics for this trial (no cache found).")
            logger.info(f"Feature means over training set:\n{overall_mean}")
            logger.info(f"Feature stddevs over training set:\n{overall_stddev}")


    train_dataloader = DataLoader(
        train_dataset,
        # Larger batch size for efficiency
        batch_size=batch_size,
        shuffle=True,
        # Use multiple workers for large files
        num_workers=n_workers,
        # Faster GPU transfer
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        # Larger batch size for efficiency
        batch_size=batch_size,
        # No need to shuffle validation data
        shuffle=False,
        # Use multiple workers for large files
        num_workers=n_workers,
        # Faster GPU transfer
        pin_memory=True,
    )
    logger.info(f"\nDataLoader created with batch_size={batch_size}, num_workers={n_workers}")

    # Test iteration (only first batch to avoid long output)
    for batch_idx, (batch_features, batch_labels) in enumerate(train_dataloader):
        logger.info(
            f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}"
        )
        input_dim = batch_features.shape[1]
        logger.info(f"{input_dim = }")
        break

    device = select_device(arg.gpu, logger)

    # %% Test statistics
    if run_settings.test_standardisation.value:
        fulldataloader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        features, labels = next(iter(fulldataloader))
        features_overall_mean   = features.mean(dim=0)
        features_overall_stddev = features.std(dim=0)
        # TODO: Calculate correct mean by multiplying for ZZ with UU xsec before averaging.
        xsec_estimate  = labels.sum(dim=0)
        xsec_estimate /= (split_ratios[0] * arg.n_generated_events)

        logger.info(f"Feature means over training set:\n{features_overall_mean}")
        logger.info(f"Feature stddevs over training set:\n{features_overall_stddev}")
        logger.info(f"xSec estimate over training set:\n{xsec_estimate}")
        return {"mode": "test_standardisation"}

    if run_settings.count_negative_weights.value:
        fulldataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        features, labels = next(iter(fulldataloader))
        features, labels = features.to(device), labels.to(device)

        valid_fraction    = torch.isfinite(labels[..., 0]).type(torch.float).mean().item()
        neg_fraction      = ((labels[..., 0]) < 0.0).type(torch.float).mean().item()
        small_fraction    = ((0 < labels[..., 0]) & (labels[..., 0] < 1e-10)).type(torch.float).mean().item()
        neg_fraction_log1 = (labels[..., 0] < (1e-10 - 1.0)).type(torch.float).mean().item()

        logger.info(f"Shape of features: {features.shape}")
        logger.info(f"Shape of labels:   {labels.shape}")
        logger.info(f"Valid fraction:    {valid_fraction}")
        logger.info(f"Negative fraction: {neg_fraction}")
        logger.info(f"Small fraction:    {small_fraction}")
        logger.info(f"Negative log1 fraction: {neg_fraction_log1}")
        return {"mode": "count_negative_weights"}

    # Specify the loss function

    # Mean Squared Error loss for regression tasks
    loss_fn = nn.MSELoss()

    # loss_fn = nn.SmoothL1Loss()

    # %% Set the model and choose an optimizer.
    input_dim = dataset.input_shape[0]
    model_name = arg.model
    if arg.standardise:
        model = model_dict[model_name](input_dim=input_dim, stat_norm=stat_norm)
    elif model_name == "FFNN_general":
        model = model_dict[model_name](input_dim=input_dim, width=run_settings.width.value, n_hidden=run_settings.n_hidden.value)
    else:
        model = model_dict[model_name](input_dim=input_dim)

    # TODO: Change the class name to model_name.

    # if torch.cuda.device_count() > 1:
    #   logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
    #   model = nn.DataParallel(model)

    model.to(device)
    logger.info(f"Model {model_name} device: {next(model.parameters()).device}")

    if torch.cuda.is_available():
        model_summary = str(summary(model.cuda(), input_size=(input_dim,), batch_dim=0, verbose=0))
    else:
        model_summary = str(summary(model, input_size=(input_dim,), batch_dim=0, verbose=0))
    logger.info(f"\n{model_summary}")

    if arg.outputdir is not None:
        model_dir = arg.outputdir
        model_dir.mkdir(exist_ok=True)
    else:
        model_dir = Path().cwd()

    run_settings.set("model_dir", model_dir, overwrite=True)
    run_settings.set("device", device, overwrite=True)
    run_settings.set("histogram_dir", files[0].parent, overwrite=True)

    try:
        run_settings.dump_yaml(model_dir / "run_settings.yaml")
    except Exception as exc:
        logger.warning(f"Could not dump run settings YAML: {exc}")

    optimizers = {
        "SGD":            torch.optim.SGD(    model.parameters(), lr=learning_rate),
        "Adam":           torch.optim.Adam(   model.parameters(), lr=learning_rate),
        "AdamW":          torch.optim.AdamW(  model.parameters(), lr=learning_rate, weight_decay=1e-4),
        "RMSprop":        torch.optim.RMSprop(model.parameters(), lr=learning_rate),
        "paper":          torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.0),
        "paper_momentum": torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.9),
    }
    optimizer = optimizers[arg.optimizer]
    logger.info(f"Using optimizer:\n{optimizer}")

    # %% Test implementation on one batch before to train
    xb, yb = next(iter(train_dataloader))

    # test_batch_size = 10
    # xb = torch.randn(test_batch_size, input_dim)
    # yb = torch.randn(test_batch_size)

    # Put tensors on device
    xb = xb.type(torch.float).to(device)
    yb = yb.type(torch.float).to(device)

    # Prediction
    pred = model(xb)
    # Loss and metric
    loss = loss_fn(pred, yb[:, 0].unsqueeze(-1))
    # loss = loss_fn(pred, torch.unsqueeze(yb,1))  # Bring yb to shape (batch_size, 1) to match pred shape.
    # metric = binary_accuracy(pred, torch.unsqueeze(yb,1))

    logger.info(f"{xb.shape = }")
    logger.info(f"{yb.shape = }")
    logger.info(f"output shape: {pred.shape}")
    logger.info(f"{pred.squeeze().shape = }")
    logger.info(f"loss: {loss.item()}")

    if arg.test_mode:
        logger.info("Exiting script now after testing implementation of the model on one point.")
        return {"mode": "test_mode", "model_dir": model_dir, "model_name": model_name}

    # %% Training loop

    # Save the seed used for this training
    with open(model_dir / "training_seed.txt", "w") as handle:
        handle.write(f"{seed}\n")

    hist_loss     = []
    hist_val_loss = []
    hist_lr       = []
    best_val_loss = float("inf")
    patience      = arg.patience
    patience_counter = 0
    best_model_state = None

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
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
    with open(train_loss_file, "w") as handle:
        handle.write("epoch,train_loss\n")
    with open(val_loss_file, "w") as handle:
        handle.write("epoch,val_loss\n")
    with open(lr_file, "w") as handle:
        handle.write("epoch,learning_rate\n")

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Early stopping patience: {patience}")

    if not arg.penalties:
        penalties = {}
    else:
        penalties = {penalty: True for penalty in arg.penalties}
        # Ensure that Z decay angle penalty is disabled when using Z+jet dataset or the januar2026 input choice, as the relevant features are not included in these cases.
        if arg.use_zjet or (arg.input_choice in ["jan2026"]):
            penalties["ZdecayAngles"] = False

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        logger.info("-" * 50)

        # Training phase
        train_loss = train_loop(
            epoch,
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            print_freq=2500,
            penalties=penalties,
        )
        # Validation phase
        valid_loss = valid_loop(val_dataloader, model, loss_fn, device)

        hist_loss.append(train_loss)
        hist_val_loss.append(valid_loss)
        hist_lr.append(current_lr)

        with open(train_loss_file, "a") as handle:
            handle.write(f"{epoch + 1},{train_loss:.10e}\n")
        with open(val_loss_file, "a") as handle:
            handle.write(f"{epoch + 1},{valid_loss:.10e}\n")
        with open(lr_file, "a") as handle:
            handle.write(f"{epoch + 1},{current_lr:.10e}\n")

        # Learning rate scheduling
        # scheduler.step()
        scheduler.step(valid_loss)

        # Report intermediate value to Optuna for pruning
        # This allows Optuna to monitor trial progress at each epoch and make pruning decisions.
        # If the trial is underperforming relative to its pruner's strategy, it will be terminated early
        # to save training time and GPU resources.
        if trial is not None:
            trial.report(valid_loss, step=epoch)
            if trial.should_prune():
                logger.info(f"Trial pruned at epoch {epoch + 1}")
                raise optuna.TrialPruned()

        # Early stopping check
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch time: {epoch_time:.2f} seconds")
        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            logger.info(f"Best validation loss: {best_val_loss:.6f}")
            break

    # Save final model
    model_filename = model_dir / f"{model_name}_model_weights_final.pt"
    torch.save(model.state_dict(), model_filename)

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save best model separately
        best_model_filename = model_dir / f"{model_name}_model_weights_best.pt"
        torch.save(best_model_state, best_model_filename)
        logger.info(f"Best model saved as: {best_model_filename}")

    hist_loss = np.array(hist_loss)
    hist_val_loss = np.array(hist_val_loss)
    hist_lr = np.array(hist_lr)

    logger.info(f"Final model saved as: {model_filename}")
    logger.info("Training completed!")
    # Print training summary
    logger.info("Training Summary:")
    logger.info(f"Total epochs: {len(hist_loss)}")
    logger.info(f"Final train loss: {hist_loss[-1]:.6f}")
    logger.info(f"Final validation loss: {hist_val_loss[-1]:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Plot loss
    logger.info(f"Plotting training history, using best model weights: {best_model_state is not None}")
    # Generate plot using the plotting function
    try:
        plot_training_history(model_dir, model_name, use_log_scale=True)
    except Exception as exc:
        logger.warning(f"Could not generate plot: {exc}")

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    if arg.do_test:
        if best_val_loss == float("inf"):
            logger.warning("Best validation loss is infinite, skipping test run.")
        else:
            do_test_run(run_settings, model, test_dataset)

    return {
        "mode": "train",
        "model_dir": model_dir,
        "model_name": model_name,
        "best_val_loss": best_val_loss,
    }


def main() -> int:
    log_file = "output.log"
    mllogger = setup_file_logger(log_file, level="DEBUG", console=False, mode="w", force=True)

    mllogger.info(f"numpy:  {np.__version__}")
    mllogger.info(f"pandas: {pd.__version__}")
    mllogger.info(f"torch:  {torch.__version__}")

    arg = parse_args()
    run_settings = prepare_run_settings(arg)

    try:
        run_training(run_settings, mllogger)
    except Exception as exc:
        mllogger.error(f"Training failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
