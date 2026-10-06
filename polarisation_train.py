# %% Imports
import os
import time
import torch
import optuna
import copy
import sys
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader

from ml_events_utils.transforms import januar2026_input_choice
from ml_events_utils import MLEventsDataset, get_statistics_from_dataset, scale_target, boost_into_four_lepton_cm_frame, log_target_transform, exp_target_transform
from ml_events_utils import boost_into_Zjet_cm_frame
from ml_events_utils import train_loop, valid_loop
from ml_events_utils import ZJetDataset
from ml_events_utils.models import *
from ml_events_utils import log_file, setup_file_logger
from ml_events_utils import prepare_run_settings, Settings, select_device
from polarisation_test import do_test_run
from plot_training_history import plot_training_history


def load_statistics_from_file(output_root_parent: Path, logger) -> dict:
    """Load pre-computed dataset statistics from JSON file.

    Args:
        output_root_parent: Parent directory containing dataset_statistics.json
                           (typically {output_root} from hyperparameter scan)
        logger: Logger instance for logging messages

    Returns:
        dict with 'mean' and 'stddev' numpy arrays, or None if file doesn't exist/is invalid
    """
    stats_file = output_root_parent / "dataset_statistics.json"

    if not stats_file.exists():
        logger.warning(f"File not found: {stats_file}")
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
        logger.warning(f"Could not load statistics from {stats_file}: {e}")
        return None


def run_training(run_settings: Settings, logger, trial=None):
    start_time = time.time()

    run_settings.log_to_logger(logger, header="Arguments:")

    seed = run_settings.seed.value
    torch.manual_seed(seed)
    np.random.seed(seed)

    # %% Replot mode - load existing data and regenerate plot
    if run_settings.replot_only.value:
        logger.info("Running in replot mode - loading existing training history...")

        if not run_settings.outputdir.value.exists():
            raise FileNotFoundError(f"Directory {run_settings.outputdir.value} does not exist")

        plot_training_history(run_settings.outputdir.value, run_settings.model.value, use_log_scale=True)
        logger.info("Replot completed!")
        return {"mode": "replot", "outputdir": run_settings.outputdir.value, "model_name": run_settings.model.value}

    # Data Handling
    logger.info(f"Cache events: {run_settings.cache_events.value}")

    if not run_settings.use_zjet.value:
        if run_settings.labframe.value:
            trafo = None
        else:
            trafo = boost_into_four_lepton_cm_frame

        if run_settings.input_choice.value == "jan2026":
            if trafo:
                trafo = lambda x: januar2026_input_choice(trafo(x))
            else:
                trafo = januar2026_input_choice

        dataset = MLEventsDataset(
            run_settings.mlfiles.value,
            labels=[f"{run_settings.polarisation.value}/UU", "UU"],
            transform=trafo,
            # Apply log transform to reduce outlier impact
            target_transform=log_target_transform,
            # Inverse transform to revert log transformation
            inv_target_transform=exp_target_transform,
            cache_events=run_settings.cache_events.value,
            # Specify whether standardisation over the whole dataset is enabled (this changes the dataset).
            standardise=False,
        )
    else:
        if run_settings.labframe.value:
            trafo = None
        else:
            trafo = boost_into_Zjet_cm_frame

        if run_settings.input_choice.value == "jan2026":
            if trafo:
                trafo = lambda x: januar2026_input_choice(trafo(x))
            else:
                trafo = januar2026_input_choice

        dataset = ZJetDataset(
            run_settings.mlfiles.value[0],
            transform=trafo,
            target_transform=None,
            # Maximum number of events to load (useful for testing). Max = 10^6.
            max_events=None,
            # Specify whether standardisation over the whole dataset is enabled (this changes the dataset).
            standardise=False,
        )

    logger.info(f"Dataset info: {dataset.get_file_info()}")

    # Hyperparameters
    learning_rate = run_settings.learning_rate.value
    batch_size    = run_settings.batch_size.value
    epochs        = run_settings.epochs.value
    n_workers     = run_settings.nworkers.value
    split_ratios  = run_settings.split_ratios.value

    # Split the dataset into training, validation and test sets
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, split_ratios, generator=generator
    )

    logger.info(f"Train dataset size:      {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size:       {len(test_dataset)}")

    stat_norm = None
    if run_settings.standardise.value:
        # Try to load cached statistics from parent directory (hyperparameter scan cache)
        output_root_parent = Path(run_settings.outputdir.value).parent
        cached_stats = load_statistics_from_file(output_root_parent, logger)

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
    logger.info(f"DataLoader created with batch_size={batch_size}, num_workers={n_workers}")

    # Test iteration (only first batch to avoid long output)
    for batch_idx, (batch_features, batch_labels) in enumerate(train_dataloader):
        logger.info(
            f"Batch {batch_idx}: features shape {batch_features.shape}, labels shape {batch_labels.shape}"
        )
        input_dim = batch_features.shape[1]
        logger.info(f"{input_dim = }")
        break

    device = select_device(run_settings.gpu.value, logger)

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
        xsec_estimate /= (split_ratios[0] * run_settings.n_generated_events.value)

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
    model_name = run_settings.model.value
    if run_settings.standardise.value:
        model = model_dict[model_name](input_dim=input_dim, stat_norm=stat_norm)
    elif model_name == "FFNN_general":
        model = model_dict[model_name](input_dim=input_dim, width=run_settings.width.value, n_hidden=run_settings.n_hidden.value)
    elif model_name == "ParticleNet" and run_settings.fittable:
        model = model_dict[model_name](input_dim=input_dim, embed_dim=run_settings.width.value, num_layers=run_settings.n_hidden.value, growing_edge=run_settings.growing_edge.value)
    else:
        model = model_dict[model_name](input_dim=input_dim)


    model.to(device)
    logger.info(f"Model {model_name} device: {next(model.parameters()).device}")

    # Count trainable parameters
    def count_parameters(model):
        """Count total trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_param_count = count_parameters(model)
    logger.info(f"Model {model_name}: {model_param_count:,} trainable parameters")

    if torch.cuda.is_available():
        model_summary = str(summary(model.cuda(), input_size=(input_dim,), batch_dim=0, verbose=0))
    else:
        model_summary = str(summary(model, input_size=(input_dim,), batch_dim=0, verbose=0))
    logger.info(f"\n{model_summary}")

    outputdir = run_settings.outputdir.value
    outputdir.mkdir(parents=True, exist_ok=True)

    run_settings.set("device",    device,    overwrite=True)

    try:
        run_settings.dump_yaml(outputdir / "run_settings.yaml")
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
    optimizer = optimizers[run_settings.optimizer.value]
    logger.info(f"Using optimizer:\n{optimizer}")

    # %% Test implementation on one batch before to train
    xb, yb = next(iter(train_dataloader))

    # Put tensors on device
    xb = xb.type(torch.float).to(device)
    yb = yb.type(torch.float).to(device)

    # Prediction
    pred = model(xb)
    # Loss and metric
    loss = loss_fn(pred, yb[:, 0].unsqueeze(-1))

    logger.info(f"{xb.shape = }")
    logger.info(f"{yb.shape = }")
    logger.info(f"output shape: {pred.shape}")
    logger.info(f"{pred.squeeze().shape = }")
    logger.info(f"loss: {loss.item()}")

    if run_settings.test_mode.value:
        logger.info("Exiting script now after testing implementation of the model on one point.")
        return {"mode": "test_mode", "outputdir": outputdir, "model_name": model_name}

    # %% Training loop

    # Save the seed used for this training
    with open(outputdir / "training_seed.txt", "w") as handle:
        handle.write(f"{seed}\n")

    hist_loss     = []
    hist_val_loss = []
    hist_lr       = []
    best_val_loss = float("inf")
    patience      = run_settings.patience.value
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
    train_loss_file = outputdir / f"{model_name}_train_loss.csv"
    val_loss_file   = outputdir / f"{model_name}_val_loss.csv"
    lr_file         = outputdir / f"{model_name}_learning_rates.csv"

    # Write headers (optional, just the first value will be written)
    with open(train_loss_file, "w") as handle:
        handle.write("epoch,train_loss\n")
    with open(val_loss_file, "w") as handle:
        handle.write("epoch,val_loss\n")
    with open(lr_file, "w") as handle:
        handle.write("epoch,learning_rate\n")

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Early stopping patience: {patience}")

    if not run_settings.penalties.value:
        penalties = {}
    else:
        penalties = {penalty: True for penalty in run_settings.penalties.value}
        # Ensure that Z decay angle penalty is disabled when using Z+jet dataset or the januar2026 input choice, as the relevant features are not included in these cases.
        if run_settings.use_zjet.value or (run_settings.input_choice.value in ["jan2026"]):
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
    model_filename = outputdir / f"{model_name}_model_weights_final.pt"
    torch.save(model.state_dict(), model_filename)

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save best model separately
        best_model_filename = outputdir / f"{model_name}_model_weights_best.pt"
        torch.save(best_model_state, best_model_filename)
        run_settings.set("model_weight_file", best_model_filename, overwrite=True)
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
        plot_training_history(outputdir, model_name, use_log_scale=True)
    except Exception as exc:
        logger.warning(f"Could not generate plot: {exc}")

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    if run_settings.do_test.value:
        if best_val_loss == float("inf"):
            logger.warning("Best validation loss is infinite, skipping test run.")
        else:
            do_test_run(run_settings, model, test_dataset)

    # Dump settings again, containing the path to the best model weight file if it was saved successfully.
    try:
        run_settings.dump_yaml(outputdir / "run_settings.yaml")
    except Exception as exc:
        logger.warning(f"Could not dump run settings YAML: {exc}")

    return {
        "mode": "train",
        "outputdir": outputdir,
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "model_param_count": model_param_count,
    }


def main() -> int:
    run_settings = prepare_run_settings(parser_type="train")

    outputdir = run_settings.outputdir.value
    if run_settings.replot_only.value:
        if not outputdir.is_dir():
            print(f"Replot failed: directory {outputdir} does not exist.", file=sys.stderr)
            return 1
    else:
        outputdir.mkdir(parents=True, exist_ok=True)

    log_file = outputdir / "output.log"
    logger   = setup_file_logger(log_file=log_file, level="DEBUG", console=run_settings.verbose.value, mode="w", force=True)

    try:
        run_training(run_settings, logger)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
