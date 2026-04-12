from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import argparse
import os
import subprocess
import sys

from ml_events_utils import Settings, setup_file_logger
from ml_events_utils import stylesheet_default
from polarisation_train import run_training

try:
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_slice,
        plot_param_importances,
    )
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install with: pip install optuna"
    ) from exc

# Example usage:
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan_optuna.py path/to/run_settings.yaml --study-name pol_scan --storage sqlite:///pol_scan.db --n-trials 120 --gpus 0 1 2 3 --output-root scan_runs_optuna
# With pruning:
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan_optuna.py path/to/run_settings.yaml --study-name pol_scan --storage sqlite:///pol_scan.db --n-trials 120 --gpus 0 1 2 3 --output-root scan_runs_optuna --enable-pruning --pruner median
# With auto-plotting after optimization:
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan_optuna.py path/to/run_settings.yaml --study-name pol_scan --storage sqlite:///pol_scan.db --n-trials 120 --gpus 0 1 2 3 --output-root scan_runs_optuna --plot-after
# Plot existing study only (no optimization):
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan_optuna.py path/to/run_settings.yaml --study-name pol_scan --storage sqlite:///pol_scan.db --plot-only --output-root scan_runs_optuna

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization with Optuna for polarisation training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("settings_file", type=Path, help="Base run_settings.yaml file.")
    parser.add_argument("--study-name", type=str, default="polarisation_scan", help="Optuna study name.")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_scan.db",
        help="Optuna storage URL. Use sqlite:///... for resumable local studies.",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Total number of trials across all workers.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("scan_runs_optuna"),
        help="Root directory for per-trial folders.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=[0],
        help="GPU ids to use in parallel, e.g. --gpus 0 1 2 3.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of worker processes per GPU.",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test stage after each trial (slower).",
    )
    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        help="Enable Optuna pruning to terminate unpromising trials early.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "percentile"],
        help=(
            "Pruning algorithm to use. "
            "'median': Prunes trials performing worse than the median (conservative, good default). "
            "'percentile': Prunes bottom 25%% of trials (aggressive, for faster scanning)."
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag: run a single worker process instead of launching workers.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Internal flag: physical GPU id assigned to this worker.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing study and generate plots without running optimization.",
    )
    parser.add_argument(
        "--plot-after",
        action="store_true",
        help="Generate plots automatically after optimization completes.",
    )
    args = parser.parse_args()

    # Convert output_root to absolute path to avoid ambiguity when workers spawn in different contexts
    args.output_root = args.output_root.resolve()

    return args


def suggest_from_fit_spec(trial: Trial, name: str, fit_spec: Any, current_value: Any) -> Any:
    # fit=False means: keep the original value from base settings.
    if fit_spec is False:
        return current_value

    # Scalar value means: fixed override to this value.
    if not isinstance(fit_spec, (dict, list, tuple)):
        return fit_spec

    # List/tuple shorthand:
    # - [a, b] numeric -> sample in range
    # - otherwise      -> categorical choice
    if isinstance(fit_spec, (list, tuple)):
        values = list(fit_spec)
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            low, high = values
            if all(isinstance(v, int) for v in values):
                return trial.suggest_int(name, int(low), int(high))
            return trial.suggest_float(name, float(low), float(high))
        return trial.suggest_categorical(name, values)

    # Dict-based rich specification.
    # Supported examples:
    # 1) {"choices": ["Adam", "RMSprop"]}
    # 2) {"min": 1e-5, "max": 1e-2, "log": true, "type": "float"}
    # 3) {"min": 32, "max": 2048, "step": 32, "type": "int"}
    if "choices" in fit_spec:
        return trial.suggest_categorical(name, list(fit_spec["choices"]))

    if "min" in fit_spec and "max" in fit_spec:
        low  = fit_spec["min"]
        high = fit_spec["max"]
        value_type = fit_spec.get("type", "float")
        step = fit_spec.get("step", None)
        log  = bool(fit_spec.get("log", False))

        if value_type == "int":
            kwargs = {}
            if step is not None:
                kwargs["step"] = int(step)
            return trial.suggest_int(name, int(low), int(high), **kwargs)

        kwargs = {"log": log}
        if step is not None:
            kwargs["step"] = float(step)
        return trial.suggest_float(name, float(low), float(high), **kwargs)

    raise ValueError(f"Unsupported fit specification for '{name}': {fit_spec}")


def build_trial_settings(base_settings: Settings, trial: Trial, output_root: Path, run_test: bool) -> Settings:
    overrides: Dict[str, Any] = {}

    for key, parameter in base_settings.items():
        sampled_value = suggest_from_fit_spec(trial, key, parameter.fit, parameter.value)
        overrides[key] = sampled_value

    # Per-trial isolation for outputs and logs.
    trial_dir = output_root / f"trial_{trial.number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides["outputdir"]   = trial_dir
    overrides["replot_only"] = False
    overrides["do_test"]     = bool(run_test)

    return base_settings.with_overrides(overrides, overwrite=True), overrides


def objective_factory(base_settings: Settings, output_root: Path, run_test: bool):
    def objective(trial: Trial) -> float:
        trial_settings, overwritten_settings = build_trial_settings(base_settings, trial, output_root, run_test)

        # Each trial gets its own log file in its own output folder.
        trial_log_file = trial_settings.outputdir.value / "output.log"
        trial_logger = setup_file_logger(
            trial_log_file,
            level="DEBUG",
            console=False,
            mode="w",
            force=True,
        )

        # Store sampled params in Optuna UI / DB for convenient filtering.
        for key, parameter in trial_settings.items():
            trial.set_user_attr(key, str(parameter.value))

        result = run_training(trial_settings, trial_logger, trial=trial)
        if "best_val_loss" not in result:
            raise RuntimeError("run_training did not return best_val_loss for this trial.")

        return float(result["best_val_loss"])

    return objective


def create_study_summary_figure(study: optuna.Study, study_name: str, storage: str, target_name: str) -> plt.Figure:
    """Create a matplotlib figure containing study summary information.

    Args:
        study: Loaded Optuna study object
        study_name: Name of the study
        storage: Storage URL/path

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    # Get study statistics
    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
    best_trial = study.best_trial
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format best parameters
    params_str = "\n".join([f"  {k}: {v}" for k, v in best_trial.params.items()])

    # Create text content
    summary_text = f"""
OPTUNA STUDY OPTIMIZATION SUMMARY

Study Name:              {study_name}
Storage:                 {storage}
Analysis Date/Time:      {timestamp}

TRIAL STATISTICS
Total Trials:            {n_trials}
Completed Trials:        {n_complete}
Pruned Trials:           {n_pruned}

BEST TRIAL FOUND
Trial Number:            {best_trial.number}
Best {target_name}:    {best_trial.value:.6e}

Best Parameters:
{params_str}
    """.strip()

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    return fig


def generate_optimization_plots(study: optuna.Study, output_root: Path,
                                 study_name: str, storage: str) -> Path:
    """Generate and save optimization analysis plots to a multipage PDF.

    Uses matplotlib-based Optuna visualization functions for direct integration.

    Args:
        study: Loaded Optuna study object
        output_root: Directory to save PDF
        study_name: Name of the study
        storage: Storage URL/path

    Returns:
        Path to generated PDF file
    """
    plt.style.use(stylesheet_default)

    pdf_path = output_root / f"optimization_analysis_{study_name}.pdf"

    # Save all figures to multipage PDF
    with PdfPages(str(pdf_path)) as pdf:
        d = pdf.infodict()
        d["Title"] = f"Optuna Study Analysis: {study_name}"
        d["Author"] = "Optuna"
        d["Subject"] = "Hyperparameter Optimization Analysis"
        d["Keywords"] = "Optuna, Hyperparameter Optimization"
        d["CreationDate"] = datetime.now()

        target_name="Validation Loss"

        # Page 1: Study summary
        fig_summary = create_study_summary_figure(study, study_name, storage, target_name=target_name)
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)

        # Page 2: Optimization history
        try:
            plot_optimization_history(study, target_name=target_name)
            fig_history = plt.gcf()  # Get current figure
            pdf.savefig(fig_history, bbox_inches="tight")
            plt.close(fig_history)
        except Exception as e:
            print(f"Warning: Could not generate optimization history plot: {e}")

        # Page 3: Slice plot (parameter importance via slices)
        try:
            plot_slice(study, target_name=target_name)
            fig_slice = plt.gcf()  # Get current figure (may have multiple subplots)
            pdf.savefig(fig_slice, bbox_inches="tight")
            plt.close(fig_slice)
        except Exception as e:
            print(f"Warning: Could not generate slice plot: {e}")

        # Page 4: Parameter importance ranking (only if enough trials)
        if len(study.trials) >= 2:
            try:
                plot_param_importances(study, target_name=target_name)
                fig_importance = plt.gcf()  # Get current figure
                pdf.savefig(fig_importance, bbox_inches="tight")
                plt.close(fig_importance)
            except Exception as e:
                print(f"Warning: Could not generate parameter importance plot: {e}")

    return pdf_path


def run_worker(arg: argparse.Namespace) -> int:
    # Bind this worker to one physical GPU by masking visible devices.
    # Inside training we then set settings.gpu=0, because each worker sees only one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)

    base_settings = Settings.load_yaml(arg.settings_file)
    base_settings.set("gpu", 0, overwrite=True)

    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Create pruner if enabled
    # Pruning terminates unpromising trials early to save computational resources.
    #
    # MedianPruner (conservative, recommended):
    #   - Compares each trial's intermediate values against the median of completed trials
    #   - n_startup_trials=20: Don't prune until 20 trials complete (gives algorithm warm-up period)
    #   - n_warmup_steps=10: Start pruning from epoch 11 (10 warm-up epochs)
    #   - Best for: Balanced exploration/exploitation, reducing noise
    #   - Benefit: Avoids pruning good long-training models too early
    #
    # PercentilePruner (aggressive):
    #   - Prunes trials in the bottom percentile (here: 25th, i.e., worst 25%)
    #   - Aggressive pruning can speed up scans but may miss good hyperparameters
    #   - Best for: Large scans where speed is priority, or after median found good region
    #   - Benefit: Faster iteration, but requires more trials to find good hyperparameters
    pruner = None
    if arg.enable_pruning:
        if arg.pruner == "median":
            pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        else:  # percentile
            pruner = optuna.pruners.PercentilePruner(percentile=25, n_startup_trials=20, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=arg.study_name,
        storage=arg.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
    )

    objective = objective_factory(base_settings, output_root, arg.run_test)

    # MaxTrialsCallback enforces a global cap on the number of trials across all workers connected to this study.
    # callbacks=[max_trials_cb]: Registers the MaxTrialsCallback to monitor progress and halt optimization when the global limit n_trials is reached.
    max_trials_cb = MaxTrialsCallback(arg.n_trials)
    study.optimize(objective, n_trials=None, callbacks=[max_trials_cb])
    return 0


def run_coordinator(arg: argparse.Namespace) -> int:
    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Create pruner if enabled
    # Pruning terminates unpromising trials early to save computational resources.
    #
    # MedianPruner (conservative, recommended):
    #   - Compares each trial's intermediate values against the median of completed trials
    #   - n_startup_trials=20: Don't prune until 20 trials complete (gives algorithm warm-up period)
    #   - n_warmup_steps=10: Start pruning from epoch 11 (10 warm-up epochs)
    #   - Best for: Balanced exploration/exploitation, reducing noise
    #   - Benefit: Avoids pruning good long-training models too early
    #
    # PercentilePruner (aggressive):
    #   - Prunes trials in the bottom percentile (here: 25th, i.e., worst 25%)
    #   - Aggressive pruning can speed up scans but may miss good hyperparameters
    #   - Best for: Large scans where speed is priority, or after median found good region
    #   - Benefit: Faster iteration, but requires more trials to find good hyperparameters
    pruner = None
    if arg.enable_pruning:
        if arg.pruner == "median":
            pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        else:  # percentile
            pruner = optuna.pruners.PercentilePruner(percentile=25, n_startup_trials=20, n_warmup_steps=10)

    # Create the study once so workers can attach immediately.
    optuna.create_study(
        study_name=arg.study_name,
        storage=arg.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
    )

    worker_cmd_base: List[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(arg.settings_file),
        "--study-name",
        arg.study_name,
        "--storage",
        arg.storage,
        "--n-trials",
        str(arg.n_trials),
        "--output-root",
        str(arg.output_root),
        "--worker",
    ]
    if arg.run_test:
        worker_cmd_base.append("--run-test")
    if arg.enable_pruning:
        worker_cmd_base.append("--enable-pruning")
        worker_cmd_base.extend(["--pruner", arg.pruner])

    procs: List[subprocess.Popen] = []
    for gpu_id in arg.gpus:
        for _ in range(arg.workers_per_gpu):
            cmd = worker_cmd_base + ["--gpu-id", str(gpu_id)]
            procs.append(subprocess.Popen(cmd))

    exit_code = 0
    for proc in procs:
        code = proc.wait()
        if code != 0:
            exit_code = code

    # Print best trial summary after all workers finished.
    study = optuna.load_study(study_name=arg.study_name, storage=arg.storage)
    best = study.best_trial
    print(f"Best trial in study {arg.study_name}:")
    print(f"  number: {best.number}")
    print(f"  value:  {best.value}")
    print(f"  params: {best.params}")
    print(f"Best trial directory: {best.params.get('outputdir', 'N/A')}")
    print(f"Output root: {output_root}")
    print(f"Study DB: {arg.storage}")

    # Generate plots if requested
    if arg.plot_after:
        print("\nGenerating optimization analysis plots...")
        pdf_path = generate_optimization_plots(study, output_root, arg.study_name, arg.storage)
        print(f"Plots saved to: {pdf_path}")

    return exit_code


def main() -> int:
    arg = parse_args()

    # Handle plot-only mode: load existing study and generate plots
    if arg.plot_only:
        try:
            study = optuna.load_study(study_name=arg.study_name, storage=arg.storage)
        except Exception as e:
            print(f"Error loading study '{arg.study_name}' from storage '{arg.storage}': {e}")
            return 1

        # Print study summary to console
        best = study.best_trial
        print(f"Study: {arg.study_name}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best trial number: {best.number}")
        print(f"Best objective value: {best.value}")
        print(f"Best parameters: {best.params}\n")

        # Generate plots
        print("Generating optimization analysis plots...")
        try:
            pdf_path = generate_optimization_plots(study, arg.output_root, arg.study_name, arg.storage)
            print(f"Plots saved to: {pdf_path}")
        except Exception as e:
            print(f"Error generating plots: {e}")
            return 1

        return 0

    # Workflow overview:
    # 1) Put fit specifications into run_settings.yaml (in each parameter's fit field).
    # 2) Start this script with --gpus 0 1 2 3 ... to launch one worker per GPU.
    # 3) Workers connect to a shared Optuna study (SQLite by default) and pick trials.
    # 4) Each trial writes outputs into output_root/trial_XXXXX/.
    # 5) Study can be resumed by rerunning with the same --study-name and --storage.
    if arg.worker:
        return run_worker(arg)
    return run_coordinator(arg)


if __name__ == "__main__":
    raise SystemExit(main())
