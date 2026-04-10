import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict

from ml_events_utils import Settings, setup_file_logger
from polarisation_train import run_training

# Example usage:
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan.py path/to/run_settings.yaml --n-trials 20 --output-root scan_runs


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for hyperparameter scan execution.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - settings_file (Path): Path to a base run_settings.yaml file.
            - n_trials (int): Number of hyperparameter trials. Default: 20
            - output_root (Path): Root directory for per-trial folders and summary. Default: Path("scan_runs")
            - seed (int): Random seed for hyperparameter sampling. Default: 12345
            - run_test (bool): Whether to run do_test_run() after each trial. Default: False
    """
    parser = argparse.ArgumentParser(
        description="Run a random hyperparameter scan using run_settings.yaml and run_training().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "settings_file",
        type=Path,
        help="Base run_settings.yaml file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of hyperparameter trials.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("scan_runs"),
        help="Root directory for per-trial folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for hyperparameter sampling.",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test stage after each trial (slower).",
    )
    return parser.parse_args()


def sample_from_fit_spec(fit_spec: Any, rng: random.Random) -> Any:
    """
    Sample a value from a fit specification using random sampling.

    Supported formats:
    - fit: false -> skip this parameter (return None)
    - fit: scalar -> use fixed value
    - fit: [a, b] -> numeric range (uniform) or categorical choice
    - fit: {choices: [...]} -> categorical sampling
    - fit: {min, max, type, step, log} -> numeric sampling (step only for int)
    """
    # fit=False means do not scan this setting.
    if fit_spec is False:
        return None

    # A scalar fit specification means fixed override value.
    if not isinstance(fit_spec, (dict, list, tuple)):
        return fit_spec

    # List/tuple shorthand:
    # - [a, b] with numeric values -> uniform sampling
    # - [v1, v2, ...] -> categorical choice
    if isinstance(fit_spec, (list, tuple)):
        values = list(fit_spec)
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            if all(isinstance(v, int) for v in values):
                return rng.randint(int(values[0]), int(values[1]))
            return rng.uniform(float(values[0]), float(values[1]))
        return rng.choice(values)

    # Dict-based rich specification.
    # Supported examples:
    # 1) {"choices": ["Adam", "RMSprop"]}
    # 2) {"min": 1e-5, "max": 1e-2, "log": true, "type": "float"}
    # 3) {"min": 32, "max": 2048, "step": 32, "type": "int"}
    if "choices" in fit_spec:
        return rng.choice(list(fit_spec["choices"]))

    if "min" in fit_spec and "max" in fit_spec:
        low  = fit_spec["min"]
        high = fit_spec["max"]
        value_type = fit_spec.get("type", "float")
        step = fit_spec.get("step", None)
        log  = bool(fit_spec.get("log", False))

        if log and (value_type == "float" or value_type not in ["int"]):
            import math
            sampled = math.exp(rng.uniform(math.log(float(low)), math.log(float(high))))
        else:
            sampled = rng.uniform(float(low), float(high))

        if value_type == "int":
            sampled = int(round(sampled))
            if step is not None:
                step = int(step)
                sampled = int(round(sampled / step) * step)
            return sampled

        return float(sampled)

    raise ValueError(f"Unsupported fit specification: {fit_spec}")


def build_trial_settings(
    base_settings: Settings,
    trial_idx: int,
    rng: random.Random,
    output_root: Path,
    run_test: bool,
) -> Settings:
    """Build settings for a single trial by sampling from fit specifications."""
    overrides: Dict[str, Any] = {}

    for key, parameter in base_settings.items():
        sampled_value = sample_from_fit_spec(parameter.fit, rng)
        if sampled_value is not None:
            overrides[key] = sampled_value
        # fit=False results in sampled_value=None and means to keep the original value.

    # Per-trial isolation for outputs and logs.
    trial_dir = output_root / f"trial_{trial_idx:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides["outputdir"]   = trial_dir
    overrides["replot_only"] = False
    overrides["do_test"]     = bool(run_test)

    return base_settings.with_overrides(overrides, overwrite=True), overrides


def main() -> int:
    arg = parse_args()

    base_settings = Settings.load_yaml(arg.settings_file)
    rng = random.Random(arg.seed)

    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary_file = output_root / "scan_summary.csv"
    best_val_loss  = None
    best_trial_idx = None
    failed_count   = 0

    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "status", "best_val_loss", "model_dir", "overrides"])

        for trial_idx in range(arg.n_trials):
            trial_settings, overwritten_settings = build_trial_settings(
                base_settings, trial_idx, rng, output_root, arg.run_test
            )
            trial_dir = trial_settings.outputdir.value

            # Setup per-trial logging.
            trial_log_file = trial_dir / "output.log"
            trial_logger = setup_file_logger(
                trial_log_file,
                level="DEBUG",
                console=False,
                mode="w",
                force=True,
            )
            trial_logger.info(f"Starting trial {trial_idx}")

            try:
                result = run_training(trial_settings, trial_logger)
                if "best_val_loss" not in result:
                    raise RuntimeError("run_training did not return best_val_loss for this trial.")

                trial_best_loss = float(result["best_val_loss"])
                status = "ok"

                # Track best trial so far.
                if best_val_loss is None or trial_best_loss < best_val_loss:
                    best_val_loss  = trial_best_loss
                    best_trial_idx = trial_idx

            except Exception as exc:
                trial_logger.error(f"Trial failed: {exc}")
                trial_best_loss = ""
                status = "failed"
                failed_count += 1

            if trial_best_loss == float("inf"):
                trial_logger.warning("Trial resulted in infinite validation loss, marking as failed.")
                trial_best_loss = ""
                status = "failed"
                failed_count += 1

            # Record trial results.
            writer.writerow([
                trial_idx,
                status,
                trial_best_loss,
                str(trial_dir),
                repr(overwritten_settings),
            ])
            handle.flush()

    # Print summary after all trials complete.
    print("\n" + "=" * 70)
    print("Scan finished.")
    print(f"  Total trials:   {arg.n_trials}")
    print(f"  Successful:     {arg.n_trials - failed_count}")
    print(f"  Failed:         {failed_count}")
    if best_trial_idx is not None:
        print(f"  Best trial:     trial_{best_trial_idx:05d}")
        print(f"  Best loss:      {best_val_loss}")
    print(f"  Summary file:   {summary_file}")
    print(f"  Output root:    {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

