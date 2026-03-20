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
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter scan using run_settings.yaml and run_training().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "settings_file",
        type=Path,
        help="Path to a base run_settings.yaml file.",
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
        help="Root directory where per-trial folders and summary are written.",
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
        help="Run do_test_run() after each trial (slower).",
    )
    return parser.parse_args()


def _sample_from_fit_spec(fit_spec: Any, rng: random.Random) -> Any:
    # fit=False means do not scan this setting.
    if fit_spec is False:
        return None

    # A scalar fit specification means fixed override value.
    if not isinstance(fit_spec, (dict, list, tuple)):
        return fit_spec

    # List/tuple shorthand:
    # - [a, b] with numeric values -> uniform float
    # - [v1, v2, ...] -> categorical choice
    if isinstance(fit_spec, (list, tuple)):
        if len(fit_spec) == 2 and all(isinstance(v, (int, float)) for v in fit_spec):
            low, high = float(fit_spec[0]), float(fit_spec[1])
            return rng.uniform(low, high)
        return rng.choice(list(fit_spec))

    # Dict-based rich specification.
    # Supported keys:
    # - {"choices": [...]} for categorical
    # - {"min": x, "max": y, "type": "float"|"int", "log": bool}
    if "choices" in fit_spec:
        return rng.choice(list(fit_spec["choices"]))

    if "min" in fit_spec and "max" in fit_spec:
        low = fit_spec["min"]
        high = fit_spec["max"]
        value_type = fit_spec.get("type", "float")
        use_log = bool(fit_spec.get("log", False))

        if use_log:
            import math

            sampled = math.exp(rng.uniform(math.log(float(low)), math.log(float(high))))
        else:
            sampled = rng.uniform(float(low), float(high))

        if value_type == "int":
            step = int(fit_spec.get("step", 1))
            base = int(round(sampled))
            return int(round(base / step) * step)
        return float(sampled)

    raise ValueError(f"Unsupported fit specification: {fit_spec}")


def _build_trial_overrides(base_settings: Settings, rng: random.Random) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, parameter in base_settings.items():
        sampled = _sample_from_fit_spec(parameter.fit, rng)
        if sampled is not None:
            overrides[key] = sampled
    return overrides


def main() -> int:
    arg = parse_args()

    base_settings = Settings.load_yaml(arg.settings_file)
    rng = random.Random(arg.seed)

    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary_file = output_root / "scan_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "status", "best_val_loss", "model_dir", "overrides"])

        for trial_idx in range(arg.n_trials):
            trial_overrides = _build_trial_overrides(base_settings, rng)
            trial_dir = output_root / f"trial_{trial_idx:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            trial_overrides["outputdir"] = trial_dir
            trial_overrides["replot_only"] = False
            trial_overrides["do_test"] = bool(arg.run_test)

            trial_settings = base_settings.with_overrides(trial_overrides, overwrite=True)

            logger = setup_file_logger(
                trial_dir / "output.log",
                level="DEBUG",
                console=False,
                mode="w",
                force=True,
            )
            logger.info(f"Starting trial {trial_idx}")

            try:
                result = run_training(trial_settings, logger)
                best_val_loss = result.get("best_val_loss", "")
                status = "ok"
            except Exception as exc:
                logger.error(f"Trial failed: {exc}")
                best_val_loss = ""
                status = "failed"

            writer.writerow([
                trial_idx,
                status,
                best_val_loss,
                str(trial_dir),
                repr(trial_overrides),
            ])
            handle.flush()

    print(f"Scan finished. Summary written to: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
