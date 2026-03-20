import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ml_events_utils import Settings, setup_file_logger
from polarisation_train import run_training

try:
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import Trial
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install with: pip install optuna"
    ) from exc

# Example usage:
# python ML_Giovanni/polarisation_tagging/run_hyperparam_scan_optuna.py path/to/run_settings.yaml --study-name pol_scan --storage sqlite:///pol_scan.db --n-trials 120 --gpus 0 1 2 3 --output-root scan_runs_optuna

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
    return parser.parse_args()


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
        low = fit_spec["min"]
        high = fit_spec["max"]
        value_type = fit_spec.get("type", "float")
        step = fit_spec.get("step", None)
        log = bool(fit_spec.get("log", False))

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

    overrides["outputdir"] = trial_dir
    overrides["replot_only"] = False
    overrides["do_test"] = bool(run_test)

    return base_settings.with_overrides(overrides, overwrite=True)


def objective_factory(base_settings: Settings, output_root: Path, run_test: bool):
    def objective(trial: Trial) -> float:
        trial_settings = build_trial_settings(base_settings, trial, output_root, run_test)

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

        result = run_training(trial_settings, trial_logger)
        if "best_val_loss" not in result:
            raise RuntimeError("run_training did not return best_val_loss for this trial.")

        return float(result["best_val_loss"])

    return objective


def run_worker(arg: argparse.Namespace) -> int:
    # Bind this worker to one physical GPU by masking visible devices.
    # Inside training we then set settings.gpu=0, because each worker sees only one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)

    base_settings = Settings.load_yaml(arg.settings_file)
    base_settings.set("gpu", 0, overwrite=True)

    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=arg.study_name,
        storage=arg.storage,
        direction="minimize",
        load_if_exists=True,
    )

    objective = objective_factory(base_settings, output_root, arg.run_test)

    # MaxTrialsCallback enforces a global cap across all workers connected to this study.
    max_trials_cb = MaxTrialsCallback(arg.n_trials)
    study.optimize(objective, n_trials=None, callbacks=[max_trials_cb])
    return 0


def run_coordinator(arg: argparse.Namespace) -> int:
    output_root = arg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Create the study once so workers can attach immediately.
    optuna.create_study(
        study_name=arg.study_name,
        storage=arg.storage,
        direction="minimize",
        load_if_exists=True,
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
    print("Best trial:")
    print(f"  number: {best.number}")
    print(f"  value:  {best.value}")
    print(f"  params: {best.params}")
    print(f"Output root: {output_root}")
    print(f"Study DB: {arg.storage}")

    return exit_code


def main() -> int:
    arg = parse_args()

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
