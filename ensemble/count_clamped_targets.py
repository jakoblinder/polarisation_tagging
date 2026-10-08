"""Count the events whose target log(1 + w_LL/w_UU) is clamped by log_target_transform, i.e. r_LL = w_LL/w_UU <= -1
(possible only with negative event weights). The clamp sets the target to log(1e-10) = -23.03, so a single such event
dominates the unweighted validation/test MSE (e.g. ~5.5e-3 for a 96k-event validation set).

The counts are given per train/validation/test split, reproduced exactly as in polarisation_train.py
(MLEventsDataset sorts the files, random_split with a generator seeded by the run's seed), for
  - the production data sets of FFNN/AE and of ParticleNet (file lists from their run_settings.yaml),
  - the ensemble bunches (ensemble_runs/ensemble_overview.json).

Usage (from polarisation_tagging/, on /scratch where the event files are local):
    python ensemble/count_clamped_targets.py [--output ensemble_runs/clamped_targets.csv]
"""
# %% Imports
import argparse
import json
import re
import sys

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from braceexpand import braceexpand

PACKAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))

from ml_events_utils import Settings

ML_FILES_DIR      = PACKAGE_DIR.parent / "ML_FILES"
FINAL_RESULTS_DIR = PACKAGE_DIR / "final_results"
ENSEMBLE_RUNS_DIR = PACKAGE_DIR / "ensemble_runs"
ORDER_DIRS        = {"LO": "UU_LO", "LOwS": "UU_LOwS", "NLOPS": "UU_NLO"}

WEIGHT_PATTERN = {wid: re.compile(rf"<weight id='{wid}'>\s*(\S+)") for wid in ("UU", "LL")}


def read_weights(eventfile: Path) -> tuple[np.ndarray, np.ndarray]:
    """w_UU and w_LL of all events of a file, in file order (the trailing <MLMeanValues> summary block is skipped)."""
    text = eventfile.read_text().split("<MLMeanValues>")[0]
    w_uu = np.array(WEIGHT_PATTERN["UU"].findall(text), dtype=np.float64)
    w_ll = np.array(WEIGHT_PATTERN["LL"].findall(text), dtype=np.float64)
    if len(w_uu) != len(w_ll) or len(w_uu) != text.count("<event>"):
        raise RuntimeError(f"Inconsistent number of weights in {eventfile}")
    return w_uu, w_ll


def files_from_settings(settings_file: Path, order: str) -> list[Path]:
    """Event files of a run, resolved by their names in the local ML_FILES directory of the order."""
    files = []
    for pattern in Settings.load_yaml(settings_file).mlfiles.value:
        files += [ML_FILES_DIR / ORDER_DIRS[order] / Path(p).name for p in braceexpand(Path(pattern).name)]
    return sorted(set(files))


def count(name: str, files: list[Path], seed: int, split_ratios, weights: dict) -> dict:
    w_uu = np.concatenate([weights[f][0] for f in files])
    w_ll = np.concatenate([weights[f][1] for f in files])
    # Same arithmetic as the training targets: ratio in float32, clamp of 1 + r at 1e-10 in log_target_transform.
    r = torch.tensor(w_ll / w_uu, dtype=torch.float32)
    y = torch.log(torch.clamp(r + 1.0, min=1e-10)).numpy()

    subsets = torch.utils.data.random_split(range(len(y)), split_ratios, generator=torch.Generator().manual_seed(seed))
    row = {"set": name, "n_files": len(files), "seed": seed, "n_events": len(y),
           "n_neg_w_UU": int((w_uu < 0).sum()), "n_neg_w_LL": int((w_ll < 0).sum())}
    for split, subset in zip(("train", "val", "test"), subsets):
        ys = y[np.asarray(subset.indices)]
        row[f"n_{split}"]           = len(ys)
        row[f"clamped_{split}"]     = int((ys <= np.log(1e-10) + 1e-3).sum())
        row[f"y_below_-5_{split}"]  = int(((ys < -5) & (ys > np.log(1e-10) + 1e-3)).sum())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ENSEMBLE_RUNS_DIR / "clamped_targets.csv")
    args = parser.parse_args()

    # (name, order, files, seed, split ratios) of every data set to count
    sets = []
    for arch in ("ffnn", "pn"):
        for order in ORDER_DIRS:
            settings_file = (FINAL_RESULTS_DIR / arch / order / f"{order}_evts" / "run_settings.yaml" if arch == "ffnn"
                             else FINAL_RESULTS_DIR / arch / order / "run_settings.yaml")
            settings = Settings.load_yaml(settings_file)
            label = "FFNN/AE" if arch == "ffnn" else "PN"
            sets.append((f"production {label} {order}", order, files_from_settings(settings_file, order),
                         settings.seed.value, list(settings.split_ratios.value)))
    with open(ENSEMBLE_RUNS_DIR / "ensemble_overview.json") as handle:
        for info in json.load(handle):
            if info["arch"] == "ffnn":   # the bunches and seeds are the same for all architectures
                sets.append((f"ensemble {info['order']} bunch {info['bunch']}", info["order"],
                             [ML_FILES_DIR / ORDER_DIRS[info["order"]] / Path(f).name for f in info["files"]],
                             info["seed"], info["split_ratios"]))

    eventfiles = sorted({f for s in sets for f in s[2]})
    with ProcessPoolExecutor(max_workers=32) as pool:
        weights = dict(zip(eventfiles, pool.map(read_weights, eventfiles, chunksize=8)))

    table = pd.DataFrame([count(name, files, seed, ratios, weights) for name, _, files, seed, ratios in sets])
    table.to_csv(args.output, index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
