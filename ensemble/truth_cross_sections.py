"""Truth cross section sigma_LL and its MC statistical uncertainty on the common test sets.

The truth of the test histograms (LL_true.top, written by polarisation_test.py) is the sum of r_LL * w_UU over the
test events, normalised to the generated events of the test split, 0.2 * n_generated_events. LL_true.top contains no
uncertainty, so it is computed here from the event weights: sqrt(sum (r_LL * w_UU)^2) / (0.2 * n_generated_events).
As in the test, r_LL is the float32 ratio with the clamp of log_target_transform (r_LL <= -1 -> r_LL = -1 + 1e-10),
so that sigma reproduces LL_true.top; the unclamped sum of w_LL is given for comparison.

Output: final_results/powheg/truth_test_sets.csv (in pb and fb), read by final_results/plots/network_comparison.ipynb.

Usage (from polarisation_tagging/, on /scratch where the event files are local):
    python ensemble/truth_cross_sections.py
"""
# %% Imports
import sys

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_clamped_targets import FINAL_RESULTS_DIR, Settings, files_from_settings, read_weights

ORDERS = ["LO", "LOwS", "NLOPS"]


def read_top_totxsec(path: Path) -> float:
    lines = path.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("# totxsec"))
    return float(lines[index + 1].split()[2].replace("D", "E"))


def main():
    rows = []
    for order in ORDERS:
        settings_file = FINAL_RESULTS_DIR / "ffnn" / order / f"{order}_evts" / "run_settings.yaml"
        settings = Settings.load_yaml(settings_file)
        files    = files_from_settings(settings_file, order)
        with ProcessPoolExecutor(max_workers=32) as pool:
            weights = list(pool.map(read_weights, files, chunksize=8))
        w_uu = np.concatenate([w[0] for w in weights])
        w_ll = np.concatenate([w[1] for w in weights])

        split_ratios = list(settings.split_ratios.value)
        subsets = torch.utils.data.random_split(range(len(w_uu)), split_ratios, generator=torch.Generator().manual_seed(settings.seed.value))
        test = np.asarray(subsets[2].indices)

        # Same arithmetic as the test: float32 ratio and w_UU, clamp of 1 + r_LL at 1e-10.
        r     = torch.clamp(torch.tensor(w_ll[test] / w_uu[test], dtype=torch.float32) + 1.0, min=1e-10) - 1.0
        w     = (r * torch.tensor(w_uu[test], dtype=torch.float32)).double().numpy()
        n_gen = split_ratios[2] * settings.n_generated_events.value

        sigma, unc = w.sum() / n_gen, np.sqrt((w ** 2).sum()) / n_gen
        ll_true    = read_top_totxsec(FINAL_RESULTS_DIR / "powheg" / order / "LL_true.top")
        rows.append({"test_order": order, "n_test": len(test), "n_generated_test": n_gen,
                     "sigma_pb": sigma, "sigma_unc_pb": unc, "sigma_fb": 1e3 * sigma, "sigma_unc_fb": 1e3 * unc,
                     "sigma_unclamped_pb": w_ll[test].sum() / n_gen, "LL_true_top_pb": ll_true,
                     "rel_diff_to_LL_true_top": sigma / ll_true - 1})

    table = pd.DataFrame(rows)
    table.to_csv(FINAL_RESULTS_DIR / "powheg" / "truth_test_sets.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
