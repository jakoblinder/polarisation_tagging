"""Summarise the tests of the controls and the ensemble replicas on the common test sets (run_ensemble.py --mode test).

CONSERVATIVE: every replica is trained on 1/10 of the nominal training data, so the replica spread overestimates the
uncertainty of the production networks. All outputs are labelled accordingly.

Reads, for every test directory in final_results/ensemble/test_overview.json, the test MSE (unweighted, log space,
"Avg (per batch) test loss" in ml_events_utils.log) and the histograms in LL_pred.top / LL_true.top. For ParticleNet
NLOPS there are no production weights; its control histograms are the production final_results/pn/NLOPS/NLOPS.top
(no test MSE available).

Output in final_results/ensemble/report/:
  test_summary.csv                       per network: test MSE, sigma_LL pred/true, replica/bunch/seed
  band_<arch>_<train>_<test>.csv         per observable and bin: true, control, replica mean/std/min/max,
                                         pull = (mean - control) / (std / sqrt(n)), relative band width std/mean
  SUMMARY.md                             check 1 (test MSE of the replicas vs. the control) and check 2 (offset of
                                         the replica mean from the control) per (arch, train, test)

Usage (from polarisation_tagging/):
    python ensemble/summarise_tests.py
"""
# %% Imports
import json
import re

from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR    = PACKAGE_DIR / "final_results" / "ensemble"
REPORT_DIR  = TEST_DIR / "report"
LABEL       = "conservative band: 10 replicas x ~470k events (1/10 of the nominal training data)"

# Observables of LL_pred.top used for the band (r_LL histograms of the predictions themselves are left out).
SKIP_HISTOGRAMS = {"r_LL", "r_LL_count"}
# Flags (stated in SUMMARY.md): replica test MSE above this multiple of the control; |pull| of the total cross section.
MSE_RATIO_FLAG  = 1.5
PULL_FLAG       = 2.0


def read_top(path: Path) -> dict[str, np.ndarray]:
    """Histograms of a .top file: name -> array of rows (low edge, high edge, value, uncertainty)."""
    histograms, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            name = line.split()[1]
            histograms[name] = []
        elif line.strip() and name:
            histograms[name].append([float(x.replace("D", "E")) for x in line.split()])
    return {k: np.array(v) for k, v in histograms.items()}


def test_loss(test_dir: Path) -> float:
    match = re.findall(r"Avg \(per batch\) test loss:\s*([0-9.eE+-]+)", (test_dir / "ml_events_utils.log").read_text(errors="ignore"))
    return float(match[-1]) if match else np.nan


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_DIR / "test_overview.json") as handle:
        overview = pd.DataFrame(json.load(handle))

    rows, verdicts = [], []
    for (arch, train, test), group in overview.groupby(["arch", "train", "test"], sort=False):
        nets = {}
        for _, net in group.iterrows():
            nets[net["network"]] = {"pred": read_top(Path(net["dir"]) / "LL_pred.top"), "true": read_top(Path(net["dir"]) / "LL_true.top"),
                                    "mse": test_loss(Path(net["dir"])), "replica": net["replica"], "bunch": net["bunch"]}
        true = next(iter(nets.values()))["true"]
        if "control" not in nets:   # ParticleNet NLOPS: production histograms only
            nets["control"] = {"pred": read_top(PACKAGE_DIR / "final_results" / "pn" / train / f"{test}.top"), "mse": np.nan,
                               "replica": -1, "bunch": -1, "true": true}
        for name, net in nets.items():
            rows.append({"arch": arch, "train": train, "test": test, "network": name, "replica": net["replica"], "bunch": net["bunch"],
                         "test_mse": net["mse"], "sigma_pred_over_true": net["pred"]["totxsec"][0, 2] / true["totxsec"][0, 2]})

        replicas = [n for n in nets if n != "control"]
        band_rows = []
        for hist in nets["control"]["pred"]:
            if hist in SKIP_HISTOGRAMS:
                continue
            reps = np.array([nets[r]["pred"][hist][:, 2] for r in replicas])
            ctrl = nets["control"]["pred"][hist][:, 2]
            mean, std = reps.mean(axis=0), reps.std(axis=0, ddof=1)
            for i, (lo, hi) in enumerate(nets["control"]["pred"][hist][:, :2]):
                band_rows.append({"observable": hist, "bin_low": lo, "bin_high": hi, "true": true[hist][i, 2], "control": ctrl[i],
                                  "replica_mean": mean[i], "replica_std": std[i], "replica_min": reps[:, i].min(), "replica_max": reps[:, i].max(),
                                  "pull": (mean[i] - ctrl[i]) / (std[i] / np.sqrt(len(replicas))) if std[i] > 0 else np.nan,
                                  "rel_band_width": std[i] / mean[i] if mean[i] != 0 else np.nan})
        band = pd.DataFrame(band_rows)
        band.insert(0, "label", LABEL)
        band.to_csv(REPORT_DIR / f"band_{arch}_{train}_{test}.csv", index=False)

        tot = band[band.observable == "totxsec"].iloc[0]
        rep_mse = np.array([nets[r]["mse"] for r in replicas])
        weight = np.abs(band.control) * (band.observable != "totxsec")
        verdicts.append({"arch": arch, "train": train, "test": test, "n_replicas": len(replicas),
                         "control_mse": nets["control"]["mse"], "replica_mse_mean": rep_mse.mean(), "replica_mse_min": rep_mse.min(),
                         "mse_ratio": rep_mse.mean() / nets["control"]["mse"],
                         "sigma_control_over_true": tot.control / tot.true, "sigma_replica_mean_over_true": tot.replica_mean / tot.true,
                         "sigma_replica_rel_std": tot.replica_std / tot.replica_mean, "sigma_pull": tot.pull,
                         "bins_median_abs_pull": band.loc[band.observable != "totxsec", "pull"].abs().median(),
                         "bins_frac_pull_gt2": (band.loc[band.observable != "totxsec", "pull"].abs() > 2).mean(),
                         "bins_weighted_mean_rel_band_width": np.average(band.rel_band_width.fillna(0), weights=weight)})

    pd.DataFrame(rows).to_csv(REPORT_DIR / "test_summary.csv", index=False)
    verdicts = pd.DataFrame(verdicts)
    verdicts.to_csv(REPORT_DIR / "checks.csv", index=False)

    lines = [f"# Ensemble tests on the common test sets\n", f"**{LABEL}.** The replica spread overestimates the uncertainty "
             "of the production networks.\n", "Test MSE: unweighted MSE of log(1 + r_LL) on the common test set "
             "(seed-42 test split of the production files). At NLOPS it contains the 4 clamped targets (r_LL <= -1), "
             "which add ~2.2e-3 to every network.\n",
             f"Flags: check 1 if the mean replica test MSE exceeds {MSE_RATIO_FLAG} x the control; "
             f"check 2 if |pull| of sigma_LL > {PULL_FLAG}, pull = (replica mean - control) / (replica std / sqrt(n)).\n",
             "| arch | train | test | control MSE | replica MSE (mean) | ratio | check 1 | sigma control/true | sigma replica mean/true | sigma rel. spread | sigma pull | check 2 | bins: median abs pull | bins: frac abs pull > 2 |",
             "|:--|:--|:--|--:|--:|--:|:--:|--:|--:|--:|--:|:--:|--:|--:|"]
    for _, v in verdicts.iterrows():
        flag1 = "n/a" if np.isnan(v.mse_ratio) else ("FLAG" if v.mse_ratio > MSE_RATIO_FLAG else "ok")
        flag2 = "FLAG" if abs(v.sigma_pull) > PULL_FLAG else "ok"
        lines.append(f"| {v.arch} | {v.train} | {v.test} | {v.control_mse:.3e} | {v.replica_mse_mean:.3e} | {v.mse_ratio:.2f} | {flag1} | "
                     f"{v.sigma_control_over_true:.4f} | {v.sigma_replica_mean_over_true:.4f} | {v.sigma_replica_rel_std:.2%} | "
                     f"{v.sigma_pull:+.1f} | {flag2} | {v.bins_median_abs_pull:.1f} | {v.bins_frac_pull_gt2:.0%} |")
    (REPORT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
