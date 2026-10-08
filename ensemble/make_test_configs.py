"""Create the settings to test the production networks ("control") and the ensemble replicas on the common test sets,
run with the unchanged polarisation_test.py (via run_ensemble.py --mode test).

The common test set of a test sample (LO, LOwS, NLOPS) is the test split of its production training files with seed 42
(the same for FFNN, AE and ParticleNet), and every network trained on the same or a lower accuracy is tested on it, as in
the production results: LO on LO/LOwS/NLOPS, LOwS on LOwS/NLOPS, NLOPS on NLOPS. Networks per (arch, train):
  - control: the production network (FFNN/AE: final_results/<arch>/<train>/<train>_evts, ParticleNet LO/LOwS: the
    250-file reruns in full_runs/ParticleNet/<train>; ParticleNet NLOPS: no weights, only final_results/pn/NLOPS/*.top),
  - the 10 replicas in ensemble_runs/<arch>/<train>/rep<RR>_bunch<BB> (conservative: trained on 1/10 of the data).

Each test directory final_results/ensemble/<arch>/<train>/<test>_evts/<control|repRR_bunchRR>/ gets
  - run_settings.yaml: the network's settings with the test sample's production files and POWHEG histograms
    (showered: true for NLOPS), n_generated_events = 1e7, the absolute model_weight_file, inputdir = outputdir = itself,
  - training_seed.txt = 42: polarisation_test.py takes the seed of the split from <inputdir>/training_seed.txt, so the
    replicas (trained with seeds 1000 + r) are tested on the production test split as well.
All test directories are listed in final_results/ensemble/test_overview.json.

Usage (from polarisation_tagging/, on /scratch):
    python ensemble/make_test_configs.py
"""
# %% Imports
import json
import sys

from pathlib import Path

from braceexpand import braceexpand

PACKAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))

from ml_events_utils import Settings

ML_FILES_DIR      = PACKAGE_DIR.parent / "ML_FILES"
FINAL_RESULTS_DIR = PACKAGE_DIR / "final_results"
ENSEMBLE_RUNS_DIR = PACKAGE_DIR / "ensemble_runs"
TEST_DIR          = FINAL_RESULTS_DIR / "ensemble"

ORDERS       = ["LO", "LOwS", "NLOPS"]
ORDER_DIRS   = {"LO": "UU_LO", "LOwS": "UU_LOwS", "NLOPS": "UU_NLO"}
ARCHS        = ["ffnn", "autoencoder", "pn"]
SEED         = 42
SPLIT_RATIOS = [0.6, 0.2, 0.2]
N_GENERATED  = int(1e7)     # generated events of the production files (250 * 4e4 at LO/LOwS, 1000 * 1e4 at NLOPS)


def production_files(test_order: str) -> list[Path]:
    """Production training files of the test sample; identical for all architectures (checked)."""
    file_lists = []
    for settings_file in [FINAL_RESULTS_DIR / arch / test_order / f"{test_order}_evts" / "run_settings.yaml" for arch in ("ffnn", "autoencoder")] \
                         + [FINAL_RESULTS_DIR / "pn" / test_order / "run_settings.yaml"]:
        settings = Settings.load_yaml(settings_file)
        if settings.seed.value != SEED or list(settings.split_ratios.value) != SPLIT_RATIOS:
            raise RuntimeError(f"Unexpected seed or split ratios in {settings_file}")
        files = set()
        for pattern in settings.mlfiles.value:
            files |= {ML_FILES_DIR / ORDER_DIRS[test_order] / Path(p).name for p in braceexpand(Path(pattern).name)}
        file_lists.append(sorted(files))
    if any(files != file_lists[0] for files in file_lists):
        raise RuntimeError(f"The production file lists of {test_order} differ between the architectures.")
    missing = [f for f in file_lists[0] if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing event files: {missing[:3]} ...")
    return file_lists[0]


def networks(arch: str, train: str) -> list[dict]:
    """Control (if its weights exist) and replicas: name, replica, bunch, settings file and weight file."""
    if arch == "pn":
        control_dir = PACKAGE_DIR / "full_runs" / "ParticleNet" / train if train != "NLOPS" else None
        settings_file = control_dir / "run_settings.yaml" if control_dir else None
    else:
        control_dir = FINAL_RESULTS_DIR / arch / train / f"{train}_evts"
        settings_file = control_dir / "run_settings.yaml"
    nets = []
    if control_dir is not None:
        model = Settings.load_yaml(settings_file).model.value
        nets.append({"network": "control", "replica": -1, "bunch": -1, "settings": settings_file,
                     "weights": control_dir / f"{model}_model_weights_best.pt"})

    with open(ENSEMBLE_RUNS_DIR / "ensemble_overview.json") as handle:
        infos = sorted((i for i in json.load(handle) if i["arch"] == arch and i["order"] == train), key=lambda i: i["replica"])
    for info in infos:
        rundir = ENSEMBLE_RUNS_DIR / arch / train / f"rep{info['replica']:02d}_bunch{info['bunch']:02d}"
        nets.append({"network": rundir.name, "replica": info["replica"], "bunch": info["bunch"],
                     "settings": rundir / "run_settings.yaml", "weights": rundir / f"{info['model']}_model_weights_best.pt"})
    for net in nets:
        if not net["weights"].exists():
            raise FileNotFoundError(f"Missing weights {net['weights']}")
    return nets


def main():
    overview = []
    for test in ORDERS:
        files = production_files(test)
        for arch in ARCHS:
            for train in ORDERS[:ORDERS.index(test) + 1]:
                for net in networks(arch, train):
                    testdir  = TEST_DIR / arch / train / f"{test}_evts" / net["network"]
                    settings = Settings.load_yaml(net["settings"])
                    settings.set("mlfiles",            files,                         overwrite=True)
                    settings.set("histogram_dir",      ML_FILES_DIR / ORDER_DIRS[test], overwrite=True)
                    settings.set("n_generated_events", N_GENERATED,                   overwrite=True)
                    settings.set("showered",           test == "NLOPS",               overwrite=True)
                    settings.set("seed",               SEED,                          overwrite=True)
                    settings.set("split_ratios",       SPLIT_RATIOS,                  overwrite=True)
                    settings.set("model_weight_file",  net["weights"],                overwrite=True)
                    settings.set("inputdir",           testdir,                       overwrite=True)
                    settings.set("outputdir",          testdir,                       overwrite=True)
                    settings.set("gpu",                0,                             overwrite=True)
                    testdir.mkdir(parents=True, exist_ok=True)
                    settings.dump_yaml(testdir / "run_settings.yaml")
                    (testdir / "training_seed.txt").write_text(f"{SEED}\n")
                    overview.append({"arch": arch, "train": train, "test": test, "network": net["network"],
                                     "replica": net["replica"], "bunch": net["bunch"], "model": settings.model.value,
                                     "weights": str(net["weights"]), "dir": str(testdir)})
                print(f"{arch:12s} {train:6s} -> {test:6s}: {sum(1 for o in overview if (o['arch'], o['train'], o['test']) == (arch, train, test))} networks")
    with open(TEST_DIR / "test_overview.json", "w") as handle:
        json.dump(overview, handle, indent=2)
    print(f"{len(overview)} tests -> {TEST_DIR / 'test_overview.json'}")


if __name__ == "__main__":
    main()
