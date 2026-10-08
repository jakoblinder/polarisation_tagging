"""Create the settings to redo the ParticleNet production trainings at LO and LOwS on the same 250 event files as the
FFNN and Autoencoder production runs.

The ParticleNet LO and LOwS run_settings.yaml (final_results/pn/<order>/run_settings.yaml) list 200 event files, while
their test histograms were made on the test split of 250 files (943,496 / 952,402 events, the FFNN/AE test sets).
If the networks were trained on 200 files, about half of those test events were part of their training split. The
retraining removes this ambiguity: the settings are taken over unchanged, apart from
  - mlfiles:            pwgevents-0001..0250 (as for FFNN/AE), n_generated_events = 250 * 4e4 = 1e7 accordingly,
  - standardise: false: no effect on ParticleNet_best (models.py drops stat_norm), as for the ensemble replicas,
  - the output paths, histogram_dir and gpu.
The seed stays 42, so the train/validation/test split is the same as for FFNN/AE.

Usage (from polarisation_tagging/, on /scratch):
    python ensemble/make_pn_production_rerun.py
    python polarisation_train.py full_runs/ParticleNet/<order>/run_settings.yaml     # inside tmux

After the trainings, --cross-eval writes the settings of the evaluations on the higher accuracies, as for the FFNN/AE
production runs (LO on LOwS and NLOPS, LOwS on NLOPS): the trained run's settings with the event files and POWHEG
histograms of the target sample (showered: true for NLOPS), the trained weights and seed (inputdir), into
full_runs/ParticleNet/<train>/<train>_with_<test>_evts/run_settings.yaml:
    python ensemble/make_pn_production_rerun.py --cross-eval
    cd full_runs/ParticleNet/<train>/<train>_with_<test>_evts && python ../../../../polarisation_test.py run_settings.yaml
"""
# %% Imports
import argparse
import sys

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))

from ml_events_utils import Settings

ML_FILES_DIR      = PACKAGE_DIR.parent / "ML_FILES"
FINAL_RESULTS_DIR = PACKAGE_DIR / "final_results"
RERUN_DIR         = PACKAGE_DIR / "full_runs" / "ParticleNet"

ORDERS   = {"LO": "UU_LO", "LOwS": "UU_LOwS"}
N_FILES  = 250
GENERATED_PER_FILE = 4e4

# Settings expected to differ from the ParticleNet production settings.
OVERRIDDEN_KEYS = {"mlfiles", "n_generated_events", "standardise", "outputdir", "model_dir", "inputdir",
                   "model_weight_file", "histogram_dir", "gpu"}


# Cross-evaluations: training order -> test orders, and the test samples (files, n_generated_events, showered).
CROSS_EVALUATIONS = {"LO": ["LOwS", "NLOPS"], "LOwS": ["NLOPS"]}
TEST_SAMPLES = {
    "LOwS":  {"files": [ML_FILES_DIR / "UU_LOwS" / f"pwgevents-{i:04d}.ml" for i in range(1, 251)],
              "n_generated_events": int(250 * 4e4), "showered": False},
    "NLOPS": {"files": [ML_FILES_DIR / "UU_NLO" / f"output_shower_events-{i:04d}.ml" for i in range(1, 1001)],
              "n_generated_events": int(1000 * 1e4), "showered": True},
}


def make_cross_eval_configs():
    for train, tests in CROSS_EVALUATIONS.items():
        traindir = RERUN_DIR / train
        trained  = Settings.load_yaml(traindir / "run_settings.yaml")   # as written by the training
        weights  = traindir / f"{trained.model.value}_model_weights_best.pt"
        if not weights.exists() or not (traindir / "training_seed.txt").exists():
            raise FileNotFoundError(f"Training in {traindir} has not finished.")
        for test in tests:
            sample  = TEST_SAMPLES[test]
            evaldir = traindir / f"{train}_with_{test}_evts"
            settings = trained.clone()
            settings.set("mlfiles",            sample["files"],                    overwrite=True)
            settings.set("histogram_dir",      sample["files"][0].parent,          overwrite=True)
            settings.set("n_generated_events", sample["n_generated_events"],       overwrite=True)
            settings.set("showered",           sample["showered"],                 overwrite=True)
            settings.set("inputdir",           traindir,                           overwrite=True)
            settings.set("model_weight_file",  weights,                            overwrite=True)
            settings.set("outputdir",          evaldir,                            overwrite=True)
            evaldir.mkdir(parents=True, exist_ok=True)
            settings.dump_yaml(evaldir / "run_settings.yaml")
            print(f"{train} -> {test}: {len(sample['files'])} files, showered {sample['showered']} -> {evaldir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cross-eval", action="store_true", help="Write the cross-evaluation settings of the finished trainings.")
    if parser.parse_args().cross_eval:
        make_cross_eval_configs()
        return

    for order, sample_dir in ORDERS.items():
        prod_file = FINAL_RESULTS_DIR / "pn" / order / "run_settings.yaml"
        prod      = Settings.load_yaml(prod_file)
        outputdir = RERUN_DIR / order
        files     = [ML_FILES_DIR / sample_dir / f"pwgevents-{i:04d}.ml" for i in range(1, N_FILES + 1)]
        missing   = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing event files: {missing[:3]} ...")

        settings = prod.clone()
        settings.set("mlfiles",            files,                                                  overwrite=True)
        settings.set("n_generated_events", int(N_FILES * GENERATED_PER_FILE),                      overwrite=True)
        settings.set("standardise",        False,                                                  overwrite=True)
        settings.set("outputdir",          outputdir,                                              overwrite=True)
        settings.set("model_dir",          outputdir,                                              overwrite=True)
        settings.set("inputdir",           outputdir,                                              overwrite=True)
        settings.set("model_weight_file",  outputdir / f"{prod.model.value}_model_weights_best.pt", overwrite=True)
        settings.set("histogram_dir",      ML_FILES_DIR / sample_dir,                              overwrite=True)
        settings.set("gpu",                0,                                                      overwrite=True)

        changed = {key for key in set(prod.keys()) | set(settings.keys())
                   if key not in OVERRIDDEN_KEYS and settings[key].value != prod[key].value}
        if changed:
            raise RuntimeError(f"Unexpected changes with respect to {prod_file}: {sorted(changed)}")

        outputdir.mkdir(parents=True, exist_ok=True)
        settings.dump_yaml(outputdir / "run_settings.yaml")
        print(f"{order}: {len(files)} files ({files[0].name} .. {files[-1].name}), seed {settings.seed.value} -> {outputdir}")


if __name__ == "__main__":
    main()
