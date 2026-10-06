"""Create the run settings of the ensemble replicas used for the (conservative) uncertainty band.

Every replica is a copy of the production run settings in final_results/<arch>/<order>/<order>_evts/run_settings.yaml,
in which only the event files (one bunch of the so far unused second half of the sample), the seed (initialisation,
batch shuffling and train/validation/test split), n_generated_events and the output directory are changed.

Replica r is trained on bunch r with seed SEED_OFFSET + r, for every architecture, such that the replicas of different
architectures are paired.

Usage (from polarisation_tagging/):
    python ensemble/make_ensemble_configs.py [--archs ffnn autoencoder] [--orders LO LOwS NLOPS] [--check-init]
"""
# %% Imports
import argparse
import hashlib
import json
import sys

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from braceexpand import braceexpand

PACKAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))

from ml_events_utils import Settings

# %% Ensemble definition
ML_FILES_DIR      = PACKAGE_DIR.parent / "ML_FILES"
FINAL_RESULTS_DIR = PACKAGE_DIR / "final_results"
ENSEMBLE_RUNS_DIR = PACKAGE_DIR / "ensemble_runs"

N_REPLICAS  = 10
SEED_OFFSET = 1000

# Production results directory per architecture.
ARCHS = ["ffnn", "autoencoder"]

# Event file directory, file prefix, first unused file, files per bunch and generated events per file for each order.
# showered: compare with the showered POWHEG histograms (pwgoutput_py8_histos-*) instead of the LHE level ones. It only
# affects the POWHEG reference histograms in the test plots. The production NLOPS training used showered: false, while
# all production evaluations on NLOPS events of the LO and LOwS trainings used showered: true; the latter is used for
# all NLOPS replicas.
ORDERS = {
    "LO":    {"dir": "UU_LO",   "prefix": "pwgevents",            "first_file": 251,  "files_per_bunch": 25,  "generated_per_file": 4e4, "showered": False},
    "LOwS":  {"dir": "UU_LOwS", "prefix": "pwgevents",            "first_file": 251,  "files_per_bunch": 25,  "generated_per_file": 4e4, "showered": False},
    "NLOPS": {"dir": "UU_NLO",  "prefix": "output_shower_events", "first_file": 1001, "files_per_bunch": 100, "generated_per_file": 1e4, "showered": True},
}

# The only settings that differ between a replica and its production run.
OVERRIDDEN_KEYS = {"mlfiles", "seed", "outputdir", "n_generated_events", "histogram_dir", "model_dir", "gpu", "showered"}


def production_settings_file(arch: str, order: str) -> Path:
    return FINAL_RESULTS_DIR / arch / order / f"{order}_evts" / "run_settings.yaml"


def replica_dir(arch: str, order: str, replica: int) -> Path:
    return ENSEMBLE_RUNS_DIR / arch / order / f"rep{replica:02d}_bunch{replica:02d}"


def bunch_files(order: str, bunch: int) -> list[Path]:
    """Event files of the given bunch (absolute paths, sorted)."""
    info  = ORDERS[order]
    first = info["first_file"] + bunch * info["files_per_bunch"]
    return [ML_FILES_DIR / info["dir"] / f"{info['prefix']}-{i:04d}.ml" for i in range(first, first + info["files_per_bunch"])]


def production_files(settings: Settings, order: str) -> list[Path]:
    """Event files of the production training, resolved by their names in the ML_FILES directory of the order."""
    files = []
    for pattern in settings.mlfiles.value:
        files += [ML_FILES_DIR / ORDERS[order]["dir"] / Path(p).name for p in braceexpand(Path(pattern).name)]
    return sorted(files)


def count_events(eventfile: Path) -> int:
    with open(eventfile, "rb") as handle:
        return handle.read().count(b"<event>")


def initial_weights_hash(settings: Settings, n_train: int) -> str:
    """Replay the random number generation of polarisation_train.run_training up to build_model and hash the initial weights.

    Before the model is built, the global torch RNG is seeded and consumed only by iterating once over the shuffled
    training DataLoader (the random_split uses its own generator), which does not depend on the event content.
    """
    import torch
    from torch.utils.data import DataLoader
    from ml_events_utils.models import build_model

    torch.manual_seed(settings.seed.value)
    probe_loader = DataLoader(torch.zeros(n_train, 1), batch_size=settings.batch_size.value, shuffle=True)
    next(iter(probe_loader))
    model = build_model(settings, 16)

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_configs(arch: str, order: str, check_init: bool, events_per_file: dict) -> list[dict]:
    prod_file = production_settings_file(arch, order)
    prod      = Settings.load_yaml(prod_file)
    prod_set  = set(production_files(prod, order))

    used_files = set()
    infos      = []
    for replica in range(N_REPLICAS):
        files = bunch_files(order, replica)
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing event files for {order} bunch {replica}: {missing}")
        if prod_set & set(files):
            raise RuntimeError(f"{order} bunch {replica} overlaps with the production training files.")
        if used_files & set(files):
            raise RuntimeError(f"{order} bunch {replica} overlaps with a previous bunch.")
        used_files |= set(files)

        outputdir = replica_dir(arch, order, replica)
        settings  = prod.clone()
        settings.set("mlfiles",            files,                                                 overwrite=True)
        settings.set("seed",               SEED_OFFSET + replica,                                 overwrite=True)
        settings.set("outputdir",          outputdir,                                             overwrite=True)
        settings.set("model_dir",          outputdir,                                             overwrite=True)
        settings.set("histogram_dir",      ML_FILES_DIR / ORDERS[order]["dir"],                   overwrite=True)
        settings.set("n_generated_events", int(len(files) * ORDERS[order]["generated_per_file"]), overwrite=True)
        settings.set("gpu",                0,                                                     overwrite=True)
        settings.set("showered",           ORDERS[order]["showered"],                             overwrite=True)

        # Everything apart from the overridden keys has to be identical to the production run.
        changed = {key for key in set(prod.keys()) | set(settings.keys())
                   if key not in OVERRIDDEN_KEYS and settings[key].value != prod[key].value}
        if changed:
            raise RuntimeError(f"Unexpected changes with respect to {prod_file}: {sorted(changed)}")

        n_events = sum(events_per_file[f] for f in files)
        info = {
            "arch":               arch,
            "order":              order,
            "model":              settings.model.value,
            "replica":            replica,
            "bunch":              replica,
            "seed":               settings.seed.value,
            "files":              [str(f) for f in files],
            "n_events":           n_events,
            "n_generated_events": settings.n_generated_events.value,
            "split_ratios":       list(settings.split_ratios.value),
            "production_settings": str(prod_file),
            "label":              "conservative: replica trained on 1/10 of the nominal training data",
        }
        if check_init:
            # Same rounding as torch.utils.data.random_split for fractional lengths.
            n_train = int(n_events * settings.split_ratios.value[0])
            n_rest  = n_events - sum(int(n_events * r) for r in settings.split_ratios.value)
            info["initial_weights_sha256"] = initial_weights_hash(settings, n_train + (1 if n_rest > 0 else 0))

        outputdir.mkdir(parents=True, exist_ok=True)
        settings.dump_yaml(outputdir / "run_settings.yaml")
        with open(outputdir / "replica_info.json", "w") as handle:
            json.dump(info, handle, indent=2)
        infos.append(info)
        print(f"{arch:12s} {order:6s} replica {replica:2d}: {n_events:7d} events, seed {info['seed']} -> {outputdir}")

    if check_init:
        hashes = [info["initial_weights_sha256"] for info in infos]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError(f"{arch} {order}: initial weights are not pairwise different.")
        print(f"{arch:12s} {order:6s}: initial weights of all {len(hashes)} replicas are pairwise different.")
    return infos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs",  nargs="+", default=ARCHS,        choices=ARCHS)
    parser.add_argument("--orders", nargs="+", default=list(ORDERS), choices=list(ORDERS),
                        help="LOwS (LO+Sud.) is always added, every run has to come with its LO+Sud. counterpart.")
    parser.add_argument("--check-init", action="store_true", help="Replay the initialisation and check that the initial weights differ.")
    args = parser.parse_args()

    if "LOwS" not in args.orders:
        print("Adding LOwS (LO+Sud.) to the orders, it is always run as well.")
        args.orders.append("LOwS")

    eventfiles = sorted({f for order in args.orders for bunch in range(N_REPLICAS) for f in bunch_files(order, bunch)})
    with ThreadPoolExecutor(max_workers=32) as pool:
        events_per_file = dict(zip(eventfiles, pool.map(count_events, eventfiles)))

    summary = [info for arch in args.archs for order in args.orders
               for info in make_configs(arch, order, args.check_init, events_per_file)]

    # Merge with the replicas of an earlier call, such that a partial rerun does not drop the others from the overview.
    overview_file = ENSEMBLE_RUNS_DIR / "ensemble_overview.json"
    if overview_file.exists():
        with open(overview_file) as handle:
            new_keys = {(i["arch"], i["order"], i["replica"]) for i in summary}
            summary  = [i for i in json.load(handle) if (i["arch"], i["order"], i["replica"]) not in new_keys] + summary
    with open(overview_file, "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
