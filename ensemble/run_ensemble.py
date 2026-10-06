"""Train the ensemble replicas created by make_ensemble_configs.py with a simple local job queue.

Every replica runs `polarisation_train.py <replica_dir>/run_settings.yaml` in its own directory (so that also
ml_events_utils.log stays separated), with the GPU chosen via CUDA_VISIBLE_DEVICES. The training is mostly CPU bound
(event parsing in the DataLoader), therefore several replicas share one GPU.
Finished replicas are skipped, so the queue can simply be restarted after an interruption.
The state of the queue is written to ensemble_runs/queue_status.json after every change.

Usage (from polarisation_tagging/). Start the queue inside tmux: processes started from the VS Code terminal (also with
nohup/setsid) are killed when the VS Code server restarts.
    python ensemble/run_ensemble.py --dry-run
    tmux new-session -d -s ensemble "source .ml/bin/activate && python ensemble/run_ensemble.py --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 3 > ensemble_runs/queue.log 2>&1"
"""
# %% Imports
import argparse
import json
import os
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

PACKAGE_DIR       = Path(__file__).resolve().parent.parent
ENSEMBLE_RUNS_DIR = PACKAGE_DIR / "ensemble_runs"
TRAIN_SCRIPT      = PACKAGE_DIR / "polarisation_train.py"
STATUS_FILE       = ENSEMBLE_RUNS_DIR / "queue_status.json"


def is_finished(run_dir: Path, model: str) -> bool:
    """A replica is finished, if the training ended normally and the best weights were written."""
    output_log = run_dir / "output.log"
    return ((run_dir / f"{model}_model_weights_best.pt").exists()
            and output_log.exists()
            and "Best validation loss" in output_log.read_text(errors="ignore"))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(jobs: list[dict]) -> None:
    tmp_file = STATUS_FILE.with_suffix(".tmp")
    with open(tmp_file, "w") as handle:
        json.dump([{key: value for key, value in job.items() if key != "process"} for job in jobs], handle, indent=2)
    tmp_file.replace(STATUS_FILE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs",           nargs="+", default=["ffnn", "autoencoder"])
    parser.add_argument("--orders",          nargs="+", default=["LO", "LOwS", "NLOPS"],
                        help="LOwS (LO+Sud.) is always added, every run has to come with its LO+Sud. counterpart.")
    parser.add_argument("--replicas",        nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gpus",            nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--jobs-per-gpu",    type=int, default=3)
    parser.add_argument("--threads-per-job", type=int, default=4, help="OMP/MKL threads per training process.")
    parser.add_argument("--python",          default=sys.executable)
    parser.add_argument("--dry-run",         action="store_true", help="Only list the replicas and their state.")
    args = parser.parse_args()

    if "LOwS" not in args.orders:
        print("Adding LOwS (LO+Sud.) to the orders, it is always run as well.", flush=True)
        args.orders.append("LOwS")

    with open(ENSEMBLE_RUNS_DIR / "ensemble_overview.json") as handle:
        overview = json.load(handle)

    # Order by sample first, then replica, so that complete replica sets become available early.
    jobs = []
    for order in args.orders:
        for replica in args.replicas:
            for arch in args.archs:
                info = next((i for i in overview if (i["arch"], i["order"], i["replica"]) == (arch, order, replica)), None)
                if info is None:
                    raise KeyError(f"No config for {arch} {order} replica {replica}; run make_ensemble_configs.py first.")
                run_dir = ENSEMBLE_RUNS_DIR / arch / order / f"rep{replica:02d}_bunch{info['bunch']:02d}"
                state   = "finished" if is_finished(run_dir, info["model"]) else "pending"
                jobs.append({"arch": arch, "order": order, "replica": replica, "bunch": info["bunch"],
                             "model": info["model"], "run_dir": str(run_dir), "state": state,
                             "gpu": None, "start": None, "end": None, "returncode": None})

    print(f"{now()} {sum(j['state'] == 'pending' for j in jobs)} pending, "
          f"{sum(j['state'] == 'finished' for j in jobs)} already finished.", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job['state']:9s} {job['run_dir']}")
        return

    slots   = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    running = {}
    write_status(jobs)
    try:
        while True:
            # Collect finished processes.
            for slot, job in list(running.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["returncode"] = returncode
                job["end"]        = now()
                job["state"]      = "finished" if returncode == 0 and is_finished(Path(job["run_dir"]), job["model"]) else "failed"
                del job["process"]
                del running[slot]
                print(f"{now()} {job['state']:8s} {job['arch']} {job['order']} replica {job['replica']} (exit {returncode})", flush=True)
                write_status(jobs)

            # Start new processes on free slots.
            pending = [job for job in jobs if job["state"] == "pending"]
            for slot, gpu in enumerate(slots):
                if slot in running or not pending:
                    continue
                job     = pending.pop(0)
                run_dir = Path(job["run_dir"])
                env     = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               OMP_NUM_THREADS=str(args.threads_per_job), MKL_NUM_THREADS=str(args.threads_per_job))
                with open(run_dir / "stdout.log", "w") as stdout:
                    job["process"] = subprocess.Popen([args.python, str(TRAIN_SCRIPT), "run_settings.yaml"],
                                                      cwd=run_dir, env=env, stdout=stdout, stderr=subprocess.STDOUT)
                job.update(state="running", gpu=gpu, start=now(), end=None, returncode=None, pid=job["process"].pid)
                running[slot] = job
                print(f"{now()} started  {job['arch']} {job['order']} replica {job['replica']} on GPU {gpu} (pid {job['pid']})", flush=True)
                write_status(jobs)

            if not running and not any(job["state"] == "pending" for job in jobs):
                break
            time.sleep(30)
    finally:
        # On interruption: stop the running trainings and mark them as pending again for the next start.
        for job in running.values():
            job["process"].terminate()
            del job["process"]
            job["state"] = "pending"
        write_status(jobs)

    failed = [job for job in jobs if job["state"] == "failed"]
    print(f"{now()} done: {sum(j['state'] == 'finished' for j in jobs)} finished, {len(failed)} failed.", flush=True)
    for job in failed:
        print(f"  FAILED {job['run_dir']} (see stdout.log / output.log)", flush=True)


if __name__ == "__main__":
    main()
