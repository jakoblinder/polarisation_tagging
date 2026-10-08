"""Train the ensemble replicas created by make_ensemble_configs.py, or run their tests created by make_test_configs.py,
with a simple local job queue.

--mode train (default): every replica runs `polarisation_train.py <replica_dir>/run_settings.yaml`.
--mode test:            every network (control and replicas) runs `polarisation_test.py <test_dir>/run_settings.yaml`
                        on the common test set of the test sample (final_results/ensemble/test_overview.json).
Each job runs in its own directory (so that also ml_events_utils.log stays separated), with the GPU chosen via
CUDA_VISIBLE_DEVICES. Training and testing are mostly CPU bound (event parsing in the DataLoader), therefore several jobs
share one GPU. Finished jobs are skipped, so the queue can simply be restarted after an interruption.
The state of the queue is written to ensemble_runs/queue_status.json (training) or
final_results/ensemble/queue_status_test.json (tests) after every change.

Usage (from polarisation_tagging/). Start the queue inside tmux: processes started from the VS Code terminal (also with
nohup/setsid) are killed when the VS Code server restarts.
    python ensemble/run_ensemble.py --dry-run
    tmux new-session -d -s ensemble "source .ml/bin/activate && python ensemble/run_ensemble.py --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 3 > ensemble_runs/queue.log 2>&1"
    tmux new-session -d -s ensemble_test "source .ml/bin/activate && python ensemble/run_ensemble.py --mode test --gpus 0 1 2 3 4 --jobs-per-gpu 4 > final_results/ensemble/queue_test.log 2>&1"
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
TEST_DIR          = PACKAGE_DIR / "final_results" / "ensemble"

MODES = {
    "train": {"script": PACKAGE_DIR / "polarisation_train.py", "overview": ENSEMBLE_RUNS_DIR / "ensemble_overview.json",
              "status": ENSEMBLE_RUNS_DIR / "queue_status.json"},
    "test":  {"script": PACKAGE_DIR / "polarisation_test.py",  "overview": TEST_DIR / "test_overview.json",
              "status": TEST_DIR / "queue_status_test.json"},
}


def is_finished(job: dict) -> bool:
    """A training is finished, if it ended normally and the best weights were written; a test, if its histograms
    were written and the test log reports its completion."""
    run_dir = Path(job["run_dir"])
    if job["mode"] == "train":
        output_log = run_dir / "output.log"
        return ((run_dir / f"{job['model']}_model_weights_best.pt").exists()
                and output_log.exists()
                and "Best validation loss" in output_log.read_text(errors="ignore"))
    test_log = run_dir / "ml_events_utils.log"
    return ((run_dir / "LL_pred.top").exists()
            and test_log.exists()
            and "Testing completed" in test_log.read_text(errors="ignore"))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(jobs: list[dict], status_file: Path) -> None:
    tmp_file = status_file.with_suffix(".tmp")
    with open(tmp_file, "w") as handle:
        json.dump([{key: value for key, value in job.items() if key != "process"} for job in jobs], handle, indent=2)
    tmp_file.replace(status_file)


def collect_jobs(args, overview: list[dict]) -> list[dict]:
    """Jobs ordered by sample first, then replica, so that complete replica sets become available early."""
    jobs = []
    if args.mode == "train":
        for order in args.orders:
            for replica in args.replicas:
                for arch in args.archs:
                    info = next((i for i in overview if (i["arch"], i["order"], i["replica"]) == (arch, order, replica)), None)
                    if info is None:
                        raise KeyError(f"No config for {arch} {order} replica {replica}; run make_ensemble_configs.py first.")
                    jobs.append({"mode": "train", "arch": arch, "order": order, "replica": replica, "bunch": info["bunch"],
                                 "model": info["model"], "label": f"{arch} {order} replica {replica}",
                                 "run_dir": str(ENSEMBLE_RUNS_DIR / arch / order / f"rep{replica:02d}_bunch{info['bunch']:02d}")})
    else:
        for info in overview:
            if info["arch"] in args.archs and info["train"] in args.orders and (info["replica"] in args.replicas or info["replica"] < 0):
                jobs.append({"mode": "test", "arch": info["arch"], "order": info["train"], "test": info["test"],
                             "replica": info["replica"], "bunch": info["bunch"], "model": info["model"],
                             "label": f"{info['arch']} {info['train']}->{info['test']} {info['network']}", "run_dir": info["dir"]})
    for job in jobs:
        job.update(state="finished" if is_finished(job) else "pending", gpu=None, start=None, end=None, returncode=None)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",            default="train", choices=list(MODES))
    parser.add_argument("--archs",           nargs="+", default=["ffnn", "autoencoder", "pn"])
    parser.add_argument("--orders",          nargs="+", default=["LO", "LOwS", "NLOPS"],
                        help="Training orders. LOwS (LO+Sud.) is always added, every run has to come with its LO+Sud. counterpart.")
    parser.add_argument("--replicas",        nargs="+", type=int, default=list(range(10)), help="Replicas (tests: controls always included).")
    parser.add_argument("--gpus",            nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--jobs-per-gpu",    type=int, default=3)
    parser.add_argument("--threads-per-job", type=int, default=4, help="OMP/MKL threads per process.")
    parser.add_argument("--python",          default=sys.executable)
    parser.add_argument("--dry-run",         action="store_true", help="Only list the jobs and their state.")
    args = parser.parse_args()

    if "LOwS" not in args.orders:
        print("Adding LOwS (LO+Sud.) to the orders, it is always run as well.", flush=True)
        args.orders.append("LOwS")

    mode = MODES[args.mode]
    with open(mode["overview"]) as handle:
        jobs = collect_jobs(args, json.load(handle))

    print(f"{now()} {sum(j['state'] == 'pending' for j in jobs)} pending, "
          f"{sum(j['state'] == 'finished' for j in jobs)} already finished.", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job['state']:9s} {job['run_dir']}")
        return

    slots   = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    running = {}
    write_status(jobs, mode["status"])
    try:
        while True:
            # Collect finished processes.
            for slot, job in list(running.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["returncode"] = returncode
                job["end"]        = now()
                job["state"]      = "finished" if returncode == 0 and is_finished(job) else "failed"
                del job["process"]
                del running[slot]
                print(f"{now()} {job['state']:8s} {job['label']} (exit {returncode})", flush=True)
                write_status(jobs, mode["status"])

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
                    job["process"] = subprocess.Popen([args.python, str(mode["script"]), "run_settings.yaml"],
                                                      cwd=run_dir, env=env, stdout=stdout, stderr=subprocess.STDOUT)
                job.update(state="running", gpu=gpu, start=now(), end=None, returncode=None, pid=job["process"].pid)
                running[slot] = job
                print(f"{now()} started  {job['label']} on GPU {gpu} (pid {job['pid']})", flush=True)
                write_status(jobs, mode["status"])

            if not running and not any(job["state"] == "pending" for job in jobs):
                break
            time.sleep(30)
    finally:
        # On interruption: stop the running jobs and mark them as pending again for the next start.
        for job in running.values():
            job["process"].terminate()
            del job["process"]
            job["state"] = "pending"
        write_status(jobs, mode["status"])

    failed = [job for job in jobs if job["state"] == "failed"]
    print(f"{now()} done: {sum(j['state'] == 'finished' for j in jobs)} finished, {len(failed)} failed.", flush=True)
    for job in failed:
        print(f"  FAILED {job['run_dir']} (see stdout.log / output.log / ml_events_utils.log)", flush=True)


if __name__ == "__main__":
    main()
