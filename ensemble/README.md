# Ensemble training for a conservative uncertainty band

Ten independent replicas of each production network are trained on disjoint ~470k-event bunches. The spread of their per-event $r_{\rm LL}$ predictions gives an uncertainty band on the production predictions in [`final_results/`](../final_results).

> **Interpretation.** Each replica sees only 1/10 of the nominal training data. The replica spread therefore **overestimates** the uncertainty of the production models. Every plot or table built from it is labelled *"conservative: replicas trained on 1/10 of the nominal training data"*.

## Status

| Step | Script | State |
|:-----|:-------|:------|
| 1. Replica configs | [`make_ensemble_configs.py`](make_ensemble_configs.py) | done: 60 configs written and checked |
| 2. Training queue | [`run_ensemble.py`](run_ensemble.py) | written, **not started** (waiting for review) |
| 3. Predictions on the common test sets (incl. cross-evaluations) | `predict_common_testset.py` | to do |
| 4. Diagnostics, band plots, summary | `analyse_ensemble.py` | to do |

## Scope

- **Architectures:** FFNN (`FFNN_paper_4extraLayers_BatchNorm`, [`final_results/ffnn`](../final_results/ffnn)) and AE (`FFNN_EMB_1024_512_256_128_64_BatchNorm`, [`final_results/autoencoder`](../final_results/autoencoder)).
- **ParticleNet is excluded for now.** It was trained by someone else and its run settings are not available. The scripts take the architecture as a parameter, so it can be added once the settings are known.
- **Samples:** LO, LO+Sud. (`LOwS`) and NLO+PS (`NLOPS`). LO+Sud. is compulsory: both scripts add it when it is not requested, so every set of runs comes with its LO+Sud. counterpart.
- **Total:** 2 architectures × 3 samples × 10 replicas = **60 trainings**.

## Data

The production networks were trained on the first half of each sample (~4.7M events). The replicas use the so far unused second half, split into 10 disjoint bunches. Replica $r$ is trained on bunch $b = r$ only. Within each bunch the usual 60/20/20 train/validation/test split is applied.

| Sample | Production files | Ensemble files | Events/file | Bunch $b = 0\dots9$ | Events/bunch |
|:-------|:-----------------|:---------------|------------:|:--------------------|-------------:|
| LO | `UU_LO/pwgevents-0001…0250` | `0251…0500` | ~18.8k | files `0251+25b … 0275+25b` | ~471.5k |
| LO+Sud. | `UU_LOwS/pwgevents-0001…0250` | `0251…0500` | ~19.0k | files `0251+25b … 0275+25b` | ~475.8k |
| NLO+PS | `UU_NLO/output_shower_events-0001…1000` | `1001…2000` | ~4.8k | files `1001+100b … 1100+100b` | ~478.8k |

`n_generated_events` is $25 \cdot 4\cdot10^4 = 100 \cdot 10^4 = 10^6$ per bunch. It only normalises the cross sections in each replica's own test plots.

## Training settings

All hyperparameters are taken unchanged from the production `final_results/<arch>/<order>/<order>_evts/run_settings.yaml`.

| Network | LO: batch size / LR | LO+Sud.: batch size / LR | NLO+PS: batch size / LR |
|:--------|:--------------------|:-------------------------|:------------------------|
| FFNN | 3000 / 2.696e-3 | 4800 / 5.415e-4 | 1000 / 1.593e-3 |
| AE | 4200 / 2.629e-3 | 5000 / 2.649e-3 | 3400 / 3.794e-3 |

Common to all runs:
- **Optimiser:** AdamW (weight decay 1e-4) with `ReduceLROnPlateau` (factor 0.5, patience 3).
- **Early stopping:** patience 35, at most 500 epochs.
- **Input:** lab frame, no standardisation, no penalties.
- **Target and loss:** target $\log(1 + w_{\rm LL}/w_{\rm UU})$. Training loss is the MSE weighted per event by $|\log(1+w_{\rm UU})|$ (`ml_events_utils/train_loop.py`). Validation loss and early stopping use the unweighted MSE.

Only these settings differ from the production run:

| Setting | Replica value |
|:--------|:--------------|
| `mlfiles` | The bunch, as an explicit list of absolute paths. `MLEventsDataset` does not expand brace patterns in absolute paths. |
| `seed` | $1000 + r$. Controls initial weights, batch shuffling and the 60/20/20 split. |
| `outputdir`, `model_dir` | The replica directory. |
| `n_generated_events` | $10^6$ |
| `histogram_dir` | Absolute path to the sample's POWHEG histograms. |
| `gpu` | 0. The GPU is chosen via `CUDA_VISIBLE_DEVICES`. |
| `showered` | `true` for NLO+PS (production training: `false`), unchanged `false` for LO and LO+Sud. Only selects the POWHEG reference histograms of the test plots, see the note in step 3. |

The FFNN and AE replicas with the same $r$ share bunch and seed, so the two architectures can be compared replica by replica.

**Initial weights.** `torch.manual_seed(seed)` ([`polarisation_train.py:71`](../polarisation_train.py)) is called before `build_model` (`:277`), and the models use PyTorch's default initialisation. A different seed therefore means different initial weights. `make_ensemble_configs.py --check-init` replays the random number generation up to `build_model` and confirmed that the 10 initial weight sets of every (architecture, sample) are pairwise different. The hashes are stored in `replica_info.json`.

## Directory layout

```
ensemble/                                   # this directory: scripts + README
ensemble_runs/
├── ensemble_overview.json                  # all replicas: arch, order, replica, bunch, seed, files, n_events, init hash
├── queue_status.json, queue.log            # written by run_ensemble.py
└── <arch>/<order>/rep<RR>_bunch<BB>/       # arch = ffnn | autoencoder, order = LO | LOwS | NLOPS
    ├── run_settings.yaml, replica_info.json
    └── output.log, stdout.log, *_train_loss.csv, *_val_loss.csv, *_learning_rates.csv,
        *_model_weights_best.pt, *_model_weights_final.pt, test_histograms.pdf, *.top   # after training
final_results/ensemble/                     # steps 3 and 4, same structure as final_results/<arch>/
├── <arch>/<train>/<test>_evts/             # e.g. ffnn/LO/NLOPS_evts = trained on LO, tested on NLO+PS
│   ├── common_test_predictions.npz
│   └── tops/{control,rep<RR>_bunch<BB>}_LL_pred.top
└── report/
```

The replica index and the bunch index are part of every directory name, every JSON entry and every row of the output tables.

## Steps

### 1. `make_ensemble_configs.py` (done)

```bash
python ensemble/make_ensemble_configs.py --check-init     # optionally --archs ... --orders ...
```

The script writes `run_settings.yaml` and `replica_info.json` per replica and merges the replicas into `ensemble_runs/ensemble_overview.json`. It stops with an error if any of these fails:
- a bunch file is missing;
- a bunch overlaps with another bunch or with the production files;
- any setting apart from the ones listed above differs from the production YAML;
- (with `--check-init`) two replicas have the same initial weights.

### 2. `run_ensemble.py` (written, not started)

```bash
python ensemble/run_ensemble.py --dry-run                 # list replicas and their state
tmux new-session -d -s ensemble "source .ml/bin/activate && \
    python ensemble/run_ensemble.py --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 3 > ensemble_runs/queue.log 2>&1"
```

> **Run on `/scratch`, not on `/ptmp`.** `/ptmp` is GPFS and gives "stale file handle" errors when the VS Code session is interrupted, which killed jobs reading from it. All runs are therefore done on the local disk:
> - `/scratch/jlinder/setup_scratch.sh` mirrors the code (a git clone plus `ensemble/` and `.ml`), all needed event files and the production inputs to `/scratch/jlinder/ML_Giovanni/`, with the same layout as on `/ptmp`.
> - The configs are generated on `/scratch` (`make_ensemble_configs.py` uses paths relative to its own location), and the trainings run there.
> - Afterwards the results are moved to the corresponding `/ptmp` location.
> - The master copy of `ensemble/` stays on `/ptmp`; rerun `setup_scratch.sh` after changing it.
>
> **Start long jobs inside tmux.** Processes started from the VS Code terminal are killed whenever the VS Code server restarts, even with `nohup`/`setsid`. A test started that way was killed twice on 2026-10-06. Jobs started in a tmux session are children of the tmux server and survive the restarts.

- **No SLURM on gpu01**, so this is a local queue. Each replica runs `polarisation_train.py run_settings.yaml` inside its own directory, which also keeps `ml_events_utils.log` separate.
- **Parallelism:** by default 3 jobs per GPU on all 8 GPUs, with 4 CPU threads per job. The training is CPU-bound (event parsing with `nworkers: 0`), and the node has 128 cores and 1.5 TB RAM. The GPUs are shared with another user; reduce `--gpus` or `--jobs-per-gpu` if needed.
- **Order:** LO, then LO+Sud., then NLO+PS. Within each sample the replicas are interleaved FFNN/AE, so complete replica sets become available early.
- **Restarting:** finished replicas (best weights exist and `output.log` contains "Best validation loss") are skipped, so the queue can simply be restarted. An interrupted queue stops its running trainings and marks them as pending again.
- **State:** every change is written to `ensemble_runs/queue_status.json`.
- **Built-in test:** it stays on (`do_test: true`) and runs on the replica's own 20% bunch test split as a sanity check. The real comparison is step 3.
- **Time estimate:** about 1/10 of the production epoch time (≈ 20–30 s per epoch, the first epoch longer while the event cache is filled). That gives roughly 1–3 h per replica and ≈ 5–9 h for all 60 runs.

**Planned launch sequence:**
1. Smoke test of one replica per architecture with `-t`.
2. One full LO FFNN replica, to check the epoch time and the stopping epoch.
3. The full queue.

### 3. `predict_common_testset.py` (to do)

**Evaluation matrix.** The training is the same for all samples (step 2), but the evaluation mirrors the production results. There, every network was also tested on the test sets of the higher accuracies, to see how much of the higher-order effects the lower-order training already captures. In `final_results/<arch>/<train>/<test>_evts/`, `<train>` is the accuracy of the training events and `<test>` the accuracy of the test events. The replicas are evaluated on exactly the same combinations:

| Trained on ↓ / tested on → | LO | LO+Sud. | NLO+PS |
|:---------------------------|:--:|:-------:|:------:|
| LO | ✓ | ✓ | ✓ |
| LO+Sud. | – | ✓ | ✓ |
| NLO+PS | – | – | ✓ |

That gives 6 combinations × 2 architectures × (10 replicas + 1 control) = 132 evaluations. Only the diagonal (`<train>` = `<test>`) enters check 1 below; check 2 and the bands are made for all six combinations. The off-diagonal combinations are where a bias from the small training size could show up differently, because the network extrapolates there.

**Common test sets.** There is one common test set per *test* sample, shared by all training samples and both architectures. It is the production test split of that sample: `random_split(<production files>, [0.6, 0.2, 0.2], seed 42)`, third subset. This is what all production evaluations used, including the cross-evaluations: e.g. `ffnn/LO/NLOPS_evts` used the same 957,189 events as `ffnn/NLOPS/NLOPS_evts`. `MLEventsDataset` sorts the files, so the split is fully determined by the file list and the seed. All test sets are disjoint from the ensemble bunches, because the bunches use only files `0251–0500` / `1001–2000`.

| Test sample | Production files | Events in the common test set |
|:------------|:-----------------|------------------------------:|
| LO | `UU_LO/pwgevents-0001…0250` | ≈ 943.5k (0.2 × 4,717,481) |
| LO+Sud. | `UU_LOwS/pwgevents-0001…0250` | 952,402 |
| NLO+PS | `UU_NLO/output_shower_events-0001…1000` | 957,189 |

**Procedure**, looping over the test samples:
- Build each test set once and load it into memory once.
- Evaluate on it every model whose training sample is at or below that accuracy: the **control** and the **10 replicas** of each (arch, `<train>`), with `*_best.pt`, `model.eval()` and $r_{\rm pred} = e^{\rm out} - 1$.
- The control is the production weights `final_results/<arch>/<train>/<train>_evts/*_model_weights_best.pt`, identical to the copies in `full_runs/…/trial_00000`.
- Reuses `MLEventsDataset`, `log_target_transform`, `build_model` and the `random_split` call of `polarisation_train.py`.

**`common_test_predictions.npz`** per (arch, `<train>`, `<test>`) contains:
- `train_order`, `test_order`;
- `global_index`, `file_index`, `event_in_file`, `file_list`: event mapping into the sorted file list of the test sample;
- `w_UU`, `w_LL`, `r_true`;
- `r_pred_control` `[N]`, `r_pred_replicas` `[10, N]`;
- `replica_index`, `bunch_index`, `seeds` `[10]`;
- `n_generated_events_test` ($0.2 \cdot 10^7$);
- the "conservative" label.

**`.top` files:** one `LL_pred.top` for the control and one per replica, with the same observables and binning as the production `.top` files. They are made with `polarisation_test.py` from the production YAML of the *same combination*, `final_results/<arch>/<train>/<test>_evts/run_settings.yaml`, with `model_weight_file` set to the replica weights and `training_seed.txt` = 42. That way every production choice is inherited, except `showered` (next note).

> **`showered: true` for every evaluation on NLO+PS events.** The production settings were inconsistent:
> - The cross-evaluations on NLO+PS (`LO/NLOPS_evts`, `LOwS/NLOPS_evts`) have `showered: true`. They compare with the showered POWHEG histograms (`pwgoutput_py8_histos-…`).
> - The NLO+PS training run itself (`NLOPS/NLOPS_evts`) has `showered: false`. It compares with the LHE-level histograms (`pwgLHEF_analysis-…`).
>
> **Decision (2026-10-06):** all ensemble evaluations on NLO+PS events use `showered: true`. This includes `NLOPS/NLOPS_evts`, for the control as well as the replicas, and the NLO+PS replica training configs, for their built-in test plots.
>
> `showered` only selects the POWHEG reference histograms ([`polarisation_test.py:486`](../polarisation_test.py)). It does not change the training or the predictions. The production files in `final_results/` are left untouched, so the control `.top` for `NLOPS/NLOPS_evts` is regenerated. Its predicted histograms still have to match the production `LL_pred.top`; only the POWHEG reference differs.

### 4. `analyse_ensemble.py` (to do) → `final_results/ensemble/report/`

**Training summary:**
- `training_summary.csv` per replica: arch, order, replica, bunch, seed, $n_{\rm train}$, stopping epoch, best epoch, best/final validation MSE, final LR, optimiser steps to the best epoch, plus the production row.
- `loss_histories.csv`: all loss and learning-rate histories in long format, tagged with replica and bunch.

**Check 1: are the 470k-event networks underfitted, rather than just noisier?** The hyperparameters were tuned at 4.7M events, and at 1/10 of the data the patience criterion may trigger much earlier.
- Compare stopping epoch, best epoch and best validation MSE with production. Flag a replica if its best epoch is below 0.5× production or its validation MSE above 1.5× production; the thresholds are stated in the report.
- Compare like with like: the unweighted log-space MSE of every replica and of the control on the **common test set** of the training sample (diagonal of the evaluation matrix). The replicas' own validation sets differ from production.
- Plot validation loss versus *optimiser steps* (one replica epoch is 1/10 of the steps of a production epoch), overlaid with the production curve.
- Look at the train–validation gap and whether learning-rate reductions happened before the stop.

**Check 2: is the replica mean offset from the full-sample control?** For each of the six (`<train>`, `<test>`) combinations, for the total $\sigma_{\rm LL}$ and every bin of every `.top` observable:
- offset = mean over replicas − control, compared with $\sigma_{\rm rep}/\sqrt{10}$ (pull) and with $\sigma_{\rm rep}$;
- also mean/true and control/true.
- Flag a significant offset (e.g. pull on the total cross section > 2, or a coherent same-sign shift across bins). An offset means the reduced training size introduces a bias, not just a spread, and has to be reported as such.

**Band plots:** for each (arch, `<train>`, `<test>`), control ± replica standard deviation (and the min/max envelope) per observable, with a ratio panel to POWHEG. Each carries the label "conservative band: 10 replicas × ~470k events (1/10 of the nominal training data)". For the cross-evaluations, the band shows how well the conclusion "the lower-order training captures X % of the higher-order effects" holds, given the training statistics.

**`SUMMARY.md`:** the two verdicts (check 2 per combination) and the interpretation note.

## Production reference values

From the production `output.log` / `*_val_loss.csv`. The validation MSE is unweighted, in log space.

| Run | Events | Best epoch | Stopping epoch | Best validation MSE |
|:----|-------:|-----------:|---------------:|--------------------:|
| FFNN LO | 4,717,481 | 54 | 89 | 1.962e-4 |
| FFNN LO+Sud. | 4,762,011 | 31 | 66 | 2.239e-4 |
| FFNN NLO+PS | 4,785,948 | 104 | 139 | 1.218e-3 |
| AE LO | 4,717,481 | 101 | 136 | 1.819e-4 |
| AE LO+Sud. | 4,762,011 | 66 | 101 | 1.985e-4 |
| AE NLO+PS | 4,785,948 | 87 | 122 | 1.206e-3 |

## Verification

1. ✅ Configs: bunches disjoint from each other and from the production files; all hyperparameters identical to production apart from the settings listed above.
2. ✅ Initial weights of the 10 replicas pairwise different (`--check-init`).
3. Smoke test (`-t`) of one replica per architecture.
4. One full LO FFNN replica: epoch time and stopping epoch, before starting the queue.
5. Common test sets: for all six combinations, the size matches production (`r_LL_count` sum in the production `LL_pred.top`), and the control's `LL_pred.top` reproduces the production `final_results/<arch>/<train>/<test>_evts/LL_pred.top` bin by bin.
6. After the training: summary table and the two checks reported back.
