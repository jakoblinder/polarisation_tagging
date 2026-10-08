# Ensemble training for a conservative uncertainty band

Ten independent replicas of each production network (FFNN, Autoencoder, ParticleNet) are trained on disjoint ~470k-event bunches. They are then tested together with the production networks on the common production test sets. The spread of the replicas gives an uncertainty band on the predictions in [`final_results/`](../final_results).

> **Interpretation.**
> - **Conservative:** each replica sees only 1/10 of the nominal training data, so the replica spread **overestimates** the uncertainty of the production networks. Every table and plot built from it is labelled *"conservative band: 10 replicas × ~470k events (1/10 of the nominal training data)"*.
> - **Biased:** the reduced training size also introduces a **bias**, not just a spread (check 2 below). The replicas are systematically less accurate than the production networks, and their mean is offset from them. The band therefore shows the scatter of networks trained on 470k events; it is not a spread centred on the production result.

## Status

| Step | Script | State |
|:-----|:-------|:------|
| 0. Mirror code and data to `/scratch` | `/scratch/jlinder/setup_scratch.sh` | done |
| 1. Replica configs | [`make_ensemble_configs.py`](make_ensemble_configs.py) | done: 90 configs |
| 2. Training | [`run_ensemble.py`](run_ensemble.py) | done: 90/90 trainings, 0 failed |
| – Clamped targets | [`count_clamped_targets.py`](count_clamped_targets.py) | done |
| – ParticleNet LO/LO+Sud. production rerun on 250 files | [`make_pn_production_rerun.py`](make_pn_production_rerun.py) | done, replaces `final_results/pn/{LO,LOwS}` |
| 3. Tests on the common test sets | [`make_test_configs.py`](make_test_configs.py), `run_ensemble.py --mode test` | done: 197/197 tests, 0 failed |
| 4. Checks and band tables | [`summarise_tests.py`](summarise_tests.py) | done: `final_results/ensemble/report/` |
| 5. Band plots | [`final_results/plots/network_comparison.ipynb`](../final_results/plots/network_comparison.ipynb), section "Ensemble uncertainty bands" | done: `final_results/plots/network_comparison/ensemble_bands_<train>_on_<test>.pdf` |

Not done, by decision: **per-event predictions.** The tests use the unchanged `polarisation_test.py`, which writes histograms (`.top`) and the test MSE but no per-event predictions. They would only be needed for observables or binnings not in `polarisation_test.py`, or for a test MSE without the clamped events (see below).

## Results

From [`final_results/ensemble/report/SUMMARY.md`](../final_results/ensemble/report/SUMMARY.md), which contains the full table for all 18 (architecture, training sample, test sample) combinations. The "control" is the production network.

**Check 1: are the 470k-event networks underfitted?** Every replica has a higher test MSE than its control, in all 17 combinations with a control. The table gives the mean replica test MSE divided by the control's, on the identical test set (same-sample tests):

| Trained and tested on | FFNN | AE | PN |
|:--|:--:|:--:|:--:|
| LO | 1.63 | 1.60 | 3.52 |
| LO+Sud. | 1.84 | 2.11 | 3.08 |
| NLO+PS | 1.08* | 1.08* | – (no PN NLO+PS weights) |

- **ParticleNet** suffers much more from the 10× smaller training set than FFNN and AE.
- **\*NLO+PS:** the ratio is diluted. About 2.2·10⁻³ of the ~2.9·10⁻³ test MSE comes from the 4 clamped events in the test set and is the same for every network. Without them, the ratio would roughly be 1.3–1.4.
- **Early stopping:** in optimiser steps, the replicas stop after only 5–22% of the production training. A replica epoch has 1/10 of the steps of a production epoch, and the replicas reach their best epoch at about the same epoch number. Early stopping, tuned at 4.7M events, very probably triggers too early at 470k.

**Check 2: is the replica mean offset from the control?** Yes, in most combinations.
- **Same-sample tests:** the replica mean of σ_LL lies 0.2–0.9% *below* the control, with pulls of −1.8 to −6.1. Pull = (replica mean − control)/(replica std/√10). In most combinations 50–90% of the histogram bins have |pull| > 2.
- **AE cross-evaluations:** for LO → LO+Sud. and LO → NLO+PS, the AE replica mean lies 3.9% and 3.6% *above* the control, with a replica spread of 1.9%.
- **No significant offset only for:** FFNN LO → LO+Sud., FFNN LO → NLO+PS, FFNN NLO+PS → NLO+PS and AE NLO+PS → NLO+PS.
- **Spread:** the relative replica spread of σ_LL is 0.2–2.2%.

## Scope

- **Architectures:**
  - FFNN: `FFNN_paper_4extraLayers_BatchNorm`, [`final_results/ffnn`](../final_results/ffnn).
  - AE: `FFNN_EMB_1024_512_256_128_64_BatchNorm`, [`final_results/autoencoder`](../final_results/autoencoder).
  - PN: `ParticleNet_best` at LO and LO+Sud., `ParticleNet_best_NLO` at NLO+PS, [`final_results/pn`](../final_results/pn).
- **Samples:** LO, LO+Sud. (`LOwS`) and NLO+PS (`NLOPS`). LO+Sud. is compulsory: the scripts add it when it is not requested.
- **Trainings:** 3 architectures × 3 samples × 10 replicas = 90.
- **Tests:** 197. That is 6 combinations of training and test sample per architecture, each with the control and 10 replicas; PN NLO+PS has no production weights, so no control test there.

## Data

The production networks were trained on the first half of each sample. The replicas use the so far unused second half, split into 10 disjoint bunches. Replica $r$ is trained on bunch $b = r$ only, with the usual 60/20/20 train/validation/test split within the bunch. Replicas with the same $r$ share bunch and seed in all architectures, so the architectures can be compared replica by replica.

| Sample | Production files | Ensemble files | Events/file | Bunch $b = 0\dots9$ | Events/bunch |
|:-------|:-----------------|:---------------|------------:|:--------------------|-------------:|
| LO | `UU_LO/pwgevents-0001…0250` | `0251…0500` | ~18.8k | files `0251+25b … 0275+25b` | ~471.5k |
| LO+Sud. | `UU_LOwS/pwgevents-0001…0250` | `0251…0500` | ~19.0k | files `0251+25b … 0275+25b` | ~475.8k |
| NLO+PS | `UU_NLO/output_shower_events-0001…1000` | `1001…2000` | ~4.8k | files `1001+100b … 1100+100b` | ~478.8k |

`n_generated_events` is $10^6$ per bunch. It only normalises the cross sections in each replica's own test plots.

### Clamped targets ($r_{\rm LL} \le -1$)

The target $\log(1 + r_{\rm LL})$ is clamped at $\log(10^{-10}) = -23.03$ (`log_target_transform`). This happens only for $r_{\rm LL} = w_{\rm LL}/w_{\rm UU} \le -1$, which requires negative weights. A single such event adds about $23^2/N$ to the unweighted MSE of a set of $N$ events. The counts, per train/validation/test split as in the training, are in `ensemble_runs/clamped_targets.csv` ([`count_clamped_targets.py`](count_clamped_targets.py)):

| Data set | Negative $w_{\rm UU}$ / $w_{\rm LL}$ | Clamped events: train / val / test |
|:--|:--:|:--:|
| LO, LO+Sud. (production and bunches) | 0 / 0 | 0 / 0 / 0 |
| NLO+PS production (4.79M events) | 10,959 / 53,737 | 11 / 1 / 4 |
| NLO+PS bunches (~479k events each) | ~1,100 / ~5,400 | validation: 1 each in bunches 0, 2, 3 |

Consequences:
- **NLO+PS bunches 0, 2, 3:** their replicas have a validation MSE of ~6.3–6.9·10⁻³ instead of ~8.5·10⁻⁴–1.3·10⁻³, in all architectures. The prediction noise on that one event probably drives their early stopping.
- **NLO+PS production test sets:** the 4 clamped events add ~2.2·10⁻³ to every test MSE (production: 2.9·10⁻³ test vs. 1.2·10⁻³ validation).
- **Cross sections:** negligible effect.

## Training settings

All hyperparameters are taken unchanged from the production settings:
- FFNN/AE: `final_results/<arch>/<order>/<order>_evts/run_settings.yaml`;
- PN: `final_results/pn/<order>/run_settings.yaml`; for LO/LO+Sud. these are now the settings of the 250-file reruns, which are identical to the original ones apart from the files.

| Network | LO: batch size / LR | LO+Sud.: batch size / LR | NLO+PS: batch size / LR |
|:--------|:--------------------|:-------------------------|:------------------------|
| FFNN | 3000 / 2.696e-3 | 4800 / 5.415e-4 | 1000 / 1.593e-3 |
| AE | 4200 / 2.629e-3 | 5000 / 2.649e-3 | 3400 / 3.794e-3 |
| PN | 1024 / 1e-3 | 1024 / 1e-3 | 256 / 1.9e-5 |

- **FFNN/AE:**
  - Optimiser: AdamW (weight decay 1e-4).
  - Early stopping: patience 35, at most 500 epochs.
  - Input and loss: lab frame, no standardisation, no penalties.
- **PN:**
  - Optimiser: AdamW at LO/LO+Sud., `paper_momentum` (RMSprop with momentum 0.9) at NLO+PS.
  - Early stopping: patience 40, at most 300 epochs (LO, LO+Sud.) or 500 (NLO+PS).
  - Input and loss: lab frame, `penalties: [cross_section]`.
- **All:**
  - `ReduceLROnPlateau` (factor 0.5, patience 3).
  - Target $\log(1 + w_{\rm LL}/w_{\rm UU})$; training loss is the MSE weighted per event by $|\log(1+w_{\rm UU})|$ (`ml_events_utils/train_loop.py`).
  - Validation loss and early stopping use the unweighted MSE.

Only these settings differ from the production run (checked by `make_ensemble_configs.py`):

| Setting | Replica value |
|:--------|:--------------|
| `mlfiles` | The bunch, as an explicit list of absolute paths (`MLEventsDataset` does not expand brace patterns in absolute paths). |
| `seed` | $1000 + r$. It controls initial weights, batch shuffling and the 60/20/20 split. |
| `outputdir`, `model_dir`, PN also `inputdir`, `model_weight_file` | The replica directory. |
| `n_generated_events` | $10^6$ |
| `histogram_dir` | Absolute path to the sample's POWHEG histograms. |
| `gpu` | 0; the GPU is chosen via `CUDA_VISIBLE_DEVICES`. |
| `showered` | `true` for NLO+PS (production training: `false`), see the note in step 3. |
| PN only: `standardise` | `false` (production: `true`). `ParticleNet_best(_NLO)` ignores the statistics (`models.py` drops `stat_norm`), so it has no effect on the model. It only adds a pass over the training set and consumes one random number before the initialisation. |

**Initial weights:**
- **Seeding:** `torch.manual_seed(seed)` ([`polarisation_train.py:71`](../polarisation_train.py)) is called before `build_model` (`:277`), and the models use PyTorch's default initialisation. A different seed therefore means different initial weights.
- **Check:** `make_ensemble_configs.py --check-init` replays the random number generation up to `build_model`, including the extra draw with `standardise`. It confirmed that the 10 initial weight sets of every (architecture, sample) are pairwise different. The hashes are stored in `replica_info.json`.

**Training outcome** (validation MSE on each replica's own 96k-event validation set, so not comparable with production; the like-for-like comparison is check 1):

| | Best epoch (min–median–max) | Stopped (min–max) | Best val MSE (mean) | Production best val MSE |
|:--|:--:|:--:|:--:|:--:|
| FFNN LO / LO+Sud. / NLO+PS | 86–118–144 / 49–62–91 / 28–47–111 | 121–179 / 84–126 / 63–146 | 3.2e-4 / 4.2e-4 / 2.6e-3† | 1.96e-4 / 2.24e-4 / 1.22e-3 |
| AE LO / LO+Sud. / NLO+PS | 76–86–129 / 74–102–126 / 31–79–121 | 111–164 / 109–161 / 66–156 | 2.9e-4 / 4.3e-4 / 2.5e-3† | 1.82e-4 / 1.98e-4 / 1.21e-3 |
| PN LO / LO+Sud. / NLO+PS | 94–111–200 / 82–92–284 / 25–49–64 | 134–240 / 122–300 / 65–104 | 8.8e-4 / 1.0e-3 / ~1.25e-3† | 2.50e-4 / 3.22e-4 (reruns) / – |

† Without bunches 0, 2, 3 (clamped validation event): ~8.5e-4 (FFNN, AE) and ~1.25e-3 (PN). One PN LO+Sud. replica ran to the 300-epoch limit.

## ParticleNet LO and LO+Sud. production rerun

**The inconsistency in the original PN results:**
- The original LO and LO+Sud. run settings list 200 event files.
- Their test histograms, however, were made on the 250-file test splits: 943,496 and 952,402 events, the same test sets as FFNN/AE.
- If the networks were trained on 200 files, 48% of those test events were in their training split.

The PN NLO+PS network is consistent (confirmed by the person who trained it), so only LO and LO+Sud. were redone.

**The rerun** (`make_pn_production_rerun.py`):
- **Training:** the original settings on the same 250 files as FFNN/AE (`n_generated_events` = 10⁷, `standardise: false`, seed 42), in `full_runs/ParticleNet/{LO,LOwS}/`.
- **Cross-evaluations:** `--cross-eval` writes the evaluations on the higher accuracies, run with `polarisation_test.py`, into `full_runs/ParticleNet/<train>/<train>_with_<test>_evts/`.
- **Replacement:** the results replace `final_results/pn/{LO,LOwS}/*.top` and `run_settings.yaml`. The old files are in the git history.

| Trained → tested | σ_LL pred/true, rerun | Original | Per-bin rerun/original: median (max) deviation |
|:--|:--:|:--:|:--|
| LO → LO | 0.9993 | 0.9996 | 0.17 % (3.3 %) |
| LO → LO+Sud. | 0.9973 | 1.0024 | 0.56 % (5.6 %) |
| LO → NLO+PS | 0.9716 | 0.9777 | 0.71 % (5.0 %) |
| LO+Sud. → LO+Sud. | 0.9994 | 0.9992 | 0.13 % (1.4 %) |
| LO+Sud. → NLO+PS | 0.9744 | 0.9745 | 0.14 % (1.8 %) |

Rerun training: best validation MSE 2.50e-4 at epoch 162 of 202 (LO) and 3.22e-4 at epoch 131 of 171 (LO+Sud.).

## Working on `/scratch` and in tmux

> **Run on `/scratch`, not on `/ptmp`.** `/ptmp` is GPFS and gives "stale file handle" errors when the VS Code session is interrupted, which kills jobs reading from it.
> - **Mirror:** `/scratch/jlinder/setup_scratch.sh` mirrors to `/scratch/jlinder/ML_Giovanni/`, with the same layout as on `/ptmp`: a git clone of the repository, `ensemble/`, `.ml`, all event files (LO/LO+Sud. `0001–0500`, NLO+PS `0001–2000`, POWHEG histograms), the production FFNN/AE weights, and the inputs of the FFNN/AE NLO+PS production runs.
> - **Master copy:** `ensemble/` on `/ptmp`. Rerun `setup_scratch.sh` after changing it.
> - **Configs:** generated on `/scratch` (the scripts use paths relative to their own location); trainings and tests run there.
> - **Results:** copied to the corresponding `/ptmp` location with `rsync --checksum` and checked file by file (md5).
>
> **Start long jobs inside tmux.** Processes started from the VS Code terminal are killed whenever the VS Code server restarts, even with `nohup`/`setsid`. Jobs in a tmux session survive this, but not the end of the *last* login session on the node (lingering is off), so keep one ssh session open.

## Steps

All commands are run from `polarisation_tagging/` on `/scratch`, with `.ml` activated; long ones inside tmux.

### 1. Configs: `make_ensemble_configs.py`

```bash
python ensemble/make_ensemble_configs.py --check-init     # optionally --archs ffnn autoencoder pn --orders ...
```

The script writes `run_settings.yaml` and `replica_info.json` per replica and merges all replicas into `ensemble_runs/ensemble_overview.json`. It stops with an error if any of these fails:
- a bunch file is missing;
- a bunch overlaps another bunch or the production files;
- a setting other than the ones listed above differs from the production settings;
- (with `--check-init`) two replicas have the same initial weights.

### 2. Training: `run_ensemble.py`

```bash
python ensemble/run_ensemble.py --dry-run
tmux new-session -d -s ensemble "source .ml/bin/activate && \
    python ensemble/run_ensemble.py --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 3 > ensemble_runs/queue.log 2>&1"
```

- **Queue:** there's no SLURM on gpu01, so this is a local queue. Each replica runs `polarisation_train.py run_settings.yaml` in its own directory, which also keeps `ml_events_utils.log` separate.
- **Parallelism:** several jobs per GPU, because the training is CPU-bound (event parsing, `nworkers: 0`). The node has 128 cores and 1.5 TB RAM.
- **Restarting:** finished replicas are skipped; an interrupted queue marks its running jobs as pending again.
- **State:** written to `ensemble_runs/queue_status.json`.
- **What was run:** FFNN/AE with 3 jobs per GPU on 7 GPUs, about 16 s per LO epoch, done in about 2 h. PN with 4 jobs per GPU on 5 GPUs, about 18 s per LO epoch, done in about 1.5 h. The FFNN/AE queue logs are kept as `queue_ffnn_ae.log` and `queue_status_ffnn_ae.json`.

### 3. Tests on the common test sets: `make_test_configs.py`, `run_ensemble.py --mode test`

```bash
python ensemble/make_test_configs.py
tmux new-session -d -s ensemble_test "source .ml/bin/activate && \
    python ensemble/run_ensemble.py --mode test --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 6 --threads-per-job 2 \
    > final_results/ensemble/queue_test.log 2>&1"
```

**Evaluation matrix.** As in the production results, every network is also tested on the test sets of the higher accuracies:

| Trained on ↓ / tested on → | LO | LO+Sud. | NLO+PS |
|:---------------------------|:--:|:-------:|:------:|
| LO | ✓ | ✓ | ✓ |
| LO+Sud. | – | ✓ | ✓ |
| NLO+PS | – | – | ✓ |

**Common test sets.** There is one per *test* sample, shared by all training samples and all three architectures: the production test split, `random_split(<production files>, [0.6, 0.2, 0.2], seed 42)`, third subset. All production evaluations used them, including the cross-evaluations. They are disjoint from the ensemble bunches.

| Test sample | Production files | Test events |
|:--|:--|--:|
| LO | `UU_LO/pwgevents-0001…0250` | 943,496 |
| LO+Sud. | `UU_LOwS/pwgevents-0001…0250` | 952,402 |
| NLO+PS | `UU_NLO/output_shower_events-0001…1000` | 957,189 |

**What a test is.** Each network is run once with the unchanged `polarisation_test.py`:
- **Control:** FFNN/AE weights from `final_results/<arch>/<train>/<train>_evts/`; PN weights from the reruns in `full_runs/ParticleNet/<train>/`.
- **Replicas:** their own best weights.
- **Settings:** the network's own settings with the test sample's production files and POWHEG histograms. `inputdir` is the test directory, which holds `training_seed.txt` = 42, because `polarisation_test.py` takes the split seed from there. That puts the replicas, trained with seeds 1000 + r, on the production test split too.
- **Run:** 197 tests, 48 in parallel, about 7.5 min each, about 50 min in total.
- **Verification:** the FFNN LO control reproduces the production test. `LL_true.top`, `POWHEG_LL.top` and the test loss are identical; `LL_pred.top` agrees to ≤ 3·10⁻⁷ (GPU rounding).

> **`showered: true` for every evaluation on NLO+PS events** (decision of 2026-10-06).
> - **The inconsistency:** the production cross-evaluations on NLO+PS used the showered POWHEG histograms (`pwgoutput_py8_histos-…`), the NLO+PS training runs the LHE-level ones (`pwgLHEF_analysis-…`).
> - **What it affects:** `showered` only selects the POWHEG reference ([`polarisation_test.py:486`](../polarisation_test.py)), not the predictions.
> - **Production results:** the FFNN and AE NLO+PS production tests were redone with `showered: true`, and `final_results/{ffnn,autoencoder,powheg}/NLOPS` were updated.

### 4. Checks and band tables: `summarise_tests.py`

```bash
python ensemble/summarise_tests.py
```

For every network the script reads the test MSE (unweighted, log space, from `ml_events_utils.log`) and the histograms (`LL_pred.top`, `LL_true.top`). For PN NLO+PS, the control histograms are the production `final_results/pn/NLOPS/NLOPS.top`, and there is no control MSE. Output in `final_results/ensemble/report/`:

| File | Content |
|:--|:--|
| `SUMMARY.md`, `checks.csv` | Both checks per (arch, train, test): test MSE of control and replicas, σ_LL ratios, replica spread, pull, fraction of bins with \|pull\| > 2 |
| `test_summary.csv` | One row per network: test MSE, σ_LL pred/true, replica, bunch |
| `band_<arch>_<train>_<test>.csv` | Per observable and bin: true, control, replica mean/std/min/max, pull, relative band width; labelled "conservative" |

Flags: check 1 if the mean replica test MSE exceeds 1.5× the control; check 2 if |pull| of σ_LL > 2.

### 5. Band plots: `network_comparison.ipynb`

The section "Ensemble uncertainty bands (conservative)" in [`final_results/plots/network_comparison.ipynb`](../final_results/plots/network_comparison.ipynb), after the kinematic plots:
- **Inputs:** it reads the production `.top` files and the 10 replica `LL_pred.top` files per (architecture, training sample, test sample) from `final_results/ensemble/`.
- **The band:** the **production curve ± the standard deviation of the 10 replicas**, in every bin and in the ratio to POWHEG. The replicas only set the width of the band; their mean, which is offset from the production curve (check 2), is not shown. The band is computed after the rebinning of each plot.
- **Settings:** the same plot settings (ranges, rebinning, axes) as the kinematic plots of the test sample; the plotting cell stores them in `plot_observables_per_order`.
- **RFR:** in the same-sample PDFs (LO, LO+Sud., NLO+PS) the RFR is shown too, with its own uncertainty from its `.top` files. That is a per-event variance from a LightGBM fit to the squared out-of-bag residuals of the forest, added per bin as independent errors; it is not a replica spread.
- **Output:** one PDF per training and test sample, `final_results/plots/network_comparison/ensemble_bands_<train>_on_<test>.pdf` (8–10 observables each). The same-sample plots have the titles of the kinematic plots. The plots contain no text on the origin of the uncertainties; that is explained only in the notebook's markdown note. The cell also prints σ_LL with its band per network (and for the RFR).

## Directory layout

```
ensemble/                                    # scripts + this README
ensemble_runs/
├── ensemble_overview.json                   # all 90 replicas: arch, order, replica, bunch, seed, files, n_events, init hash
├── clamped_targets.csv                      # count_clamped_targets.py
├── queue_status*.json, queue*.log           # run_ensemble.py (training)
└── <arch>/<order>/rep<RR>_bunch<BB>/        # arch = ffnn | autoencoder | pn, order = LO | LOwS | NLOPS
    ├── run_settings.yaml, replica_info.json
    └── output.log, stdout.log, *_{train_loss,val_loss,learning_rates}.csv, *_model_weights_{best,final}.pt,
        *_training_history.pdf, test_histograms.pdf, *.top          # built-in test on the bunch's own test split
full_runs/ParticleNet/<train>/               # PN LO/LOwS production reruns (+ <train>_with_<test>_evts/ cross-evaluations)
final_results/ensemble/
├── test_overview.json                       # all 197 tests: arch, train, test, network, replica, bunch, weights, dir
├── queue_status_test.json, queue_test.log   # run_ensemble.py --mode test
├── <arch>/<train>/<test>_evts/<control|rep<RR>_bunch<BB>>/
│   └── run_settings.yaml, training_seed.txt (42), LL_pred.top, LL_true.top, POWHEG_*.top, test_histograms.pdf,
│       ml_events_utils.log (test MSE), stdout.log
└── report/                                  # summarise_tests.py
```

The replica and bunch indices are part of every directory name, every JSON entry and every table row.

## Production reference values

From the production `output.log` / `*_val_loss.csv`. Unweighted validation MSE in log space; test MSE on the common test sets from step 3.

| Run | Training events | Best epoch | Stopping epoch | Best validation MSE | Test MSE |
|:----|-------:|-----------:|---------------:|--------------------:|------:|
| FFNN LO | 4,717,481 | 54 | 89 | 1.962e-4 | 1.970e-4 |
| FFNN LO+Sud. | 4,762,011 | 31 | 66 | 2.239e-4 | 2.280e-4 |
| FFNN NLO+PS | 4,785,948 | 104 | 139 | 1.218e-3 | 2.895e-3 |
| AE LO | 4,717,481 | 101 | 136 | 1.819e-4 | 1.820e-4 |
| AE LO+Sud. | 4,762,011 | 66 | 101 | 1.985e-4 | 2.010e-4 |
| AE NLO+PS | 4,785,948 | 87 | 122 | 1.206e-3 | 2.878e-3 |
| PN LO (rerun) | 4,717,481 | 162 | 202 | 2.497e-4 | 2.490e-4 |
| PN LO+Sud. (rerun) | 4,762,011 | 131 | 171 | 3.222e-4 | 3.260e-4 |

The NLO+PS test MSE includes the 4 clamped test events (~2.2·10⁻³).

## Verification

1. ✅ Configs: bunches disjoint from each other and from the production files; hyperparameters identical to production apart from the settings listed above.
2. ✅ Initial weights of the 10 replicas pairwise different for every (architecture, sample).
3. ✅ Smoke tests (`-t`, in a separate `--outputdir`) for all five models before the queues; parameter counts identical to production.
4. ✅ Trainings: 90/90 finished; tests: 197/197 finished.
5. ✅ Common test sets: sizes match production (`r_LL_count` sums in the production `.top`); the FFNN LO control reproduces the production test.
6. ✅ All results copied to `/ptmp` and checked file by file.
