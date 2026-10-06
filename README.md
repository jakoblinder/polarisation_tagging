# Polarisation Tagging

PyTorch framework to learn, event by event, the polarisation fractions of ZZ (and Z+jet) events, i.e. the ratio of a polarised to the unpolarised weight (e.g. $w_{\mathrm{LL}}/w_{\mathrm{UU}}$), from the lepton momenta in POWHEG `.ml` event files. The predicted distributions are compared with the POWHEG reference histograms, and the network hyperparameters can be optimised with Optuna.

## Installation

```bash
git clone git@github.com:jakoblinder/polarisation_tagging.git
cd polarisation_tagging
python3 -m venv .ml && source .ml/bin/activate
pip install -r requirements.txt
```

All commands below are run from the `polarisation_tagging/` directory.

## Training: `polarisation_train.py`

The script splits the events into training/validation/test sets (60/20/20 %, fixed by `--seed`), trains the model on $\log(1 + w_{\mathrm{pol}}/w_{\mathrm{UU}})$ with early stopping, and finally evaluates the best model on the test set and compares it with the POWHEG histograms.

### Quick start

`examplary_eventfiles/` contains ten `UU_LO` event files (~19 000 events each) and the matching POWHEG histograms, enough to try the whole chain in a few minutes:

```bash
# Smoke test: build dataset and model, run one batch, exit.
python polarisation_train.py examplary_eventfiles/pwgevents-0001.ml --outputdir runs/LL_example -t --verbose

# Short training of the LL fraction on two files.
python polarisation_train.py "examplary_eventfiles/pwgevents-{0001..0002}.ml" \
    -m FFNN_paper_BatchNorm -o AdamW -l 3e-3 -e 20 -p 5 \
    --polarisation LL --n_generated_events 8e4 --penalties cross_section \
    --gpu 0 --outputdir runs/LL_example
```

- **File patterns** (`{0001..0002}`, `*`, `?`) are expanded by the program when quoted (relative paths only), otherwise by the shell.
- **`--n_generated_events`** is the number of events *generated* by POWHEG (before the fiducial cuts) for the files used: `numevts × number of files`, i.e. $4\cdot10^4$ per file for `UU_LO`/`UU_LOwS` and $10^4$ for `UU_NLO` (see [below](#event-files)). It only normalises the cross sections in the test plots. The default ($10^7$) is rarely right.
- **Put the event files first**, otherwise `--penalties` may take them as penalty names.
- **Log:** Progress goes to `<outputdir>/output.log` (`--verbose` also prints it). Errors are always printed. The output directory is created if needed, and an existing run in it is overwritten.

### Output

| File | Content |
|:-----|:--------|
| `output.log`, `run_settings.yaml` | Log and complete settings of the run. |
| `<model>_model_weights_best.pt` / `_final.pt` | Weights with the lowest validation loss / after the last epoch. |
| `<model>_{train_loss,val_loss,learning_rates}.csv`, `<model>_training_history.pdf` | Loss history, written after every epoch, and its plot. |
| `test_histograms.pdf`, `*.top` | Predicted vs. true vs. POWHEG distributions on the test set. |
| `training_seed.txt` | Seed of the data split, needed to rebuild the test set. |

The POWHEG histograms (`pwgLHEF_analysis-mean-W*.top`, or `pwgoutput_py8_histos-mean-W*.top` with `--showered`) are taken from the directory of the first event file.

### Main options

See `python polarisation_train.py --help` for all options and defaults.

| Option | Meaning |
|:-------|:--------|
| `-m`, `--model` | Architecture, a key of `model_dict` in [`models.py`](ml_events_utils/models.py), e.g. `FFNN_paper_BatchNorm` (default), `FFNN_EMB_1024_512_256_128_64_BatchNorm`, `ParticleNet_best`. |
| `--polarisation` | Learnt ratio: `LL` (default), `LT`, `TL`, `TT`, `LU`, `UL`. |
| `-o`, `-l`, `-b`, `-e`, `-p` | Optimizer (`paper`, `AdamW`, …), learning rate, batch size, max. epochs, early-stopping patience. |
| `--penalties` | Extra loss terms: `cross_section` (deviation of the learnt total cross section), `ZdecayAngles`. |
| `--outputdir`, `-g`, `-n` | Output directory, GPU index, number of DataLoader workers. |
| `--cmframe`, `--input_choice jan2026` | Boost into the four-lepton rest frame / use $p_T$, $y$ of both Z bosons and the decay angles as input. |
| `--standardise` | Standardise the inputs (no effect for the `FFNN_EMB_…_BatchNorm/LayerNorm` models). Statistics are read from `<outputdir>/../dataset_statistics.json` if present (cache of the scans). |
| `--useZjet`, `--showered` | Z+jet dataset / showered events. |
| `--dont_test`, `-t`, `--replot` | Skip the test step / smoke test / only redo the loss plot of an existing run. |

### Settings files

All settings can also be given as a YAML file as **first** argument. Command-line options after it override the file:

```yaml
# my_run.yaml: settings not listed take the command-line default.
parameters:
  mlfiles:
    value: [{__type__: path, value: "examplary_eventfiles/pwgevents-{0001..0002}.ml"}]
  outputdir:
    value: {__type__: path, value: runs/LL_example}
  optimizer: {value: AdamW}
  epochs: {value: 20}
  penalties: {value: [cross_section]}
  n_generated_events: {value: 80000}
```

```bash
python polarisation_train.py my_run.yaml
python polarisation_train.py my_run.yaml --polarisation TT -e 50 --cmframe
# Repeat an earlier run, or vary it (use a new output directory!):
python polarisation_train.py runs/LL_example/run_settings.yaml --outputdir runs/LL_lr1e-4 -l 1e-4
```

Keys are the long option names (`learning_rate`, `batch_size`, `test_mode`, `labframe`, …). The paths `mlfiles` and `outputdir` need the `__type__: path` form shown above. `ParticleNet` additionally reads `width`, `n_hidden` and `growing_edge`, and `FFNN_general` requires `width` and `n_hidden`. A file with all keys is [`run_settings_scan_example.yaml`](run_settings_scan_example.yaml).

## Testing: `polarisation_test.py`

The test runs automatically after the training. To redo it, e.g. after changing the plots:

```bash
python polarisation_test.py runs/LL_example/run_settings.yaml
# or without settings file (repeat the data options of the training):
python polarisation_test.py "examplary_eventfiles/pwgevents-{0001..0002}.ml" \
    FFNN_paper_BatchNorm FFNN_paper_BatchNorm_model_weights_best.pt \
    --inputdir runs/LL_example --n_generated_events 8e4
```

The results are written to the run directory (`--inputdir`), and the log to `ml_events_utils.log` in the current directory.

## Hyperparameter optimisation

The scans take a settings file whose `fit:` entries define the ranges to sample, see [`run_settings_scan_example.yaml`](run_settings_scan_example.yaml) (its `value`s are the best settings found so far) and [`scan_particlenet.yaml`](scan_particlenet.yaml) for the `ParticleNet` architecture. New architecture parameters have to be passed to the model in `build_model` in [`models.py`](ml_events_utils/models.py).

```bash
# Bayesian search with Optuna on 4 GPUs (resumable via the database):
python run_hyperparam_scan_optuna.py run_settings_scan_example.yaml --study-name pol_scan \
    --storage sqlite:///pol_scan.db --output-root scan_runs --gpus 0 1 2 3 --n-trials 50 \
    --run-test --enable-pruning --plot-after
# Plot the results at any time, also while the scan is running:
python run_hyperparam_scan_optuna.py run_settings_scan_example.yaml --study-name pol_scan \
    --storage sqlite:///pol_scan.db --output-root scan_runs --plot-only
# Random search (not recommended):
python run_hyperparam_scan.py run_settings_scan_example.yaml --n-trials 20 --output-root scan_runs --run-test
```

`--run-test` produces the test histograms of every trial, `--enable-pruning` stops bad trials early (`--pruner median` or `percentile`, see `run_worker`), and `--complexity-weight` (experimental) penalises large models. See `--help` for all options.

## Data

### Event files

The event files are on Nextcloud: [Download ZIP (updated 24.02.2026)](https://nextcloud.mpp.mpg.de/nextcloud/index.php/s/ZY6nDHryzHemgrE) (~13 GB, ~67 GB extracted). There is one directory each for LO (`UU_LO`), LO with the radiation from the POWHEG Sudakov (`UU_LOwS`) and NLO (`UU_NLO`), containing

- `pwgevents-????.ml`: events from POWHEG (after stage 4), and `output_shower_events-????.ml`: the showered events (QCD radiation only),
- `pwgLHEF_analysis-mean-W*.top` / `pwgoutput_py8_histos-mean-W*.top`: the POWHEG histograms at LHE level / after the shower,
- `powheg.input-save`: the POWHEG settings, including the weight definitions.

The events already passed the fiducial cuts [TODO: Add reference]. Cross sections are therefore the sum of the event weights divided by the number of *generated* events:

| Run       | #seeds | `numevts` per seed | generated events |
|:----------|-------:|-------------------:|-----------------:|
| `UU_LO`   |  500   | $4 \cdot 10^{4}$   | $2 \cdot 10^{7}$ |
| `UU_LOwS` |  500   | $4 \cdot 10^{4}$   | $2 \cdot 10^{7}$ |
| `UU_NLO`  | 2000   | $10^{4}$           | $2 \cdot 10^{7}$ |

Results of models tested on both ZZ and Z+jet (with `test_histograms.pdf` and `<model>_training_history.pdf` per run, and the ZZ event files) are [here (updated 13.01.2026)](https://nextcloud.mpp.mpg.de/nextcloud/index.php/s/CtQrZamtorfWYdC). The Z+j events for rL and rT are [here (updated 12.01.2026)](https://cernbox.cern.ch/s/3JmiHNoKW0bwYtd).

### Format

Each event holds the four-momenta (`px py pz E`) of the four leptons and the event weights:

```xml
<MLEvents>
<event>
 px1 py1 pz1 E1
 px2 py2 pz2 E2
 px3 py3 pz3 E3
 px4 py4 pz4 E4
<rwgt>
<weight id='UU'> weight_value </weight>
<weight id='LL'> weight_value </weight>
...
</rwgt>
</event>
</MLEvents>
```

| Weight ids | POWHEG histograms |
|:-----------|:------------------|
| `UU`, `LL`, `LT`, `TL`, `TT`, `LU`, `UL` | `W8` … `W14` |
| `LL-11`, `LL-12`, `LL-21`, `LL-22`, `LL-15`, `LL-51`, `LL-55` | `W15` … `W21` |
| `UU-11`, `UU-12`, `UU-21`, `UU-22`, `UU-15`, `UU-51`, `UU-55` | `W22` … `W28` |

`W1` … `W7` are the usual scale variations. The digits in e.g. `LL-12` denote renscfact=1 and facscfact=2, with `5` meaning 0.5 (`LL-11` = `LL`).

### Using the dataset in your own code

```python
from torch.utils.data import DataLoader
from ml_events_utils import MLEventsDataset, boost_into_four_lepton_cm_frame

dataset = MLEventsDataset(
    "examplary_eventfiles/pwgevents-{0001..0002}.ml",
    labels=["LL/UU", "UU"],                       # targets: ratios or single weights, e.g. "LL-12/UU-21"
    transform=boost_into_four_lepton_cm_frame,    # optional, see ml_events_utils/transforms.py
    cache_events=True,                            # keep parsed events in memory (small files)
)
features, labels = dataset[0]                     # shapes [16], [2]

for features, labels in DataLoader(dataset, batch_size=32, shuffle=True):
    ...
```

The dataset only indexes the files and parses events on demand. For large samples use `cache_events=False` (`--no-cache-events`) and several DataLoader workers (`--nworkers`).

## File structure

```
polarisation_tagging/
├── polarisation_train.py          # Training (CLI)
├── polarisation_test.py           # Testing / plots (CLI)
├── plot_training_history.py       # Loss history plot
├── run_hyperparam_scan_optuna.py  # Hyperparameter optimisation with Optuna
├── run_hyperparam_scan.py         # Random hyperparameter search
├── run_settings_scan_example.yaml # Example settings file for the scans
├── scan_particlenet.yaml          # Example settings file for a ParticleNet scan
├── requirements.txt
├── ml_events_utils/               # Library used by the scripts
│   ├── cli.py                     # Command-line / YAML parsing
│   ├── run_settings.py            # Settings classes
│   ├── ml_events_dataset.py       # MLEventsDataset
│   ├── Zjet_events_dataset.py     # Dataset for Z+jet
│   ├── models.py                  # Models, model_dict and build_model
│   ├── train_loop.py              # Training/validation loops incl. penalties
│   ├── transforms.py              # Boosts and other feature/target transforms
│   ├── analysis.py                # Physics observables (angles, pT, ...)
│   ├── write_top_file.py          # Writing POWHEG-like .top histograms
│   ├── logger.py                  # Logging setup
│   ├── __init__.py                # Package exports, colours, fonts
│   ├── fonts/, stylesheet*.mpl    # Plot style (stylesheet.mpl requires LaTeX)
├── examplary_eventfiles/          # Ten UU_LO .ml files + POWHEG histograms for quick tests
├── final_results/                 # Results of the final runs
├── random_forest_regression/      # Random forest regression as a baseline
├── zj_material/                   # Z+jet notebooks/scripts (experimental)
└── notes/                         # Development notes / TODOs
```
