# Polarisation Tagging

A PyTorch-based machine learning framework for analysing polarisation states in high-energy physics events using momentum data from ML events files.

## Overview

This project provides tools to:
- Parse large ML events files containing particle momentum data
- Extract polarisation reweight information (LL, LT, TL, TT)
- Train machine learning models to predict polarisation states
- Handle memory-efficient processing of large datasets

## Key Components

### MLEventsDataset Class

A custom PyTorch Dataset class designed for efficient processing of ML events files:

```python
from pathlib import Path
from polarisation_tagging import MLEventsDataset

# Initialize dataset
dataset = MLEventsDataset(
    file_path="path/to/your/events.ml",
    labels = ["LL/UU",],
    cache_events=False,  # Set to True for smaller files
    transform=None       # Optional data transformations
)
```

**Features:**
- **Select labels**: Select the label(s) to be used for the target(s) by specifying their weight id (s. below). Writing something like 'LL/UU' will result in the target to be the ration of the LL polarisation component w.r.t. the unpolarised prediction. Concerning possible scale variations, 'LL-12/UU-21' would for example be the ratio of the LL prediction with renscfact=1 & facscfact=2 and the unpolarised prediction with renscfact=2 & facscfact=1.
- **Caching/ Lazy Loading**: Possible to only load events when accessed, suitable for large files.
- **Transform Support**: Apply custom transformations to momentum and label data

### Data Format

The dataset expects ML events files with the following structure:

```xml
<MLEvents>
<event>
 particle_1_momentum_component_1  particle_1_momentum_component_2  particle_1_momentum_component_3  particle_1_momentum_component_4
 particle_2_momentum_data
 particle_3_momentum_data
 particle_4_momentum_data
<rwgt>
<weight id='UU'> weight_value </weight>
<weight id='LL'> weight_value </weight>
<weight id='LT'> weight_value </weight>
<weight id='TL'> weight_value </weight>
<weight id='TT'> weight_value </weight>
<weight id='LL-11'> weight_value </weight>
<weight id='LL-12'> weight_value </weight>
<weight id='LL-21'> weight_value </weight>
<weight id='LL-22'> weight_value </weight>
<weight id='LL-15'> weight_value </weight>
<weight id='LL-51'> weight_value </weight>
<weight id='LL-55'> weight_value </weight>
<weight id='UU-11'> weight_value </weight>
<weight id='UU-12'> weight_value </weight>
<weight id='UU-21'> weight_value </weight>
<weight id='UU-22'> weight_value </weight>
<weight id='UU-15'> weight_value </weight>
<weight id='UU-51'> weight_value </weight>
<weight id='UU-55'> weight_value </weight>
</rwgt>
</event>
</MLEvents>
```
The indices in for example 'LL-12' denote the scale variation renscfact=1 & facscfact=2, whereas for example 'LL-51' corresponds to renscfact=0.5 & facscfact=1. ('LL' and 'LL-11' denote the same weight.)

**Event files:**
The event files for this project are stored on Nextcloud: [Download ZIP (Updated on 20.11.2025)](https://nextcloud.mpp.mpg.de/nextcloud/index.php/s/fJXHsAt2CF2Qqwy). Note that the gziped file has a size of 5 GB and extends to ~25 GB.

They contain a directory for LO (UU_LO), for LO including the radiation from the POWHEG Sudakov (UU_LOwS) and at NLO (UU_NLO). In each directory there are the histograms done at lhe level (`pwgLHEF_analysis-mean-W*.top`) and after showering with only qcd radiation being activated (`pwgoutput_py8_histos-mean-W*.top`). The definition of the weights can be found in the `powheg.input-save` file.

The filtered events can be found in `pwgevents-????.ml` coming from POWHEG only, i.e. after stage 4. `output_shower_events-????.ml` are the filtered events created from the showered events.
The events are part of the fiducial phase space and are written in the form `px`, `py`, `pz`, `E`.

**Important note on the statistics in the provided datasets**

The `.ml` files provided include already fiducial cuts according to the phase space defined in [TODO: Add reference.]. Thus, in order to get the correct cross section, for example, for the unpolarised cross section $\sigma_{\mathrm{UU}}$, the weight of each event coming from all `*.ml` needs to be summed and devided by the number of simulated events before applying the cuts. This number can be infered by `numevts` parameter in the respective `powheg.input-save` file and the number of seeds for the specific run. The number of generated events in total and for each file is:

| File      | #seeds |         `numevts` | total number of simulated events |
|:----------|-------:|------------------:|---------------------------------:|
| `UU_LO`   |    100 |          $10^{5}$ |                $10^{7}$          |
| `UU_LOwS` |    100 |          $10^{5}$ |                $10^{7}$          |
| `UU_NLO`  |    100 |  $5 \cdot 10^{4}$ |        $5 \cdot 10^{6}$          |

**Other potentially interesting files**
Some results where the models are test on both datasets, i.e. for ZZ and Z+jet can be found in [this (Updated on 13.01.2026)](https://nextcloud.mpp.mpg.de/nextcloud/index.php/s/CtQrZamtorfWYdC) directory. It also includes the above mentioned event files for ZZ. Interesting plots are for each run the `test_histograms.pdf` and potentially the `<model name>_training_history.pdf`, showing some distributions done on the test data set and the loss behaviour during the training of the models.

#### Z+j Events
For testing purposes a [link (Updated on 12.01.2026)](https://cernbox.cern.ch/s/3JmiHNoKW0bwYtd) to the Z+j events for rL and rT.

## Usage Examples

### Basic Dataset Usage

```python
from torch.utils.data import DataLoader
from pathlib import Path

# Load dataset (supports pattern matching)
file_path = Path("data/*.ml")
dataset = MLEventsDataset(file_path, labels = ["LL/UU",], cache_events=True)

# Check dataset info
print(f"Dataset contains {len(dataset)} events")
print(f"File info: {dataset.get_file_info()}")

# Access single event
features, labels = dataset[0]
print(f"Features shape: {features.shape}")  # [n_momentum_components]
print(f"Labels shape: {labels.shape}")      # [1] - LL divided by UU weights
```

### DataLoader Integration

```python
# Create DataLoader for batch processing
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,      # Use 0 for notebook environments
    pin_memory=False    # Set to True if using GPU
)

# Iterate through batches
for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Features: {batch_features.shape}")  # [batch_size, n_features]
    print(f"  Labels: {batch_labels.shape}")      # [batch_size, 1]

    # Your training/inference code here
    break
```

### Custom Transforms

```python
import torch

def boost_into_four_lepton_cm_frame(features):
    """
    Boost the four leptons into their combined center-of-mass frame.
    Parameters:
    features : torch.Tensor, shape (..., 16)
        Input features containing four leptons' four-momenta in the order:
        (px1, py1, pz1, E1, px2, py2, pz2, E2, px3, py3, pz3, E3, px4, py4, pz4, E4).
    Returns:
    torch.Tensor, shape (..., 16)
        The boosted four leptons' four-momenta in the same order as the input.
    """
    momenta = features.reshape(4, -1)  # Assuming features contain x leptons with 4 momentum components each.
                                       # p1 = momenta[0], p2 = momenta[1], p3 = momenta[2], p4 = momenta[3].
    p_tot   = momenta.sum(dim=0)

    momenta_prime = torch.zeros_like(momenta)
    for i in range(4):
        momenta_prime[i] = boostinv(momenta[i], p_tot)

    # p_tot_prime = momenta_prime.sum(dim=0)
    return momenta_prime.reshape(-1)

# Apply transform
dataset = MLEventsDataset(
    file_path="data/events.ml",
    transform=boost_into_four_lepton_cm_frame
    labels = ["LL/UU", "UU"],
    transform=trafo,
)
```

## Performance Considerations

### For Large Files (>1GB)
- Set `cache_events=False` to avoid memory issues
- Use `num_workers>0` in DataLoader for loading the data in parallel.
- Consider using `pin_memory=True` when training on GPU

### For Small Files (<100MB)
- Set `cache_events=True` for faster repeated access
- Can use `num_workers>0` in production environments

### Memory Usage
- The dataset uses lazy loading - only the file index is kept in memory
- Each event is parsed on-demand
- Enable caching only if you have sufficient RAM

## File Structure

```
polarisation_tagging/
│
├── README.md                     # Project documentation
├── requirements.txt              # Python package dependencies
│
├── polarisation_train.py         # Main training entry point (CLI)
├── polarisation_test.py          # Test / evaluation entry point (CLI)
├── plot_training_history.py      # Plotting utilities for training logs
│
├── ml_events_utils/              # Core library code used by train/test
│   ├── __init__.py
│   ├── ml_events_dataset.py      # MLEventsDataset (+ helpers)
│   ├── train_loop.py             # Training loop
│   ├── valid_loop.py             # Validation loop
│   ├── transforms.py             # Feature/ target transforms (boosts, scaling, etc.)
│   ├── analysis.py               # Physics helpers (e.g. angular observables)
│   └── models/                   # Model definitions (FFNN variants, etc.)
│
├── examplary_eventfiles/         # Small example .ml files for quick tests
│   └── pwgevents-0002.ml
│
├── zj_material/                  # Z+jet related notebooks/scripts (experimental)
│   ├── *.ipynb
│   └── *.py
│
└── notes/                        # Development notes / TODOs
    └── todo_list_2025-12-10.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/polarisation_tagging.git
   cd polarisation_tagging
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```


<!-- ## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes and commit them
4. Push to your forked repository
5. Submit a pull request -->

Please ensure your code adheres to the existing style and includes appropriate tests.

## License

<!-- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
