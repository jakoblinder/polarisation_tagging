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
    cache_events=False,  # Set to True for smaller files
    transform=None       # Optional data transformations
)
```

**Features:**
- **Lazy Loading**: Only loads events when accessed, suitable for large files
- **Memory Efficient**: Uses file indexing to avoid loading entire dataset into memory
- **Caching**: Optional caching for smaller datasets to improve performance
- **Transform Support**: Apply custom transformations to momentum data

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
<weight id='LL'> weight_value </weight>
<weight id='LT'> weight_value </weight>
<weight id='TL'> weight_value </weight>
<weight id='TT'> weight_value </weight>
</rwgt>
</event>
</MLEvents>
```

**Event files:**
The event files for this project can be downloaded from [nextcloud](https://nextcloud.mpp.mpg.de/nextcloud/public.php/dav/files/kDKfLX54EkQPaJx/?accept=zip).
They contain a directory for LO (UU_LO), for LO including the radiation from the POWHEG Sudakov (UU_LOwS) and at NLO (UU_NLO). In each directory there are the histograms done at lhe level (`pwgLHEF_analysis-mean-W*.top`) and after showering with only qcd radiation being activated (`pwgoutput_py8_histos-mean-W*.top`). The definition of the weights can be found in the `powheg.input-save` file.

The filtered events can be found in `pwgevents-????.ml` coming from POWHEG only, i.e. after stage 4. `output_shower_events-????.ml` are the filtered events created from the showered events.

## Usage Examples

### Basic Dataset Usage

```python
from torch.utils.data import DataLoader
from pathlib import Path

# Load dataset (supports pattern matching)
file_path = Path("data/*.ml")
dataset = MLEventsDataset(file_path, cache_events=False)

# Check dataset info
print(f"Dataset contains {len(dataset)} events")
print(f"File info: {dataset.get_file_info()}")

# Access single event
features, labels = dataset[0]
print(f"Features shape: {features.shape}")  # [n_momentum_components]
print(f"Labels shape: {labels.shape}")      # [4] - LL, LT, TL, TT weights
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
    print(f"  Labels: {batch_labels.shape}")      # [batch_size, 4]

    # Your training/inference code here
    break
```

### Custom Transforms

```python
import torch

def normalize_momentum(features):
    """Example transform to normalize momentum components"""
    return (features - features.mean()) / features.std()

# Apply transform
dataset = MLEventsDataset(
    file_path="data/events.ml",
    transform=normalize_momentum
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
├── README.md                   # Project documentation
├── requirements.txt            # Python package dependencies
│
├── polarisation_tagging/       # Package source code
│   ├── __init__.py
│   ├── dataset.py              # MLEventsDataset class
│   ├── model.py                # Machine learning model definitions
│   └── train.py                # Training loop and utilities
│
└── data/                       # Sample ML events data
    ├── events_sample1.ml
    └── events_sample2.ml
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