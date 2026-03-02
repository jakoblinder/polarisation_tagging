"""
Dataset class for Z+jet events with momentum features and rL labels.
Each line contains: ['Emu+', 'pxmu+', 'pymu+', 'pzmu+', 'Emu-', 'pxmu-', 'pymu-', 'pzmu-', 'Ej', 'pxj', 'pyj', 'pzj', 'rL']
"""

import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ZJetDataset(Dataset):
    """
    Dataset for Z+jet events with momentum features and rL labels.

    Data format: Each line contains 13 numbers:
    - 12 momentum components: Emu+, pxmu+, pymu+, pzmu+, Emu-, pxmu-, pymu-, pzmu-, Ej, pxj, pyj, pzj
    - 1 label: rL
    """

    def __init__(self, file_path, transform=None, target_transform=None, max_events=None, standardise=True):
        """
        Initialize the Z+jet dataset.

        Parameters:
        -----------
        file_path : str or Path
            Path to the data file containing momentum and rL information
        transform : callable, optional
            Transform to apply to the momentum features
        target_transform : callable, optional
            Transform to apply to the target (rL)
        max_events : int, optional
            Maximum number of events to load (useful for testing)
        """
        self.file_path = Path(file_path)
        self.transform = transform
        self.target_transform = target_transform

        # Load and parse the data
        self._load_data(max_events)

        # Standardisation over the whole dataset:
        if standardise:
            logger.info("Computing dataset-wide feature standardisation.")
            logger.info("This may take a moment...")
            fulldataloader = DataLoader(
                self,
                batch_size=len(self),
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            features, _ = next(iter(fulldataloader))
            self.feature_mean   = features.mean(dim=0)
            self.feature_stddev = features.std(dim=0)
            del fulldataloader
            def standardise_fn(x):
                return (x - self.feature_mean) / self.feature_stddev
            if self.transform:
                original_transform = self.transform
                self.transform = lambda x: standardise_fn(original_transform(x))
            else:
                self.transform = standardise_fn

    def _load_data(self, max_events=None):
        """
        Load data from the text file.
        Note that the structure of the momenta here is [E, px, py, pz] for each particle, whereas in the
        MLEventsDataset it is [px, py, pz, E]. We shift the energy to the end here to match that format
        of the MLEventsDataset. This is not important for the learning but we want to be able to use the same
        utility functions and models written for the MLEventsDataset.
        """
        logger.info(f"Loading Z+jet data from {self.file_path}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        # Read the data
        data = []
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_events is not None and i >= max_events:
                    break

                # Parse line: 13 numbers separated by spaces
                values = [float(x) for x in line.strip().split()]

                # Shift energy to the end for each particle to match MLEventsDataset format
                # Particles: mu+, mu-, jet; each has 4 components [E, px, py, pz]
                # We want to reorder to [px, py, pz, E] for each particle
                if len(values) != 13:
                    logger.warning(f"Warning: Line {i+1} has {len(values)} values instead of 13, skipping")
                    continue
                else:
                    values = [
                        values[1], values[2],  values[3],  values[0],  # mu+
                        values[5], values[6],  values[7],  values[4],  # mu-
                        values[9], values[10], values[11], values[8],  # jet
                        values[12]  # rL label remains at the end
                    ]


                data.append(values)

        # Convert to numpy array and split features/labels
        data = np.array(data)
        self.features = data[:, :12]  # First 12 columns are momentum features
        self.labels   = np.expand_dims(data[:, 12], axis=-1)   # Last column is rL label (fix shape to (N, 1))

        logger.info(f"Loaded {len(self.features)} events")
        logger.info(f"Feature shape: {self.features.shape}")
        logger.info(f"Label   shape: {self.labels.shape}")
        logger.info(f"Feature statistics - Mean: {self.features.mean():.3f}, Std: {self.features.std():.3f}")
        logger.info(f"Label   statistics - Mean: {self.labels.mean():.3f},   Std: {self.labels.std():.3f}, Min: {self.labels.min():.3f}, Max: {self.labels.max():.3f}")

    def __len__(self):
        """Return the number of events in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a single event.

        Returns:
        --------
        tuple : (features, label)
            features: torch.Tensor of shape (12,) containing momentum components
            label: torch.Tensor scalar containing rL value
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features and label
        features = self.features[idx]
        label    = self.labels[idx]

        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)  # Shape: [4 * 3 particles]
        label    = torch.tensor(label, dtype=torch.float32)     # Shape: [1]

        # Apply transforms if specified
        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            label = self.target_transform(label)

        return features, label

    @property
    def input_shape(self) -> torch.Size:
        """Return the shape of the input features."""
        features, _ = self[0]  # Get the first event's features
        return features.shape  # 12 momentum features

    @property
    def label_shape(self) -> torch.Size:
        """Return the shape of the labels."""
        _, label = self[0]  # Get the first event's label
        return label.shape  # Should be [1] for rL

    def get_feature_names(self):
        """Return the names of the 12 momentum features."""
        return ['Emu+', 'pxmu+', 'pymu+', 'pzmu+', 'Emu-', 'pxmu-', 'pymu-', 'pzmu-', 'Ej', 'pxj', 'pyj', 'pzj']

    def get_dataset_info(self):
        """Return information about the dataset."""
        return {
            'num_events': len(self),
            'num_features': 12,
            'feature_names': self.get_feature_names(),
            'label_name': 'rL',
            'file_path': str(self.file_path)
        }

    def get_file_info(self) -> Dict[str, Union[float, int, bool]]:
        """Get information about the dataset file."""
        return {
            'total_files_size_mb': self.file_path.stat().st_size / (1024**2),  # MB
            'num_events': len(self),
            'num_features': 12,
            'feature_names': self.get_feature_names(),
            'label_name': 'rL',
            'cache_enabled': True,
        }
