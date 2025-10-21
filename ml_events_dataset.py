import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
from pathlib import Path

class MLEventsDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for large ML events files.
    Uses lazy loading to handle big files without loading everything into memory.
    """

    def __init__(self, file_path: Union[str, Path,List], labels: List[str], transform: Optional[Callable] = None, target_transform: Optional[Callable]=None, cache_events: bool = False) -> None:
        """
        Args:
            file_path: Path to the ML events file
            transform: Optional transform to be applied on features
            cache_events: Whether to cache parsed events in memory (for smaller files)
        """
        if isinstance(file_path, list):
            self.file_path = [Path(fp) for fp in file_path]
        else:
            self.file_path = [Path(file_path)]

        eventfiles = []
        for eventfile in self.file_path:
            if '*' in str(eventfile) or '?' in str(eventfile):
                if eventfile.is_absolute():
                    raise NotImplementedError("Absolute paths with wildcards are not supported. Relative paths are though.")
                else:
                    eventfiles += Path.cwd().glob(str(eventfile))
            else:
                eventfiles.append(Path(eventfile).resolve())

        self.eventfiles = sorted(list(set(eventfiles)))
        if not self.eventfiles:
            raise FileNotFoundError(f"No event files found for path: {file_path}")

        self.labels           = sorted([label.upper() for label in labels])
        self.transform        = transform
        self.target_transform = target_transform
        self.cache_events     = cache_events

        # Build index of event positions specific to each file for lazy loading
        self.event_positions = self._build_event_index()

        # Number of events per file
        self.number_of_events = {efp: len(positions) for efp, positions in self.event_positions.items()}

        self._cached_events = {} if cache_events else None

        print(f"Found {len(self)} events in {self.file_path}")

    def _build_event_index(self) -> List[Tuple[int, int]]:
        """Build an index of event start/end positions in the file for efficient access."""
        event_positions = {}

        for eventfile_path in self.eventfiles:
            event_positions[eventfile_path] = []
            with eventfile_path.open('r') as file:
                pos = 0
                in_event = False
                event_start = 0

                for line in file:
                    line_start = pos
                    pos += len(line)

                    if '<event>' in line:
                        in_event = True
                        event_start = line_start
                    elif '</event>' in line and in_event:
                        event_positions[eventfile_path].append((event_start, pos))
                        in_event = False

        return event_positions

    def _parse_single_event(self, event_content: str) -> Tuple[List[List[float]], List[float]]:
        """Parse a single event string and extract features and labels."""
        momenta_information = []
        rwgt_content = ""

        lines = event_content.strip().split('\n')
        in_rwgt = False

        for line in lines:
            line = line.strip()
            if '<rwgt>' in line:
                in_rwgt = True
                continue
            elif '</rwgt>' in line:
                in_rwgt = False
                continue
            elif in_rwgt:
                rwgt_content += line + '\n'
            elif line and not line.startswith('<') and not line.startswith('#'):
                # Check if line contains momentum data (arbitrary number of numbers)
                try:
                    numbers = [float(x) for x in line.split()]
                except ValueError:
                    raise ValueError(f"Non-numeric data found in momentum line: {line}")

                momenta_information += numbers

        # Extract reweight values
        weight_pattern = r"<(?:weight|rwgt) id='(\w+)'>\s*([\d\.\-E\+]+)\s*</(?:weight|rwgt)>"
        weights = re.findall(weight_pattern, rwgt_content)

        # Convert to ordered list [LL, LT, TL, TT]
        weight_dict = {weight_id: float(weight_val) for weight_id, weight_val in weights}

        weight_labels = []
        for label in self.labels:
            weight_labels.append(weight_dict.get(label, 0.0))

        return np.array(momenta_information), np.array(weight_labels)

    def _load_event(self, idx: int | list[int]) -> Dict[str, Any]:
        """Load and parse a single event by index.
        Since events are stored across multiple files, we need to determine which file to read from.
        Since the event files are stored in a list, we need to map the global index to the specific file and local index.
        """
        # Make sure that idx is a list
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))

        assert len(idx) == 1, "Only single index access is supported."

        i = idx[0]
        # Check cache first
        if self._cached_events and i in self._cached_events:
            event_data = self._cached_events[i]
        else:
            # Determine which file and local index
            cumulative_events = 0
            for eventfile_path in self.eventfiles:
                num_events_in_file = self.number_of_events[eventfile_path]
                if i < cumulative_events + num_events_in_file:
                    local_index = i - cumulative_events
                    local_event_file = eventfile_path
                    break
                else:
                    cumulative_events += num_events_in_file

            # Read event from file
            start_pos, end_pos = self.event_positions[local_event_file][local_index]

            with local_event_file.open('r') as file:
                file.seek(start_pos)
                event_content = file.read(end_pos - start_pos)

            # Parse event
            momenta_information, weight_labels = self._parse_single_event(event_content)

            event_data = {
                'features': momenta_information,
                'labels':   weight_labels
            }

            # Cache if enabled
            if self._cached_events is not None:
                self._cached_events[i] = event_data

        return event_data

    def __len__(self) -> int:
        return sum(self.number_of_events.values())

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Load event lazily
        event = self._load_event(idx)

        # Convert to tensors
        features = torch.tensor(event['features'], dtype=torch.float32)  # Shape: [4 * n_particles]
        labels   = torch.tensor(event['labels'], dtype=torch.float32)      # Shape: [4]

        # # Flatten features (concatenate all momentum components)
        # features = features.flatten()  # Shape: [n_particles * 4]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels   = self.target_transform(labels)

        return features, labels


    def get_file_info(self) -> Dict[str, Union[float, int, bool]]:
        """Get information about the dataset file."""
        total_file_size = 0
        for eventfile in self.eventfiles:
            total_file_size += eventfile.stat().st_size / (1024**2)  # MB
        return {
            'total_files_size_mb': total_file_size,
            'num_events': len(self),
            'cache_enabled': self.cache_events
        }