import pandas as pd
import numpy as np
import torch
import re
import logging

from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
from pathlib import Path
from braceexpand import braceexpand

logger = logging.getLogger(__name__)

class MLEventsDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for large ML events files.
    Uses lazy loading to handle big files without loading everything into memory.
    """

    def __init__(self, file_path: Union[str, Path,List],
                 labels: List[str],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 inv_target_transform: Optional[Callable] = None,
                 cache_events: bool = False,
                 standardise: bool = False) -> None:
        """
        Args:
            file_path: Path to the ML events file.
            transform: Optional transform to be applied on features.
            target_transform: Optional transform to be applied on labels.
            cache_events: Whether to cache parsed events in memory (for smaller files).
            standardise: Whether to standardise features (not labels) using global statistics (BatchNorm should be preferred).
        """
        if isinstance(file_path, list):
            self.file_path = [Path(fp) for fp in file_path]
        else:
            self.file_path = [Path(file_path)]

        eventfiles = []
        for eventfile in self.file_path:
            eventfile = Path(eventfile)
            if '*' in str(eventfile) or '?' in str(eventfile) or '.' in str(eventfile) or '{' in str(eventfile):
                if eventfile.exists():
                    # If the event file actually exists there's nothing to expand, allowing for a list of absolute file paths to be given as input.
                    eventfiles.append(eventfile.resolve())
                elif eventfile.is_absolute():
                    raise NotImplementedError("Absolute paths with wildcards are not supported. Relative paths are though.")
                else:
                    # eventfiles += Path.cwd().glob(str(eventfile))
                    eventfiles += [Path(p).resolve() for p in braceexpand(str(eventfile))]
            else:
                eventfiles.append(Path(eventfile).resolve())

        self.eventfiles = sorted(list(set(eventfiles)))
        if not self.eventfiles:
            raise FileNotFoundError(f"No event files found for path: {file_path}")

        # self.labels           = sorted([label.upper() for label in labels])
        self.labels           = labels
        self.transform        = transform
        self.target_transform = target_transform
        self.inv_target_transform = inv_target_transform
        self.cache_events     = cache_events

        # Build index of event positions specific to each file for lazy loading.
        self.event_positions = self._build_event_index()

        # Number of events per file
        self.number_of_events = {efp: len(positions) for efp, positions in self.event_positions.items()}

        self._cached_events = {} if cache_events else None

        logger.info(f"Found {len(self)} events in {self.file_path}")

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

    def compute_global_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute global averages and standard deviations of features across all events.
        At the end of each event file there is a summary block with mean and stddev values, having the following format:
        <MLMeanValues>
        -1.702100676E-02 +-  9.606176686E-02 -3.408339058E-02 +-  9.600048558E-02 -5.791682997E-01 +-  2.615258801E-01  4.712723475E+01 +-  2.542308961E-01
        -2.034589650E-01 +-  9.526494128E-02 -1.475438737E-02 +-  9.471735668E-02 -2.811466050E-01 +-  2.643637577E-01  4.779138729E+01 +-  2.551392221E-01
        -3.598325990E-02 +-  9.504260448E-02 -3.964487896E-02 +-  9.476517172E-02 -1.927529404E-01 +-  2.911641865E-01  5.019067491E+01 +-  2.785668316E-01
        2.564632316E-01 +-  9.584894464E-02  8.848265693E-02 +-  9.596826355E-02  1.094046139E-02 +-  3.065987580E-01  5.173959160E+01 +-  2.926270836E-01
        <rwgt>
        <weight id='UU'>  0.112086574E-01 +-  0.376359522E-04 </weight>
        <weight id='LL'>  0.647128843E-03 +-  0.495244735E-05 </weight>
        <weight id='LT'>  0.132310008E-02 +-  0.802026938E-05 </weight>
        <weight id='TL'>  0.133214637E-02 +-  0.805506687E-05 </weight>
        <weight id='TT'>  0.778457063E-02 +-  0.290583370E-04 </weight>
        </rwgt>
        </MLMeanValues>
        For example, "-1.702100676E-02 +-  9.606176686E-02" is the mean and stddev of the first momentum component across all events.

        Note: Only the training data should be normalised. Thus, if the same .ml file is used for validation/test,
              this function cannot be used, as there wouldn't be a strict distinction anymore between training and
              validation/test data.
              For those cases, the means and stddevs should be computed separately on the training dataset only,
              using for example the torch.nn.BatchNorm1d or torch.nn.BatchNorm2d layers in the model.
              This is anyway more robust, since it allows to ignore the normalisation should it turn out to be not beneficial.
        Note2: For the weights, the statistical quantities which are combined here are the calculated per file set, i.e.
               this is for, example the mean of only the LL weights across all files and not the mean of the ratio LL/UU!

        returns: Tuple of (mean, stddev) dictionaries with 'features' and 'labels' keys.
            mean   = sum(mean_i) / N_files
            stddev = sqrt(sum(stddev_i^2)) / N_files
        """
        all_momentum_means   = []
        all_momentum_stddevs = []
        all_weight_means   = []
        all_weight_stddevs = []
        for eventfile_path in self.eventfiles:
            with eventfile_path.open('r') as file:
                content = file.read()

            mean_block_match = re.search(r"<MLMeanValues>(.*?)</MLMeanValues>", content, re.DOTALL)
            # Note, the re.DOTALL flag is important to make '.' match newlines as well.

            if mean_block_match:
                mean_block = mean_block_match.group(1)

                # Extract momentum means and stddevs
                momentum_lines = []
                rwgt_lines = []
                in_rwgt = False

                for line in mean_block.strip().split('\n'):
                    line = line.strip()
                    if '<rwgt>' in line:
                        in_rwgt = True
                        continue
                    elif '</rwgt>' in line:
                        in_rwgt = False
                        continue
                    elif in_rwgt:
                        rwgt_lines.append(line)
                    elif line and not line.startswith('<'):
                        momentum_lines.append(line)

                momentum_means = []
                momentum_stddevs = []
                for line in momentum_lines:
                    mom_means = re.findall(r"([\d\.\-ED\+]+\s*\+-\s*[\d\.\-ED\+]+)", line)
                    for mom_mean in mom_means:
                        mean_val, stddev_val = map(float, mom_mean.split(' +- '))
                        momentum_means.append(mean_val)
                        momentum_stddevs.append(stddev_val)

                # Extract weight means and stddevs
                weight_means = []
                weight_stddevs = []
                weight_pattern = r"<(?:weight|rwgt) id='(\w+)'>\s*([\d\.\-E\+]+)\s*\+-\s*([\d\.\-E\+]+)\s*</(?:weight|rwgt)>"
                for wweight_line in rwgt_lines:
                    weights = re.search(weight_pattern, wweight_line)
                    if weights:
                        weight_id, mean_val, stddev_val = weights.groups()
                        weight_means.append(float(mean_val))
                        weight_stddevs.append(float(stddev_val))

                # Combine momentum and weight statistics
                momentum_means   = np.array(momentum_means)
                momentum_stddevs = np.array(momentum_stddevs)
                weight_means     = np.array(weight_means)
                weight_stddevs   = np.array(weight_stddevs)

                all_momentum_means.append(momentum_means)
                all_momentum_stddevs.append(momentum_stddevs)
                all_weight_means.append(weight_means)
                all_weight_stddevs.append(weight_stddevs)

        # Compute global averages and stddevs across all files
        total_momentum_means   = np.mean(np.array(all_momentum_means), axis=0)
        total_momentum_stddevs = np.sqrt(np.sum(np.array(all_momentum_stddevs)**2, axis=0)) / len(all_momentum_stddevs)
        total_weight_means     = np.mean(np.array(all_weight_means), axis=0)
        total_weight_stddevs   = np.sqrt(np.sum(np.array(all_weight_stddevs)**2, axis=0)) / len(all_weight_stddevs)

        mean = {'features': total_momentum_means, 'labels': total_weight_means}
        stddev = {'features': total_momentum_stddevs, 'labels': total_weight_stddevs}

        return mean, stddev


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
        weight_pattern = r"<(?:weight|rwgt) id='(.+)'>\s*([\d\.\-E\+]+)\s*</(?:weight|rwgt)>"
        weights = re.findall(weight_pattern, rwgt_content)

        # Convert to ordered list [LL, LT, TL, TT]
        weight_dict = {weight_id: float(weight_val) for weight_id, weight_val in weights}

        weight_labels = []
        for label in self.labels:
            label_list = label.split('/')
            if len(label_list) > 1:
                assert len(label_list) == 2, f"Invalid label format: {label}"
                label_list = [ll.strip() for ll in label_list]
                weight_labels.append(weight_dict.get(label_list[0], 0.0) / weight_dict[label_list[1]])
            else:
                weight_labels.append(weight_dict.get(label, 0.0))

        return np.array(momenta_information), np.array(weight_labels)

    def _load_event(self, idx: int) -> Dict[str, Any]:
        """
        Load and parse a single event by index.
        Since events are stored across multiple files, we need to determine which file to read from.
        Since the event files are stored in a list, we need to map the global index to the specific file and local index.

        The event format is assumed to be:
        <event>
        -1.952240822E+01  1.619171734E+01 -3.960068023E+01  4.702669463E+01
        4.677072325E+00 -6.222946580E+01  1.290993778E+00  6.241833132E+01
        1.629003643E+01 -2.368169622E+01 -4.681346003E+01  5.493348763E+01
        -1.444700533E+00  6.971944468E+01 -3.560575124E+01  7.829851625E+01
        <rwgt>
        <weight id='UU'> 0.239419993E-01 </weight>
        <weight id='LL'> 0.634900003E-03 </weight>
        ...
        </rwgt>
        With the momentum lines containing an arbitrary number of float numbers (4 per particle),
        and the weights being specified in the <rwgt> block.
        In the specific example above, there are 4 particles (4 momentum lines), each with (px, py, pz, E).
        In the considered ZZ case here, they correspond to
        zl1 = e+,
        zl2 = e-,
        zl3 = mu+,
        zl4 = mu-.
        Args:
            idx: Global event index
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
        labels   = torch.tensor(event['labels'],   dtype=torch.float32)  # Shape: [#labels given in init]

        # # Flatten features (concatenate all momentum components)
        # features = features.flatten()  # Shape: [n_particles * 4]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            labels   = self.target_transform(labels)

        return features, labels

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


def get_statistics_from_dataset(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    fulldataloader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0
    )
    features, _ = next(iter(fulldataloader))
    overall_mean   = features.mean(dim=0)
    overall_stddev = features.std(dim=0)
    del fulldataloader

    return overall_mean, overall_stddev
