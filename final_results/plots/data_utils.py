# Some functions usefule for handling the data for plotting.

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, ClassVar
from dataclasses import dataclass, field

@dataclass
class HistogramData:
    observable: str
    model: str
    order: str
    left_edges: np.ndarray
    right_edges: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    styles: Dict[str, Any] = field(default_factory=dict)
    model_colors: ClassVar[Dict[str, str]]

    def __post_init__(self):
        if not self.styles:
            self.styles = {
                "color": self.model_colors.get(self.model, "black"),
            }

    @property
    def bins(self) -> np.ndarray:
        bins = np.zeros(len(self.left_edges) + 1)
        bins[:-1] = self.left_edges
        bins[-1]  = self.right_edges[-1]
        return bins

    def __truediv__(self, other):
        if type(self) is type(other):
            if not np.allclose(self.bins, other.bins):
                raise ValueError("Cannot divide histograms with different bins.")
            new_values = self.values / other.values
            # Propagate errors assuming they are uncorrelated:
            new_errors = np.sqrt((self.errors / other.values) ** 2 + (self.values * other.errors / other.values ** 2) ** 2)
            return HistogramData(
                observable  = self.observable,
                model       = f"{self.model} / {other.model}",
                order       = self.order,
                left_edges  = self.left_edges,
                right_edges = self.right_edges,
                values      = new_values,
                errors      = new_errors,
                styles      = self.styles,
            )

    def rebin(self, width_bins: float|list = None):
        """Rebin the data set, by merging all bins which have the same borders.
           This is useful for example to integrate a histogram.
        """
        if isinstance(width_bins, float) or isinstance(width_bins, int):
            # Get the bins:
            width_bins = float(width_bins)
            bins = [self.left_edges[0], ]
            while bins[-1] < self.right_edges[-1]:
                bins.append(bins[-1] + width_bins)
        elif isinstance(width_bins, list):
            bins = np.array(width_bins)
        elif isinstance(width_bins, np.ndarray):
            bins = width_bins
        else:
            raise AssertionError("width_bins has to be either a float or a list of bin edges.")

        # Make sure the new bins are not smaller than the old ones:
        assert abs(bins[1] - bins[0]) >= abs(self.bins[1] - self.bins[0]), "The new bins have to be wider than the old."

        def add_two_bins(bin1, bin2):
            new_value = bin1[0] + bin2[0]
            new_error = np.sqrt(bin1[1]**2 + bin2[1]**2)
            return [new_value, new_error]


        new_points = []

        jpoint = 0
        for ibin in range(len(bins) - 1):
            # Set new point to the first point which is in the bin:
            if jpoint >= len(self.values):
                break

            # Initialize the new point with the first point which is in the bin.
            # The edges are already set to the new bin edges.
            new_point = [bins[ibin], bins[ibin + 1], self.values[jpoint], self.errors[jpoint]]

            new_points.append(new_point)
            jpoint += 1

            while jpoint < len(self.values):
                if self.right_edges[jpoint] > bins[ibin + 1]:
                    break
                # Add up all points which are in the bin as well:
                new_points[-1] = [new_points[-1][0], new_points[-1][1], *add_two_bins(new_points[-1][2:4], [self.values[jpoint], self.errors[jpoint]])]
                jpoint += 1

        new_points = np.array(new_points)
        new_left_edges  = new_points[:, 0]
        new_right_edges = new_points[:, 1]
        new_values      = new_points[:, 2]
        new_errors      = new_points[:, 3]


        return HistogramData(
                observable  = self.observable,
                model       = self.model,
                order       = self.order,
                left_edges  = new_left_edges,
                right_edges = new_right_edges,
                values      = new_values,
                errors      = new_errors,
                styles      = self.styles
            )


def read_histogram(path: Path,
                   rescaling_factor: float,
                   observable_map: Dict[str, str],
                   model: str,
                   order:str
                   ) -> Dict[str, Any]:
    """
    Read a histogram file and return a dictionary of histograms.
    Look for lines starting with, for example, `# dphiee` and ignore the possible `index <some number>` which would be there in a proper POWHEG histogram.
    """
    histograms = {}
    aligned_observable_name = None
    with open(path, "r") as f:
        in_histogram = False
        for line in f:
            line = line.strip()
            if not line and not in_histogram:
                continue

            if line.startswith("#") and not in_histogram:
                match = re.match(r"#\s*([a-zA-Z0-9_-]+)\s*", line)
                if not match:
                    continue
                observable = match.group(1)
                aligned_observable_name = observable_map.get(observable, observable)
                histograms[aligned_observable_name] = []
                in_histogram = True
                continue

            if in_histogram:
                for char in ["d", "D", "E"]:
                    line = re.sub(rf"{char}", "e", line)

                parts = line.split()
                if len(parts) != 4:
                    in_histogram = False
                    histograms[aligned_observable_name] = HistogramData(
                        observable = aligned_observable_name,
                        model      = model,
                        order      = order,
                        left_edges =  np.array(histograms[aligned_observable_name])[:, 0],
                        right_edges = np.array(histograms[aligned_observable_name])[:, 1],
                        values     =  np.array(histograms[aligned_observable_name])[:, 2],
                        errors     =  np.array(histograms[aligned_observable_name])[:, 3]
                    )
                    continue

                values = np.array(list(map(float, parts[:])))
                values[2:4] *= rescaling_factor
                histograms[aligned_observable_name].append(values)

        # Handle the last histogram if the file ends while reading histogram data
        if in_histogram and aligned_observable_name is not None and isinstance(histograms[aligned_observable_name], list):
            histograms[aligned_observable_name] = HistogramData(
                observable = aligned_observable_name,
                model      = model,
                order      = order,
                left_edges =  np.array(histograms[aligned_observable_name])[:, 0],
                right_edges = np.array(histograms[aligned_observable_name])[:, 1],
                values     =  np.array(histograms[aligned_observable_name])[:, 2],
                errors     =  np.array(histograms[aligned_observable_name])[:, 3]
            )

    return histograms
