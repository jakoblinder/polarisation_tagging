# Some functions usefule for handling the data for plotting.

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, ClassVar
from dataclasses import dataclass, field

@dataclass
class HistogramData:
    observable: str
    name: str
    order: str
    left_edges: np.ndarray
    right_edges: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    style: Dict[str, Any] = field(default_factory=dict)
    colors: ClassVar[Dict[str, str]]

    def __post_init__(self):
        if not self.style:
            self.style = {
                "color": self.colors.get(self.name, "black"),
            }
        else:
            if "color" not in self.style:
                self.style["color"] = self.colors.get(self.name, "black")

    @property
    def bins(self) -> np.ndarray:
        bins = np.zeros(len(self.left_edges) + 1)
        bins[:-1] = self.left_edges
        bins[-1]  = self.right_edges[-1]
        return bins

    def __mul__(self, other):
        if type(self) is type(other):
            if not np.allclose(self.bins, other.bins):
                raise ValueError("Cannot multiply histograms with different bins.")
            new_values = self.values * other.values
            # Propagate errors assuming they are uncorrelated:
            new_errors = np.sqrt((self.errors * other.values)**2 + (self.values * other.errors)**2)
            new_name = f"{self.name} * {other.name}"
        elif isinstance(other, (int, float)):
            new_values = self.values * other
            new_errors = self.errors * other
            new_name = f"{self.name} * {other}"
        else:
            raise ValueError(f"Unsupported type ({type(other)}) for multiplication with HistogramData.")

        return HistogramData(
            observable  = self.observable,
            name        = new_name,
            order       = self.order,
            left_edges  = self.left_edges,
            right_edges = self.right_edges,
            values      = new_values,
            errors      = new_errors,
            style       = self.style.copy(),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(self) is type(other):
            if not np.allclose(self.bins, other.bins):
                raise ValueError("Cannot divide histograms with different bins.")
            new_values = self.values / other.values
            # Propagate errors assuming they are uncorrelated:
            new_errors = np.sqrt((self.errors / other.values)**2 + ( self.values * other.errors / other.values**2 )**2)
            new_name = f"{self.name} / {other.name}"
        elif isinstance(other, (int, float, np.ndarray)):
            new_values = self.values / other
            new_errors = self.errors / other
            new_name = f"{self.name} / {other}"
        else:
            raise ValueError(f"Unsupported type ({type(other)}) for division with HistogramData.")

        return HistogramData(
            observable  = self.observable,
            name        = new_name,
            order       = self.order,
            left_edges  = self.left_edges,
            right_edges = self.right_edges,
            values      = new_values,
            errors      = new_errors,
            style       = self.style.copy(),
        )

    def __add__(self, other):
        if type(self) is type(other):
            if not np.allclose(self.bins, other.bins):
                raise ValueError("Cannot add histograms with different bins.")
            new_values = self.values + other.values
            # Propagate errors assuming they are uncorrelated:
            new_errors = np.sqrt(self.errors ** 2 + other.errors ** 2)
            new_name = f"{self.name} + {other.name}"
        elif isinstance(other, (int, float, np.ndarray)):
            new_values = self.values + other
            new_errors = self.errors
            new_name = f"{self.name} + {other}"
        else:
            raise ValueError(f"Unsupported type ({type(other)}) for addition with HistogramData.")

        return HistogramData(
            observable  = self.observable,
            name        = new_name,
            order       = self.order,
            left_edges  = self.left_edges,
            right_edges = self.right_edges,
            values      = new_values,
            errors      = new_errors,
            style       = self.style,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(self) is type(other):
            if not np.allclose(self.bins, other.bins):
                raise ValueError("Cannot subtract histograms with different bins.")
            new_values = self.values - other.values
            # Propagate errors assuming they are uncorrelated:
            new_errors = np.sqrt(self.errors ** 2 + other.errors ** 2)
            new_name = f"{self.name} - {other.name}"
        elif isinstance(other, (int, float, np.ndarray)):
            new_values = self.values - other
            new_errors = self.errors
            new_name = f"{self.name} - {other}"
        else:
            raise ValueError(f"Unsupported type ({type(other)}) for subtraction with HistogramData.")

        return HistogramData(
            observable  = self.observable,
            name        = new_name,
            order       = self.order,
            left_edges  = self.left_edges,
            right_edges = self.right_edges,
            values      = new_values,
            errors      = new_errors,
            style       = self.style.copy(),
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

        tolerance_fac = 1. + 1e-8
        # Make sure the new bins are not smaller than the old ones:
        assert abs(bins[1] - bins[0])*tolerance_fac >= abs(self.bins[1] - self.bins[0]), "The new bins have to be wider than the old."

        def add_two_bins(point1, point2):
            # Check if the two points have one commmon edge.
            if np.isclose(point1[1], point2[0], rtol=tolerance_fac):
                # point1 is on the left of point2, so the new bin will be [point1[0], point2[1]]
                pass
            elif np.isclose(point1[0], point2[1], rtol=tolerance_fac):
                # point2 is on the left of point1, so the new bin will be [point2[0], point1[1]]
                # swap the points to make point1 the left one:
                point1, point2 = point2, point1
            else:
                raise ValueError(f"The two points, {point1} and {point2}, do not have a common edge, so they cannot be merged into one bin.")

            w1      = abs(point1[1] - point1[0])
            h1      = point1[2]
            h1_stat = point1[3]
            w2      = abs(point2[1] - point2[0])
            h2      = point2[2]
            h2_stat = point2[3]

            new_value = (h1 * w1 + h2 * w2) / (w1 + w2)
            # Combine statistical uncertainties: Add them quadratically.
            new_error = np.sqrt( ((w1 * h1_stat)**2 + (w2 * h2_stat)**2)) / (w1 + w2)

            return [point1[0], point2[1], new_value, new_error]


        new_points = []

        jpoint = 0
        for ibin in range(len(bins) - 1):
            # Set new point to the first point which is in the bin:
            if jpoint >= len(self.values):
                break

            # Initialize the new point with the first point which is in the bin.
            new_point = [self.left_edges[jpoint], self.right_edges[jpoint], self.values[jpoint], self.errors[jpoint]]
            new_points.append(new_point)
            jpoint += 1

            while jpoint < len(self.values):
                right_edge     = self.right_edges[jpoint]
                new_right_edge = bins[ibin + 1]
                # Make sure that a zero edge is not considered to be smaller than the new right edge, which could lead to numerical issues.
                if right_edge * tolerance_fac > new_right_edge and not (abs(right_edge) < 1e-9 and abs(new_right_edge) < 1e-9):
                    break
                # Add up all points which are in the bin as well:
                combined_point = add_two_bins(new_points[-1], [self.left_edges[jpoint], self.right_edges[jpoint], self.values[jpoint], self.errors[jpoint]])
                new_points[-1] = combined_point
                jpoint += 1

        # Set the left and right to the numerical values of the bin array, which has to be numerically compatible with the new left and right edges, but is more robust against numerical issues.
        for ipoint in range(len(new_points)):
            assert np.isclose(new_points[ipoint][0], bins[ipoint], rtol=tolerance_fac), f"The left edge of the new point, {new_points[ipoint][0]}, is not close to the left edge of the new bin, {bins[ipoint]}."
            assert np.isclose(new_points[ipoint][1], bins[ipoint + 1], rtol=tolerance_fac), f"The right edge of the new point, {new_points[ipoint][1]}, is not close to the right edge of the new bin, {bins[ipoint + 1]}."
            new_points[ipoint][0] = bins[ipoint]
            new_points[ipoint][1] = bins[ipoint + 1]

        new_points = np.array(new_points)
        new_left_edges  = new_points[:, 0]
        new_right_edges = new_points[:, 1]
        new_values      = new_points[:, 2]
        new_errors      = new_points[:, 3]


        return HistogramData(
                observable  = self.observable,
                name       = self.name,
                order       = self.order,
                left_edges  = new_left_edges,
                right_edges = new_right_edges,
                values      = new_values,
                errors      = new_errors,
                style       = self.style
            )


def read_histogram(path: Path,
                   rescaling_factor: float,
                   observable_map: Dict[str, str],
                   name: str,
                   order:str,
                   style: Dict[str, Any] = {}
                   ) -> Dict[str, HistogramData]:
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
                if len(parts) == 4:
                    # .top file
                    values = np.array(list(map(float, parts[:])))
                elif len(parts) >= 5:
                    # .dat file
                    values = np.array(list(map(float, parts[1:5])))
                else:
                # elif len(parts) != 4:
                    in_histogram = False
                    histograms[aligned_observable_name] = HistogramData(
                        observable  = aligned_observable_name,
                        name        = name,
                        order       = order,
                        left_edges  = np.array(histograms[aligned_observable_name])[:, 0],
                        right_edges = np.array(histograms[aligned_observable_name])[:, 1],
                        values      = np.array(histograms[aligned_observable_name])[:, 2],
                        errors      = np.array(histograms[aligned_observable_name])[:, 3],
                        style       = style,
                    )
                    continue

                values[2:4] *= rescaling_factor
                histograms[aligned_observable_name].append(values)



        # Handle the last histogram if the file ends while reading histogram data
        if in_histogram and aligned_observable_name is not None and isinstance(histograms[aligned_observable_name], list):
            histograms[aligned_observable_name] = HistogramData(
                observable  = aligned_observable_name,
                name        = name,
                order       = order,
                left_edges  = np.array(histograms[aligned_observable_name])[:, 0],
                right_edges = np.array(histograms[aligned_observable_name])[:, 1],
                values      = np.array(histograms[aligned_observable_name])[:, 2],
                errors      = np.array(histograms[aligned_observable_name])[:, 3]
            )

    return histograms
