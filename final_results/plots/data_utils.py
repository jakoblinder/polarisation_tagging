# Some functions usefule for handling the data for plotting.

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, ClassVar
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

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
    labels: ClassVar[Dict[str, str]]

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

    def __getitem__(self, key):
        """Get a subset of the data set by slicing the points with a slice object or by giving an index.
           If key == slice(float,float,(float)), the start and stop values of the slice are interpreted as the borders of
           the bins to be sliced, and the step is interpreted as the width of the bins for rebinning.
           In this case, the start and stop values have to be between the borders of the first and last point.

        Args:
            key (int | slice): _description_

        Returns:
            DataSet: New data set with the sliced points.
        """
        onestep = "onestep"
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            start = key.start if (key.start or key.start == 0) else None
            stop  = key.stop  if key.stop else None
            if any([isinstance(s, float) for s in [start, stop]]):
                # assert isinstance(key.step, int) or not key.step, "The step has to be integer."

                n_points = len(self.values)

                start_index = 0
                stop_index  = n_points
                for i in range(n_points):
                    if (start or abs(start) < 1e-10) and (start < self.left_edges[i] or abs(start - self.left_edges[i]) < 1e-10):
                        break
                    start_index +=1

                if stop == onestep:
                    stop_index = start_index + 1
                else:
                    for i in range(n_points-1, -1, -1):
                        if (not stop) or stop >= self.right_edges[i] or abs(stop - self.right_edges[i]) < 1e-10:
                            break
                        stop_index -= 1

                start = start_index
                stop  = stop_index

            partial_histogram =  HistogramData(
                                    observable  = self.observable,
                                    name        = self.name,
                                    order       = self.order,
                                    left_edges  = self.left_edges[start:stop],
                                    right_edges = self.right_edges[start:stop],
                                    values      = self.values[start:stop],
                                    errors      = self.errors[start:stop],
                                    style       = self.style.copy(),
                                )

            if isinstance(key.step, float) or isinstance(key.step, np.ndarray):
                partial_histogram = partial_histogram.rebin(width_bins=key.step)
            else:
                if key.step is not None and key.step != 1:
                    raise TypeError(f"The step has to be a float, since its used for rebinning, but it is of type {type(key.step)}.")

            return partial_histogram

        elif isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self.points)
            if key < 0 or key >= len(self.points):
                raise IndexError(f"The index {key:d} is out of range.")
            return self.__getitem__(slice(key,key+1))
        elif isinstance(key, float):
            # Find bin which contains the given value.
            return self.__getitem__(slice(key, onestep))
        else:
            raise TypeError(f"Invalid argument type, {type(key)}.")

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
                style       = self.style.copy(),
            )

    @staticmethod
    def format_value_uncertainty(value: float, unc: float, unc_sig_digits: int = 1) -> str:
        """Format a measurement as value(unc), with uncertainty rounded to <unc_sig_digits> significant digits.

        Example: 160.88362999999998 +- 0.15495453 -> 160.9(2)
        """
        def round_half_up(value: float, decimals: int = 0) -> float:
            """
            Round using decimal ROUND_HALF_UP (0.5 always rounds away from zero - no banker's rounding as e.g. in np.round).
            decimals: number of decimal places to round to (default: 0, i.e. round to integer).
                    Example: 24567.98765 with decimals=2 -> 24567.99, with decimals = -3 -> 25000.0.
            """
            d = Decimal(str(value))
            quant = Decimal(f"1e{(-decimals)}")
            # print(f"Rounding {value} to {decimals} decimal places: {d} quantized to {quant} with ROUND_HALF_UP gives {d.quantize(quant, rounding=ROUND_HALF_UP)}")
            return float(d.quantize(quant, rounding=ROUND_HALF_UP))

        if unc <= 0:
            return f"{value}"

        exponent = int(np.floor(np.log10(abs(unc))))
        decimals = -exponent + (unc_sig_digits - 1)

        unc_rounded   = round_half_up(unc, decimals)
        value_rounded = round_half_up(value, decimals)

        if decimals > 0:
            unc_digits = int(round_half_up(unc_rounded * (10 ** decimals), 0))
            return f"{value_rounded:.{decimals}f}({unc_digits})"

        # decimals <= 0: uncertainty is an integer at this precision
        unc_digits = int(round_half_up(unc_rounded, 0))
        return f"{int(round_half_up(value_rounded, 0))}({unc_digits})"

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
