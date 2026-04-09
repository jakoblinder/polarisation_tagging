import numpy as np
from pathlib import Path

class HistogramIndexTracker():
    """
    Utility class to track histogram indices for .top files.
    Each histogram gets a unique index, and this class helps manage them.
    """
    def __init__(self, start_index=0):
        self.current_index = start_index
        self.index_map = {}

    def get_index(self, observable_name: str):
        if observable_name not in self.index_map:
            self.index_map[observable_name] = self.current_index
            self.current_index += 1

        return self.index_map[observable_name]


class TopFileWriter():
    """
    Utility class to write .top files for comparison with POWHEG.
    It can create a multiple .top files with multiple histograms for different curves.
    """

    def __init__(self, basepath: Path = Path(""), start_index=0):
        self.index_tracker = HistogramIndexTracker(start_index=0)
        # Histogram data is stored as a dict of {output_path: {observable_name: (bin_edges, values, uncertainties)}}}
        self.histograms = {}
        self.basepath = basepath

    def write_histogram(self, output_path, observable_name: str, bin_edges:list[list], values:list, uncertainties:list=None):
        if output_path not in self.histograms:
            self.histograms[output_path] = {}

        bin_edges = np.array(bin_edges)
        values    = np.array(values)
        if uncertainties is None:
            uncertainties = np.zeros_like(values)
        else:
            uncertainties = np.array(uncertainties)

        nbins = len(bin_edges)
        assert len(values) == nbins, f"Length of values {len(values)} does not match number of bins {nbins} for observable {observable_name} in {output_path}"
        assert len(uncertainties) == nbins, f"Length of uncertainties {len(uncertainties)} does not match number of bins {nbins} for observable {observable_name} in {output_path}"

        values        = np.array(values).reshape(nbins, 1)
        uncertainties = np.array(uncertainties).reshape(nbins, 1)

        self.histograms[output_path][observable_name] = np.column_stack([bin_edges, values, uncertainties])
        # print(self.histograms[output_path][observable_name])

    def save_all(self):
        def to_fortran_notation(value):
            """Convert to Fortran D notation."""
            return f"{value: 14.8e}".replace('e', 'D').replace('E', 'D')

        for output_path, histograms in self.histograms.items():
            # Sort histograms by index before writing to file
            histograms = dict(sorted(histograms.items(), key=lambda x: self.index_tracker.get_index(x[0])))
            write_path = self.basepath / output_path if self.basepath else output_path
            with open(write_path, 'w') as f:
                for observable_name, data in histograms.items():
                    index = self.index_tracker.get_index(observable_name)
                    f.write(f"# {observable_name} index      {index}\n")
                    for row in data:
                        left, right, val, unc = row
                        f.write(f" {to_fortran_notation(left)} {to_fortran_notation(right)} "
                                f"{to_fortran_notation(val)} {to_fortran_notation(unc)}\n")
                    f.write("\n\n")

