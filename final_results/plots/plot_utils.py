# Some function useful for plotting.
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

repo_path = Path(__file__).parent.parent.parent.resolve()
# Add repository to path
sys.path.insert(0, str(repo_path))

from ml_events_utils import stylesheet_default
# from ml_events_utils import color_gio as color_dict
from ml_events_utils import color_deep as color_dict

network_labels = {
    "autoencoder": r"AE",
    "ffnn":        r"FFNN",
    "pn":          r"PN",
    "powheg":      r"POWHEG",
    "rfr":         r"RFR",
    "rfrct":       r"RFR$_{\mathrm{ct}}$",  # Used to compare different input features for the RFR. (rfr normally corresponds to rfrct.)
    "rfrep":       r"RFR$_{\mathrm{ep}}$",  # Used to compare different input features for the RFR.
    "dsim":        r"direct sim.",          # Direct simulation of the process with POWHEG, used for comparison with the reweighting approach.
}

network_colors = {
    "Autoencoder": color_dict["green"],
    "FFNN":        color_dict["red"],
    "PN":          color_dict["orange"],
    "POWHEG":      color_dict["black"],
    "RFR":         color_dict["blue"],
    "RFRct":       color_dict["blue"],
    "RFRep":       color_dict["pink"],
    "dsim":        color_dict["gray"],
}

order_latex = {
    "lo":   r"$\mathrm{LO}$",
    "lows": r"$\mathrm{LO}+\mathrm{Sud.}$",
    "lops": r"$\mathrm{LO}+\mathrm{PS}$",
    "nlo":  r"$\mathrm{NLO}$",
    "nlows":r"$\mathrm{NLO}+\mathrm{Sud.}$",
    "nlops":r"$\mathrm{NLO}+\mathrm{PS}$",
}

observables_latex = {
    "totxsec": r"$\sigma_{\mathrm{tot}}$",
    "ptep":    r"$p_{\mathrm{T}, \, e^{+}}$",
    "yep":     r"$y_{e^{+}}$",
    "cthep":   r"$\cos \theta^{*}_{e^{+}}$",
    "mepem":   r"$m_{e^{+} e^{-}}$",
    "dphiee":  r"$\Delta \phi_{e^{+} e^{-}}$",  # Azimuthal angle difference between e+ e-, coming from one of the Z bosons.
    "ptee":    r"$p_{\mathrm{T}, \, e^{+} e^{-}}$",
    "pt4l":    r"$p_{\mathrm{T}, \, 4l}$",
    "rll":     r"$r_{\mathrm{LL}}$",
}

def create_subplots(n_plots, rcParams):
    mpl.rcParams.update(rcParams)
    size = mpl.rcParams['figure.figsize']
    if n_plots == 1:
        fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=size)
        axs = [axs,]
    else:
        scale_factor  = 0.5 + 0.5*(n_plots-1)
        height_ratios = [3.,] + [1,]*(n_plots-1)
        fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=(size[0], scale_factor*size[1]), height_ratios=height_ratios)
    return fig, axs

def move_offset_factor(ax, ylabel):
    # Move the y-axis offset text (the "x 1e-3" part) into the y-axis label and hide the original offset text to avoid overlap with the title.
    ax.figure.draw_without_rendering()
    offset = ax.yaxis.get_major_formatter().get_offset()
    offset = r" / $" + offset[7:] if offset else ""
    ax.yaxis.set_label_text(ylabel + offset)
    ax.yaxis.offsetText.set_visible(False)

    # Move the y-axis offset text (the "x 1e-3" part) down a bit to avoid overlap with the x-axis label.
    # axs[0].yaxis.get_offset_text().set_y(0.5)

def add_legend(ax, legloc:str="", **kwargs):
    """Set legend of a plot.

    Args:
        ax (_type_): Ax object of the plot for which the legend should be set.
        legloc (str, optional): Specify manually the location of the legend. Defaults to "best".
        Select from ['left', 'center', 'right'] and combine with ['upper', 'center', 'lower'], like
            'upper right'
        leg_edge (bool, optional): Specifiy wether edge of legend should be shown. Defaults to False.
        leg_bg_alpha (float, optional): Specify transparency of background. Defaults to transparent, i.e. 0.0.
        ncol (int, optional): Number of columns for the legend. Defaults to 1 (labels stacked vertically).
            Set to a higher value to arrange labels horizontally next to each other.
    """
    handles, labels = ax.get_legend_handles_labels()
    try:
        new_handles = [plt.Line2D([], [], ls=h.get_linestyle(), c=h.get_edgecolor(), linewidth=h.get_linewidth()) for h in handles]
    except AttributeError:
        new_handles = [plt.Line2D([], [], marker=h.get_marker(), ls=h.get_linestyle(), c=h.get_color(), linewidth=h.get_linewidth()) for h in handles]

    ncol = kwargs.get("ncol", 1)

    if legloc:
        ax.legend(handles = new_handles, labels=labels, loc=legloc, ncol=ncol)
    else:
        ax.legend(handles = new_handles, labels=labels, ncol=ncol)

    leg = ax.get_legend()
    if kwargs.get("leg_edge", False):
        leg.get_frame().set_linewidth(1.0)
    else:
        leg.get_frame().set_linewidth(0.0)

    gray = [0.655, 0.655, 0.659]
    leg.get_frame().set_edgecolor(gray)

    # Change alpha of background without changing the transparency of the frame edge.
    if bg := kwargs.get("leg_bg_alpha", False):
        facealpha = bg
    else:
        facealpha = 0.0
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor(([1,1,1], facealpha))

def add_uncertainty_bands(ax, bs, ws, sigs_stat, sigs_syst=[], clr: str = 'black', ls: str = '-', bands: bool = True):
    """Plots uncertainty bands or error bars on a given axis.

        ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the uncertainty bands or error bars.
        bs (array-like): Bin edges for the data.
        ws (array-like): Bin values or weights corresponding to the bins.
        sigs_stat (array-like): Statistical uncertainties for each bin.
        sigs_syst (array-like, optional): Systematic uncertainties for each bin. Defaults to an empty list.
        clr (str, optional): Color of the uncertainty bands or error bars. Defaults to 'black'.
        ls (str, optional): Line style for the uncertainty boundaries. Defaults to '-'.
        bands (bool, optional): If True, plots uncertainty bands; otherwise, plots error bars. Defaults to True.

    Returns:
        None
    """
    if bands:
        # axs[0].bar(x=bs[:-1], height=2*sigs_stat, bottom=ws-sigs_stat, width=np.diff(bs), align='edge', linewidth=0, alpha=0.25, zorder=-1, color=color_hist0)
        # Take care of ws having possibly one more element than sigs_stat, due to prehandling of bins and values for step plots.
        if len(ws) == len(sigs_stat) + 1:
            sigs_stat = np.append(sigs_stat, sigs_stat[-1])
            bins = bs
        else:
            # len(ws) == len(sigs_stat) == len(bs) - 1
            bins = bs[:-1]

        up   = ws + sigs_stat
        down = ws - sigs_stat

        ax.fill_between(bins, down, up, step='post', color=clr, alpha=0.25, zorder=-1)
        ax.step(bins, up,   where='post', color=clr, linestyle=ls, alpha=0.5, marker='')
        ax.step(bins, down, where='post', color=clr, linestyle=ls, alpha=0.5, marker='')
    else:
        ax.errorbar(bs[1:], ws, yerr=sigs_stat, marker="", linestyle="", color=clr)

def plot_histograms(ax, weights_sig, labels, styles:list=[], uncertainties=False):
    """
    Plot histograms on the given axis.

    Parameters:
        ax: The axis to plot on.
        weights_sig: A list of tuples (bin_edges[0:n+1], weights[0:n], sigs_stat[0:n]) for each histogram to plot.
        labels: A list of labels for the histograms.
        styles: A list of style dictionaries for each histogram (e.g. color, linestyle).
                To change, for example, only the color of the second histogram, you can provide styles=[{}, {"color": "red"}, {}, ...].
        uncertainties: Whether to plot uncertainty bands.
    """
    if labels is None:
        labels = [None,] * len(weights_sig)
    if not styles:
        styles = [{},] * len(weights_sig)

    default_style = {"alpha": 1.0, "marker": "", "linestyle": "-"}  # Color is set by the color cycle or by the user-provided styles.
    for i, (bs, ws, sigs_stat) in enumerate(weights_sig):
        # Use step instead of hist to avoid edge lines at the sides
        style_settings = default_style.copy()
        style_settings.update(styles[i])

        # Make sure ws has the same length as bs by appending the last value.
        ws   = np.append(ws, ws[-1])
        line = ax.step(bs, ws, where='post', label=labels[i], **style_settings)

        if uncertainties:
            hist_colour    = line[0].get_color()
            hist_linestyle = line[0].get_linestyle()
            add_uncertainty_bands(ax, bs, ws, sigs_stat, clr=hist_colour, ls=hist_linestyle, bands=True)

