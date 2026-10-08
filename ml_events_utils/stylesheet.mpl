# Matplotlib style for scientific plotting
# This is the base style for "SciencePlots"
# see: https://github.com/garrettj403/SciencePlots
# Adjusted by me
# Always combine this style with one of the other

# ---------------------------------------------------------------------------
# General figure setup
# ---------------------------------------------------------------------------
figure.dpi : 200

# Set default figure size
# Matplotlib default: (6.4, 4.8) = (16.256 cm, 12.192 cm) = (4 : 3)
# Change it to (6.8 cm, 5.1 cm) = (2.677 in, 2.007 in) = (4 : 3)
# so that two figures side by side fit into one column (~ 7 cm) with spacing.
# Try out (13.6 cm, 10.2 cm) = (5.354 in, 4.016 in) = (4 : 3) for one-column plots.
figure.figsize : 5.354, 4.016
# figure.figsize : 2.677, 2.007
# figure.figsize : 6.6, 5.0


# ---------------------------------------------------------------------------
# Axes and text
# ---------------------------------------------------------------------------
axes.formatter.limits: -2, 4  # use scientific notation if log10(axis_range)
                               # is smaller than the first or larger than the second
axes.formatter.use_mathtext : True
axes.titlelocation: left       # alignment of the title: {left, right, center}
axes.titley: 1.0               # title position in axes coordinates
axes.titlepad: 6.0             # pad between axes and title in points

# Final effective typography values (deduplicated, keeping the later choices).
axes.titlesize: 13
axes.labelsize: 13
font.size: 13

# Specify possible serif fonts
font.serif : CMU Serif Roman, CMU Serif, CMU Classical Serif, Times New Roman, Times, DejaVu Serif, Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Palatino, Charter, serif
font.family: serif
font.weight: normal  # = 400; CMU has no lighter Serif face (200 only triggered findfont warnings)

# Set x axis
xtick.direction : in
xtick.major.size : 3
xtick.major.width : 0.75
xtick.minor.size : 1.5
xtick.minor.width : 0.75
xtick.minor.visible : True
xtick.top : True
axes.spines.top: True
axes.xmargin: 0.0

# Set y axis
ytick.direction : in
ytick.major.size : 3
ytick.major.width : 0.75
ytick.minor.size : 1.5
ytick.minor.width : 0.75
ytick.minor.visible : True
ytick.right : True
axes.spines.right: True

# Set line widths
axes.linewidth : 0.75
lines.linewidth : 1.5  # 2.0


# ---------------------------------------------------------------------------
# LaTeX block (keep all LaTeX-related keys here for easy on/off switching)
# ---------------------------------------------------------------------------
# Enable LaTeX for math formatting.
# This can avoid issues where symbols like "<" and ">" are rendered incorrectly in PDF.
text.usetex : True
text.latex.preamble : \usepackage{amsmath} \usepackage{amssymb}

# Deactivate LaTeX and use mathtext instead:
# text.usetex : False
# mathtext.fontset : cm


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend.loc: best
legend.frameon: False       # if True, draw legend on a background patch
legend.shadow: False        # if True, give background a shadow effect
legend.markerscale: 1.0     # relative size of legend markers vs. original
legend.borderaxespad: 0.25   # border between axes and legend edge

# Remaining legend options kept for quick activation:
# legend.framealpha: 0.8
# legend.facecolor: inherit
# legend.edgecolor: 0.8
# legend.fancybox: True
# legend.numpoints: 1
# legend.scatterpoints: 1
# legend.fontsize: medium
# legend.labelcolor: None
# legend.title_fontsize: None
# legend.borderpad: 0.4
# legend.labelspacing: 0.5
# legend.handlelength: 2.0
# legend.handleheight: 0.7
# legend.handletextpad: 0.8
# legend.columnspacing: 2.0


# ---------------------------------------------------------------------------
# Errorbar plots
# ---------------------------------------------------------------------------
errorbar.capsize: 2  # length of end cap on error bars in pixels


# ---------------------------------------------------------------------------
# Subplot geometry
# ---------------------------------------------------------------------------
# The figure subplot parameters. All dimensions are fractions of figure width/height.
figure.subplot.left    : 0.150
figure.subplot.bottom  : 0.150
figure.subplot.right   : 0.9
figure.subplot.top     : 0.9
figure.subplot.wspace  : 0.2
figure.subplot.hspace  : 0.0
# figure.autolayout : True


# ---------------------------------------------------------------------------
# Saving figures
# ---------------------------------------------------------------------------
# The default savefig parameters can be different from display parameters,
# e.g. higher resolution or forced white background.
savefig.dpi: 600          # figure dots per inch or 'figure'
savefig.facecolor: auto   # figure face color when saving
savefig.edgecolor: auto   # figure edge color when saving
savefig.format: pdf       # {png, ps, pdf, svg}
savefig.bbox: tight       # {tight, standard}
savefig.pad_inches: 0.05  # padding when bbox is set to 'tight'
# savefig.directory: ~
# savefig.transparent: True


# ---------------------------------------------------------------------------
# Backend-specific options (kept as commented reference)
# ---------------------------------------------------------------------------
### macosx backend params
# macosx.window_mode : system   # {system, tab, window}

### tk backend params
# tk.window_focus: False

### ps backend params
# ps.papersize: letter
# ps.useafm: False
# ps.usedistiller: False
# ps.distiller.res: 6000
# ps.fonttype: 42

### PDF backend params
# pdf.compression: 6
# pdf.fonttype: 42
# pdf.use14corefonts: False
# pdf.inheritcolor: False

### SVG backend params
# svg.image_inline: True
# svg.fonttype: path
# svg.hashsalt: None

### pgf parameter
## See https://matplotlib.org/stable/tutorials/text/pgf.html for more information.
# pgf.rcfonts: True
# pgf.preamble:
# pgf.texsystem: xelatex

### docstring params
# docstring.hardcopy: False


# ---------------------------------------------------------------------------
# Color / linestyle / marker cycle
# ---------------------------------------------------------------------------
axes.prop_cycle : (cycler('color', ['EB3323', 'D2A641', '377D22', '001EF5', 'EB46F8', '808080', '000000']*5) + cycler('ls', ['solid', (0, (5, 1)), 'dotted', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]*7) + cycler('marker', ['.', '*', 'x', 'd', 'v']*7))

# Alternative cycles:
# axes.prop_cycle : (cycler('color', ['000000', '808080', 'EB3323', 'D2A641', '377D22', '001EF5', 'EB46F8']*5) + cycler('ls', ['solid', (0, (5, 1)), 'dotted', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]*7) + cycler('marker', ['.', '*', 'x', 'd', 'v']*7))

# Giovanni's colours:
# black  = '000000' (full)
# grey   = '808080' (unpolarised)
# red    = 'EB3323' (LL)
# yellow = 'D2A641' (LT)
# green  = '377D22' (TL)
# blue   = '001EF5' (TT)
# pink   = 'EB46F8' (sum of pols)

# Linestyle references:
# See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# linestyle_str = [
#      ('solid', 'solid'),
#      ('dotted', 'dotted'),
#      ('dashed', 'dashed'),
#      ('dashdot', 'dashdot')]

# linestyle_tuple = [
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
