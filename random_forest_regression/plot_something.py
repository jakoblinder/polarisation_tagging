import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_events_utils import stylesheet_default
# from ml_events_utils import color_gio as color_dict
from ml_events_utils import color_deep as color_dict

import matplotlib.pyplot as plt
import numpy as np



color_dict = {key: hexwithhash for key, (hex, hexwithhash, floats) in color_dict.items()}

# import matplotlib as mpl
# Instead of 'mpl.rcParams['axes.titlesize'] = 13' and so on, just load the following stylesheet, which sets all the relevant parameters for a consistent look across all plots.

# Apply the package default style globally so all plots in this module are consistent.
plt.style.use(stylesheet_default)

plt.figure()
x = np.linspace(0, 10, 100)
for i, (key, hex) in enumerate(color_dict.items()):
    plt.plot(x, np.sin(x + i), label=key, color=hex)
plt.legend()
plt.title("Example Plot with Giovanni's Colors")
plt.xlabel("x")
plt.ylabel("sin(x + i)")
plt.tight_layout()
plt.savefig("giovanni_colors_plot.pdf")
plt.close()

