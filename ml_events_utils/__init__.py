from pathlib import Path
from matplotlib import font_manager

from .ml_events_dataset import MLEventsDataset, get_statistics_from_dataset
from .Zjet_events_dataset import ZJetDataset
from .transforms import scale_target, boost_into_four_lepton_cm_frame, find_scale_var_ratios, log_target_transform, exp_target_transform, boost_into_Zjet_cm_frame
from .train_loop import train_loop, valid_loop
from .logger import setup_file_logger, log_file
from .run_settings import Parameter, Settings

__version__ = "0.1.0"
# print(f"Importing {__name__} package, version {__version__}")

stylesheet_tex    = [Path(__file__).parent / Path('stylesheet.mpl')]
stylesheet_no_tex = [Path(__file__).parent / Path('stylesheet_no_tex.mpl')]

stylesheet_default = stylesheet_no_tex

# In the following, we define some colour schemes for use in the notebooks. The colours are stored as hex strings, hex strings with a leading '#', and RGB tuples with values in [0, 1] (for use in Matplotlib).
# Giovanni's colors:
color_gio = {"black":  "000000",  # full
             "gray":   "808080",  # unpolarised
             "red":    "EB3323",  # LL, long
             "yellow": "D2A641",  # LT, right-handed
             "green":  "377D22",  # TL, left-handed
             "blue":   "001EF5",  # TT
             "pink":   "EB46F8",  # sum of pols
            }
color_gio = {key: [hex, f"#{hex}", [int(hex[:2], 16)/ 255., int(hex[2:4], 16)/ 255., int(hex[4:6], 16)/ 255., ]] for key, hex in color_gio.items()}

# 'deep' palette from Seaborn. Colorblind friendly.
color_deep = {"blue":   "4c72b0",
              "orange": "dd8452",
              "green":  "55a868",
              "red":    "c44e52",
              "purple": "8172b3",
              "brown":  "937860",
              "pink":   "da8bc3",
              "gray":   "8c8c8c",
              "yellow": "ccb974",
              "cyan":   "64b5cd",
              "black":  "000000"
            }
color_deep = {key: [hex, f"#{hex}", [int(hex[:2], 16)/ 255., int(hex[2:4], 16)/ 255., int(hex[4:6], 16)/ 255., ]] for key, hex in color_deep.items()}


# Add fonts from the fonts directory to Matplotlib's font manager
try:
    font_dir = Path(__file__).parent / Path('fonts')
    font_files = list(font_dir.glob('**/*.ttf')) + list(font_dir.glob('**/*.otf'))
    # print(f"Found {len(font_files)} font files in {font_dir}: {[font.name for font in font_files]}")
except Exception as e:
    logger.warning(f"Could not find fonts in {font_dir}: {e}")
    font_files = []

for font_path in font_files:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # print(f"Added font {font_path.name}: {prop.get_name()}")

