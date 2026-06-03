import argparse
import sys
from pathlib import Path

from .run_settings import Settings
from .models import model_dict


# def namespace_from_settings(run_settings: Settings) -> argparse.Namespace:
#     return argparse.Namespace(**{key: parameter.value for key, parameter in run_settings.items()})

def _get_parser_defaults(parser: argparse.ArgumentParser) -> dict:
    """Extract all default values from a parser without parsing arguments."""
    defaults = {}
    for action in parser._actions:
        # Skip positional arguments and the help action
        if action.dest != 'help' and action.option_strings:
            if action.default is not argparse.SUPPRESS:
                defaults[action.dest] = action.default
    return defaults

def _is_yaml_file(path_str: str) -> bool:
    """Check if a string refers to an existing YAML file."""
    if not isinstance(path_str, str):
        return False
    path = Path(path_str)
    return path.suffix in ['.yaml', '.yml'] and path.exists()

def _create_parser(parser_type: str = "train") -> argparse.Namespace:
    """Prepare the run settings by parsing command-line arguments.

    Args:
        parser_type (str): Type of parser to use, either "train" or "test". Determines which command-line arguments are expected.
    """
    if parser_type == "train":
        parser = argparse.ArgumentParser(
            description="Train a neural network for polarisation tagging.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("mlfiles", nargs='+', type=Path,        action="store",               help=".ml files to be used for training.")
        parser.add_argument("-m", "--model", type=str, action="store", default="FFNN_paper_BatchNorm", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")

        # Training hyperparameters
        parser.add_argument("-o", "--optimizer",      type=str,   action="store", default="paper", help="Optimizer to use. Options: SGD, Adam, AdamW, RMSprop, paper, paper_momentum.")
        parser.add_argument("-e", "--epochs",         type=int,   action="store", default=1000,    help="Number of training epochs.")
        parser.add_argument("-l", "--learning_rate",  type=float, action="store", default=1e-3,    help="Learning rate for the optimizer.")
        parser.add_argument("-p", "--patience",       type=int,   action="store", default=25,      help="Early stopping patience.")
        parser.add_argument("--penalties", nargs="*", type=str,   action="store", default=[],      help="Specify which penalty terms to include in the loss function. Options: cross_section, ZdecayAngles.")

        # Dataloading settings
        parser.add_argument("-s", "--seed",      type=int,            action="store", default=42, help="Random seed for reproducibility.")
        parser.add_argument("--no-cache-events", dest="cache_events", action="store_false",       help="Disable caching of events in the dataset (default: cache the events).")

        # Run settings
        parser.add_argument("--dont_test", dest="do_test", action="store_false",                 help="Run the test script after training with the best model weights found during training.")

        # Simplified running options
        parser.add_argument("--replot", dest="replot_only", action="store_true", help="Only regenerate the training history plot from existing CSV files. The model and potentially the output directory need to be specified.")

    elif parser_type == "test":
        parser = argparse.ArgumentParser(
            description='Test the already trained neural network for polarisation tagging.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("mlfiles", nargs="*", type=Path, action="store", help=".ml files to be used for training. Not required when using --replot.")
        parser.add_argument("model",              type=str,  action="store", help=f"Model architecture to use. Options: {list(model_dict.keys())}.")
        parser.add_argument("model_weight_file",  type=Path, action="store", help="Path to the .pt(y) file containing the trained model weights.")

        parser.add_argument("--inputdir",         type=Path, action="store", default=Path().cwd(), help='Specify name of input directory.')
        parser.add_argument("--histogram_dir",    type=Path, action="store", default=None,         help='Directory containing the .top histogram files for comparison (They are in the folder where also the events are.).')


    if parser_type in ["train", "test"]:
        parser.add_argument("-g", "--gpu",        type=int, action="store", default=-1,  help="Specify manually which of the available gpus is supposed to be used.")
        parser.add_argument("-b", "--batch_size", type=int, action="store", default=512, help="Batch size for training.")
        parser.add_argument("-n", "--nworkers",   type=int, action="store", default=0,   help="Number of workers for DataLoader.")
        parser.add_argument("--outputdir", type=Path,      action="store", default=Path().cwd(), help="Specify name of output directory.")

        # Dataloading settings
        parser.add_argument("--standardise",        dest="standardise",           action="store_true",                   help="Enable standardisation of features over the whole dataset (default).")
        parser.add_argument("--input_choice",       type=str,                     action="store",      default=None,     help="Choice of input features. Options: Momenta, jan2026.")
        parser.add_argument("--polarisation",       type=str,                     action="store",      default="LL",     help="Specify which polarisation to train on (Only relevant for ZZ). Options: LL, LT, TL, TT, UL, LU.")
        parser.add_argument("--n_generated_events", type=lambda x: int(float(x)), action="store",      default=int(1e7), help="Number of generated events for comparison (1e7 for LO and LOwS and 5e6 for NLO).")
        parser.add_argument("--useZjet",            dest="use_zjet",              action="store_true",                   help="Use Z+jet dataset instead of default.")
        parser.add_argument("--showered",           dest="showered",              action="store_true",                   help="This run used showered events instead of parton level events (default: use parton level events). Important for plotting.")

        # Create a mutually exclusive group for specifying the reference frame
        frame_group = parser.add_mutually_exclusive_group()
        frame_group.add_argument("--labframe", dest="labframe", default=True,              action="store_true",  help="Use lab frame instead of partonic CMS.")
        frame_group.add_argument("--cmframe",  dest="labframe", default=argparse.SUPPRESS, action="store_false", help="Use partonic CMS instead of lab frame.")

        # Simplified running options
        parser.add_argument("--verbose",                           action="store_true", help="Print to stdout as well")
        parser.add_argument("-t", "--test_mode", dest="test_mode", action="store_true", help="Run in test mode (only one data point to test implementation of the model).")

    return parser

def prepare_run_settings(parser_type:str="train") -> Settings:
    """Prepare the run settings by parsing command-line arguments and optionally loading from a YAML file.


    Args:
        parser_type (str): Type of parser to use, either "train" or "test". Determines which command-line arguments are expected.
    """
    parser_type = parser_type.lower()
    if parser_type not in ["train", "test"]:
        raise ValueError(f"Invalid parser_type '{parser_type}'. Expected 'train' or 'test'.")

    # Check if first argument is a YAML file
    yaml_settings  = None
    yaml_file_path = None

    if len(sys.argv) > 1 and _is_yaml_file(sys.argv[1]):
        yaml_file_path = sys.argv[1]
        try:
            yaml_settings = Settings.load_yaml(yaml_file_path)
            # Remove YAML path from sys.argv so argparse doesn't see it
            sys.argv.pop(1)
        except Exception as e:
            raise ValueError(f"Failed to load YAML configuration from '{yaml_file_path}': {e}")

    parser = _create_parser(parser_type)

    if yaml_settings is  None:
        # No YAML file provided, parse command-line arguments as usual
        run_settings = Settings(argparse=parser.parse_args())
    else:
        # YAML file provided, use it to create Settings instance
        run_settings = yaml_settings
        if sys.argv[1:]:
            # If there are additional command-line arguments, they are expected to be in an "--key value" format and override YAML settings.
            assert len(sys.argv[1:]) % 2 == 0, "Additional command-line arguments must be in '--key value' pairs."
            cli_settings = {}
            for i in range(1, len(sys.argv), 2):
                key = sys.argv[i].lstrip("--")
                value = sys.argv[i + 1]
                cli_settings[key] = value
            run_settings.update(cli_settings)  # Override YAML settings with CLI settings

        parser_defaults = _get_parser_defaults(parser)
        for key, default_value in parser_defaults.items():
            run_settings.set_default(key, default_value)

    # Set some defaults, which can be set by the yaml file but are not expected to be set by the command line.
    run_settings.set_default("split_ratios",       [0.6, 0.2, 0.2])
    # The histogram_dir should be set for training and testing, since the testing is often done after training.
    run_settings.set_default("histogram_dir", run_settings.mlfiles.value[0].parent)

    if parser_type == "train":
        run_settings.set_default("test_standardisation",   False)
        run_settings.set_default("count_negative_weights", False)

        if not run_settings.replot_only.value and len(run_settings.mlfiles.value) == 0:
            raise ValueError("mlfiles are required when not using --replot")

        # The inputdir, used for testing and plotting, should be set to the outputdir, where the trained model is going to end up.
        run_settings.set_default("inputdir", run_settings.outputdir.value)
        # For backward compatibility, set model_dir to outputdir if not already set.
        run_settings.set_default("model_dir", run_settings.outputdir.value)

    elif parser_type == "test":
        run_settings.set_default("model_weight_file", Path(f"{run_settings.model.value}_model_weights_best.pt"))
        run_settings.set_default("outputdir", run_settings.inputdir.value)

    return run_settings



