import mowl

from dp_cgans.utils import Config

mowl.init_jvm("5g")

from dp_cgans.onto_dp_cgan_init import ONTO_DP_CGAN
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import create_path, load_csv, get_or_create_directory
from dp_cgans.utils.logging import log, LogLevel

from dp_cgans.ontology import Preprocessor
from dp_cgans.ontology.zsl import ZeroShotLearning

import pandas as pd
import pkg_resources
import typer
from dp_cgans import DP_CGAN

cli = typer.Typer()

# Variables to make the prints gorgeous:
BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[32m'


# RED = '\033[91m'
# YELLOW = '\033[33m'
# CYAN = '\033[36m'
# PURPLE = '\033[95m'
# BLUE = '\033[34m'


@cli.command("gen")
def cli_gen(
        config: str or Config = typer.Option("configs/dp_cgans/default.json",
                                             help="The path location of the configuration file."),
):
    _config = load_config(config)

    if _config.get_nested('dp_cgans', 'files', 'use_root'):
        input_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'),
                                 _config.get_nested('dp_cgans', 'files', 'input'), create=False)
        output_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'),
                                  _config.get_nested('dp_cgans', 'files', 'output'), create=True)
    else:
        input_path = _config.get_nested('dp_cgans', 'files', 'input')
        output_path = _config.get_nested('dp_cgans', 'files', 'output')

    verbose = _config.get_nested('verbose')
    gen_size = _config.get_nested('dp_cgans', 'gen_size')

    tabular_data = pd.read_csv(input_path)

    model = DP_CGAN(_config)

    if verbose: print(f'🗜️  Model instantiated, fitting...')
    model.fit(tabular_data)

    if verbose: print(f'🧪 Model fitted, sampling...')
    sample = model.sample(gen_size)

    sample.to_csv(output_path)
    if verbose: print(f'✅ Samples generated in {BOLD}{GREEN}{output_path}{END}')


@cli.command("onto")
def cli_onto(
        config: str or Config = typer.Option("configs/dp_cgans/onto.json",
                                   help="The path location of the configuration file."),
):
    _config = load_config(config)
    input_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'),
                             _config.get_nested('dp_cgans', 'files', 'input'), create=False)
    output_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'),
                              _config.get_nested('dp_cgans', 'files', 'output'), create=True)
    verbose = _config.get_nested('verbose')

    if verbose is None or not isinstance(verbose, bool):
        verbose = False

    gen_size = _config.get_nested('dp_cgans', 'gen_size')

    tabular_data = pd.read_csv(input_path)

    model = ONTO_DP_CGAN(_config)

    log(text=f'🗜️  Model instantiated, fitting...', verbose=verbose)
    model.fit(tabular_data)

    log(text=f'🧪 Model fitted, sampling...', verbose=verbose)
    sample = model.sample(gen_size)

    sample.to_csv(output_path, encoding="utf-8",
                  index=False,
                  header=True)
    log(text=f'✅ Samples generated in {BOLD}{GREEN}{output_path}{END}', verbose=verbose)


@cli.command("version")
def cli_version():
    print(pkg_resources.get_distribution('dp_cgans').version)


@cli.command("preprocess")
def cli_preprocess(
        config: str or Config = typer.Option("configs/preprocessing/config.json",
                                             help="The path location of the configuration file."),
):
    pipeline = Preprocessor(config)
    pipeline.start()


@cli.command("zsl")
def cli_zsl(
        config: str = typer.Option("configs/dp_cgans/onto.json", help="The path location of the configuration file."),
):
    pipeline = ZeroShotLearning(config)
    pipeline.fit_or_load()

    config = load_config(config)
    directory = config.get_nested('dp_cgans', 'files', 'directory')
    training_file = create_path(directory, config.get_nested('dp_cgans', 'files', 'output'),
                                create=False)
    test_features, test_classes = load_csv(path=training_file,
                                           class_header=config.get_nested('zsl', 'class_header'),
                                           verbose=True)

    predictions, scores = pipeline.predict(test_features)

    log(text=f"pred: {predictions} - score: {scores}", level=LogLevel.INFO)


if __name__ == "__main__":
    cli()
