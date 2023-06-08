import mowl
mowl.init_jvm("5g")

from dp_cgans.onto_dp_cgan_init import ONTO_DP_CGAN
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import create_path
from dp_cgans.utils.logging import log, LogLevel

from dp_cgans.ontology import Preprocessor
from dp_cgans.ontology.zsl.classifier import ZeroShotLearning

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
        config: str = typer.Option("configs/dp_cgans/default.json",
                                   help="The path location of the configuration file."),
):
    _config = load_config(config)
    input_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'), _config.get_nested('dp_cgans', 'files', 'input'), create=False)
    output_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'), _config.get_nested('dp_cgans', 'files', 'output'), create=True)
    verbose = _config.get_nested('dp_cgans', 'verbose')
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
        config: str = typer.Option("configs/dp_cgans/onto.json",
                                   help="The path location of the configuration file."),
):
    _config = load_config(config)
    input_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'), _config.get_nested('dp_cgans', 'files', 'input'), create=False)
    output_path = create_path(_config.get_nested('dp_cgans', 'files', 'directory'), _config.get_nested('dp_cgans', 'files', 'output'), create=True)
    verbose = _config.get_nested('dp_cgans', 'verbose')
    gen_size = _config.get_nested('dp_cgans', 'gen_size')

    tabular_data = pd.read_csv(input_path)

    model = ONTO_DP_CGAN(_config)

    if verbose: print(f'🗜️  Model instantiated, fitting...')
    model.fit(tabular_data)

    if verbose: print(f'🧪 Model fitted, sampling...')
    sample = model.sample(gen_size)

    sample.to_csv(output_path)
    if verbose: print(f'✅ Samples generated in {BOLD}{GREEN}{output_path}{END}')


@cli.command("version")
def cli_version():
    print(pkg_resources.get_distribution('dp_cgans').version)


@cli.command("preprocess")
def cli_preprocess(
        config: str = typer.Option("configs/preprocessing/config.json",
                                   help="The path location of the configuration file."),
):
    pipeline = Preprocessor(config)
    pipeline.start()


@cli.command("embed")
def cli_embed(
        config: str = typer.Option("configs/embedding/config.json",
                                   help="The path location of the configuration file."),
):
    pipeline = Embedding(config)
    pipeline.start()


@cli.command("zsl")
def cli_zsl(
        config: str = typer.Option("configs/zsl/config.json", help="The path location of the configuration file."),
):
    pipeline = ZeroShotLearning(config)
    pipeline.start()

    test_samples = pipeline.unseen["features"]
    converted_text = pipeline.tab_to_text(test_samples)
    embeddings = pipeline.model.encode(converted_text)

    predictions = []
    scores = []

    for embedding in embeddings:
        pred, score = pipeline.predict(embedding)
        predictions.append(pred[0])
        scores.append(score[0])

    log(text=f"pred: {predictions} - score: {scores}", level=LogLevel.INFO)


if __name__ == "__main__":
    cli()
