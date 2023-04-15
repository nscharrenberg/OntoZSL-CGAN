import pandas as pd
import typer
from src.dp_cgans import DP_CGAN
from src.dp_cgans.embeddings import owl2vec

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
        input_file: str,
        gen_size: int = typer.Option(100, help="Number of rows in the generated samples file"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        batch_size: int = typer.Option(1000, help="Batch size"),
        output: str = typer.Option("synthetic_samples.csv", help="Path to the output"),
        verbose: bool = typer.Option(True, help="Display logs")
):
    tabular_data = pd.read_csv(input_file)

    model = DP_CGAN(
        epochs=epochs,  # number of training epochs
        batch_size=batch_size,  # the size of each batch
        log_frequency=True,
        verbose=verbose,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        private=False,
    )

    if verbose: print(f'🗜️  Model instantiated, fitting...')
    model.fit(tabular_data)

    if verbose: print(f'🧪 Model fitted, sampling...')
    sample = model.sample(gen_size)

    sample.to_csv(output)
    if verbose: print(f'✅ Samples generated in {BOLD}{GREEN}{output}{END}')


@cli.command("version")
def cli_version():
    print('ontoZSL')
    # print(pkg_resources.get_distribution('dp_cgans').version)


@cli.command("embed")
def cli_embed(
        data: str = typer.Option("data.owl", help="The path specifying the location of the .owl dataset."),
        model: str = typer.Option("owl2vec.model",
                                  help="The path specifying the location where the model will be stored "
                                       "OR where the model is located."),
        dim: int = typer.Option(5, help="The Embedding dimension i.e. vector size."),
        load: bool = typer.Option(False,
                                        help="True if you want to load an existing model, False if you want to create "
                                             "a new model"),
        graph: bool = typer.Option(False, help="Show a graph of the embedded model")
):
    owl2vec.main(data, model, dim, load, graph)


if __name__ == "__main__":
    cli()
