import mowl
mowl.init_jvm("5g")
import glob
import sys
from typing import Optional, List

import pandas as pd
import pkg_resources
import typer
from dp_cgans import DP_CGAN
from dp_cgans.embeddings import embed, WalkerType, ProjectionType, load_embedding, log
from dp_cgans.preprocess import download_datasets, download, preprocess
from dp_cgans.preprocess.converters import xml_to_csv
from dp_cgans.preprocess.dataset import create_training_and_test_dataset

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

    tabular_data=pd.read_csv(input_file)

    model = DP_CGAN(
        epochs=epochs, # number of training epochs
        batch_size=batch_size, # the size of each batch
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
    print(pkg_resources.get_distribution('dp_cgans').version)


@cli.command("embed")
def cli_embed(
        data: str = typer.Option("data.owl", help="Path to a Semantic Ontology Language Dataset (.owl)"),
        model_path: str = typer.Option("data.model", help="Path to where the model should be, or is, stored (.model)"),
        dim: int = typer.Option(5, help="Dimensionality of the Word Vectors used to embed the model"),
        projection_type: str = typer.Option("owl2vecstar", help="The Projection Method that is used to project the "
                                                                "semantic information into a graph the possible "
                                                                "methods are owl2vecstar, dl2vec, taxonomy and "
                                                                "taxonomy_rels"),
        bidirectional_taxonomy: bool = typer.Option(False, help="If true, then per each subClass edge one superClass "
                                                                "edge will be generated Can only be True when "
                                                                "use_taxonomy is True."),
        include_literals: bool = typer.Option(False,
                                              help="If true, then the graph will also include triples involving data "
                                                   "property assertions and annotations, for owl2vecstar"),
        only_taxonomy: bool = typer.Option(False,
                                           help="If true, then the projection will only include subClass edges, "
                                                "for owl2vecstar"),
        use_taxonomy: bool = typer.Option(False,
                                          help="If true, then taxonomies will be used. Otherwise, the relations "
                                               "parameter should be true, for taxonomy_rels"),
        relations: Optional[List[str]] = typer.Option(None, help="Contains a list of relations in string format, "
                                                                 "for taxonomy_rels Can only be non-empty when "
                                                                 "use_taxonomy is False."),
        walker_type: str = typer.Option("deepwalk",
                                        help="The walker Method that is used to learn the representations for "
                                             "vertices in the projected graph the possible methods are deepwalk and "
                                             "node2vec"),
        num_walks: int = typer.Option(10, help="The number of walks"),
        walk_length: int = typer.Option(10, help="The length of a walk"),
        workers: int = typer.Option(4, help="The amount of workers"),
        alpha: int = typer.Option(4, help="The probability of a restart, for DeepWalk"),
        p: int = typer.Option(4, help="he Return Hyperparameter, for Node2Vec"),
        q: int = typer.Option(4, help="The In-Out Hyperparameter, for Node2Vec"),
        epochs: int = typer.Option(4, help="Number of iterations (epochs) over the sentences, for Word2Vec"),
        window: int = typer.Option(4,
                                   help="Maximum distance between the current and predicted word within a sentence, "
                                        "for Word2Vec"),
        min_count: int = typer.Option(4, help="Ignores all words with total frequency lower than this, for Word2Vec"),
        verbose: bool = typer.Option(False,
                                     help="If True, then intermediate console message will be displayed to indicate "
                                          "progress."),
        load: bool = typer.Option(False,
                                  help="If True, then the model_path will be used to load in an already embedded "
                                       "model. Otherwise, it'll create a new embedded model."),
):
    if load:
        return load_embedding(model_path)
    else:
        selected_projection_type = ProjectionType.OWL2VECSTAR
        for projection in ProjectionType:
            if projection.value is projection_type:
                selected_projection_type = projection
                break

        selected_walker_type = WalkerType.DEEPWALK
        for walker in WalkerType:
            if walker.value is walker_type:
                selected_walker_type = walker
                break

        return embed(data=data, model_path=model_path, dim=dim, verbose=verbose,
                     projection_type=selected_projection_type, bidirectional_taxonomy=bidirectional_taxonomy,
                     include_literals=include_literals, only_taxonomy=only_taxonomy, use_taxonomy=use_taxonomy,
                     relations=relations, walker_type=selected_walker_type, num_walks=num_walks,
                     walk_length=walk_length, workers=workers, alpha=alpha, p=p, q=q, epochs=epochs, window=window,
                     min_count=min_count)


@cli.command("download")
def cli_download(
    url: Optional[str] = typer.Option(None, help="The source to download from"),
    location_path: str = typer.Option(None, help="The directory path to store the downloaded file to"),
    name: Optional[str] = typer.Option(None, help="The name the file should be saved as (including extension)"),
    verbose: bool = typer.Option(True, help="Display logs"),
    default: bool = typer.Option(False, help="Whether or not to automatically download the necessary files.")
):
    if url is None and default is False:
        log(text=f'❌️Failed to download file. You must either give an url or have the "--default" flag set.',
            verbose=True)
        return

    if default is True:
        log(text=f'⚠️Ignoring "--url" and "--name" flags.', verbose=verbose)
        download_datasets(location_path=location_path, verbose=verbose)
    else:
        download(url=url, location_path=location_path, file_name=name, verbose=verbose)


@cli.command("xml_to_csv")
def cli_download(
    source: str = typer.Option(None, help="The XML file path to read"),
    target: str = typer.Option(None, help="The file path to write the CSV file to"),
    verbose: bool = typer.Option(True, help="Display logs"),
):
    xml_to_csv(source, target, verbose)


@cli.command("preprocess")
def cli_download(
    file: str = typer.Option(None, help="The file path of the CSV to be preprocessed"),
    destination: str = typer.Option(None, help="The directory path where the processed data should be saved to"),
    verbose: bool = typer.Option(True, help="Display logs"),
):
    preprocess(file, destination, verbose)


@cli.command("split")
def cli_download(
    file: str = typer.Option(None, help="The file path of the hp.obo file"),
    verbose: bool = typer.Option(True, help="Display logs"),
):
    create_training_and_test_dataset(file, verbose)


if __name__ == "__main__":
    cli()