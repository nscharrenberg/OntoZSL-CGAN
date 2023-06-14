import os.path

import mowl

from dp_cgans.experiments.ExperimentModelType import ExperimentModelType
from dp_cgans.experiments.decision_tree_model import DecisionTreeModel
from dp_cgans.experiments.logistic_regression_model import LogisticRegressionModel
from dp_cgans.experiments.random_forest_model import RandomForestModel
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

    sample.to_csv(output_path, encoding="utf-8",
                  index=False,
                  header=True)
    if verbose: print(f'✅ Samples generated in {BOLD}{GREEN}{output_path}{END}')


@cli.command("onto")
def cli_onto(
        config: str or Config = typer.Option("configs/dp_cgans/onto.json",
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


@cli.command("experiments")
def cli_experiments(
        loader: str = typer.Option(None,
                                   help="The path location of the file that specifies all the configurations to be loaded"),
        preprocess: bool = typer.Option(False, help="Perform the Preprocessing Experiments"),
        gen: bool = typer.Option(False, help="Perform the Generation Experiments (DP-CGANs)"),
        onto: bool = typer.Option(False, help="Perform the Generation Experiments (ONTO-CGANs)"),
        evaluate: bool = typer.Option(False, help="Perform Evaluation Steps")
):
    if preprocess:
        experiment_type = "Preprocessing"
    elif evaluate:
        experiment_type = "Evaluation"
    elif gen:
        experiment_type = "Generation"
    elif onto:
        experiment_type = "Ontology Generation"
    else:
        raise Exception("Unable to perform experiments when \"--preprocess\" or \"--gen\" is not defined")

    log(text=f"About to starts {experiment_type} Experiments!", level=LogLevel.INFO)

    log(text=f"Reading {experiment_type} Experiment Loader...")
    loader_config = load_config(loader)

    if preprocess:
        experiments_preprocess(loader_config)
    elif evaluate:
        experiments_evaluate(loader_config)
    elif gen:
        if onto:
            experiments_onto_cgans(loader_config)
        else:
            experiments_dp_cgans(loader_config)


def experiments_evaluate(loader_config: str or Config):
    models = loader_config.get_nested('models')

    for model_name in models:
        model_enum = ExperimentModelType(model_name)

        local_config = loader_config

        result_file_name = f"results_{model_name}.csv"
        local_config.config['results']['file'] = result_file_name

        if model_enum == ExperimentModelType.LOGISTIC_REGRESSION:
            model = LogisticRegressionModel(local_config)
        elif model_enum == ExperimentModelType.DECISION_TREE:
            model = DecisionTreeModel(local_config)
        elif model_enum == ExperimentModelType.RANDOM_FOREST:
            model = RandomForestModel(local_config)
        else:
            raise NotImplementedError("Unknown model classifier is selected for evaluation.")

        model.fit()
        model.evaluate()

        log(f"Model Evaluated: {model_name}", level=LogLevel.INFO)


def experiments_preprocess(loader_config: str or Config):
    patients_per_rd_list = loader_config.get_nested('settings', 'patients_per_rd')
    max_columns_list = loader_config.get_nested('settings', 'max_columns')

    for patients_per_rd in patients_per_rd_list:
        log(f"Starting Patients Per RD \"{patients_per_rd}\".", level=LogLevel.INFO)

        for max_columns in max_columns_list:
            log(f"Starting Max Columns \"{max_columns}\".", level=LogLevel.INFO)

            config = load_config(loader_config.get_nested('config'))

            # (patients_per_id)_(max_columns)
            current_directory_name = f'{patients_per_rd}_{max_columns}'
            directory = get_or_create_directory(
                config.get_nested('directory') % current_directory_name)
            config.config['directory'] = directory

            config.config['generator']['patients_per_rd'] = patients_per_rd
            config.config['generator']['max_columns'] = max_columns

            iteration_information = f"patients_per_id: {patients_per_rd}\nmax_columns: {max_columns}"

            with open(os.path.join(directory, 'info.txt'), "w") as f:
                f.write(iteration_information)

            cli_preprocess(config)

            log(f"Finished Max Columns \"{max_columns}\".", level=LogLevel.OK)
        log(f"Finished Patients Per RD \"{patients_per_rd}\".", level=LogLevel.OK)
    log(f"All Preprocessing Experiments have been completed!", level=LogLevel.INFO)


def experiments_dp_cgans(loader_config: str or Config):
    dimensions_list = loader_config.get_nested('settings', 'dimensions')
    batch_size_list = loader_config.get_nested('settings', 'batch_size')
    discriminator_steps_list = loader_config.get_nested('settings', 'discriminator_steps')
    epochs_list = loader_config.get_nested('settings', 'epochs')
    gen_size_list = loader_config.get_nested('settings', 'gen_size')
    preprocessing_list = loader_config.get_nested('settings', 'preprocessing')

    for preprocessing in preprocessing_list:
        log(f"Starting Preprocessing \"{preprocessing}\".", level=LogLevel.INFO)

        preprocessing_directory = get_or_create_directory(
            f"{loader_config.get_nested('preprocessing', 'directory')}" % preprocessing, error=True)

        for batch_size in batch_size_list:
            log(f"Starting Batch Size \"{batch_size}\".", level=LogLevel.INFO)
            for gen_size in gen_size_list:
                log(f"Starting Gen Size \"{gen_size}\".", level=LogLevel.INFO)
                for dimensions in dimensions_list:
                    log(f"Starting Dimensions \"{dimensions}\".", level=LogLevel.INFO)
                    for epochs in epochs_list:
                        log(f"Starting Epochs \"{epochs}\".", level=LogLevel.INFO)
                        for discriminator_steps in discriminator_steps_list:
                            log(f"Starting Discriminator Steps \"{discriminator_steps}\".", level=LogLevel.INFO)

                            config = load_config(loader_config.get_nested('config'))

                            # (preprocessing)_(batch_size)_(gen_size)_(dimensions)_(epochs)_(discriminator_steps)
                            current_directory_name = f'{preprocessing}_{batch_size}_{gen_size}_{dimensions}_{epochs}_{discriminator_steps}'
                            directory = get_or_create_directory(
                                config.get_nested('dp_cgans', 'files', 'directory') % current_directory_name)
                            config.config['dp_cgans']['files']['directory'] = directory

                            input_data_path = create_path(preprocessing_directory,
                                                          config.get_nested('dp_cgans', 'files', 'input'), create=False)
                            config.config['dp_cgans']['files']['input'] = input_data_path

                            output_data_path = create_path(directory,
                                                           config.get_nested('dp_cgans', 'files', 'output'),
                                                           create=True)
                            config.config['dp_cgans']['files']['output'] = output_data_path

                            config.config['dp_cgans']['files']['use_root'] = False
                            config.config['dp_cgans']['batch_size'] = batch_size
                            config.config['dp_cgans']['gen_size'] = gen_size
                            config.config['dp_cgans']['dimensions'] = dimensions
                            config.config['dp_cgans']['epochs'] = epochs
                            config.config['dp_cgans']['discriminator_steps'] = discriminator_steps

                            iteration_information = f"preprocessing: {preprocessing}\nbatch_size: {batch_size}\ngen_size: {gen_size}\ndimensions: {dimensions}\nepochs: {epochs}\ndiscriminator_steps: {discriminator_steps}"

                            with open(os.path.join(directory, 'info.txt'), "w") as f:
                                f.write(iteration_information)

                            cli_gen(config)

                            log(f"Finished Discriminator Steps \"{discriminator_steps}\".", level=LogLevel.OK)
                        log(f"Finished Epochs \"{epochs}\".", level=LogLevel.OK)
                    log(f"Finished Dimensions \"{dimensions}\".", level=LogLevel.INFO)
                log(f"Finished Gen Size \"{gen_size}\".", level=LogLevel.INFO)
            log(f"Finished Batch Size \"{batch_size}\".", level=LogLevel.INFO)
        log(f"Finished Preprocessing \"{preprocessing}\".", level=LogLevel.INFO)
    log(f"All experiments have been completed!", level=LogLevel.INFO)


def experiments_onto_cgans(loader_config: str or Config):
    dimensions_list = loader_config.get_nested('settings', 'dimensions')
    batch_size_list = loader_config.get_nested('settings', 'batch_size')
    discriminator_steps_list = loader_config.get_nested('settings', 'discriminator_steps')
    epochs_list = loader_config.get_nested('settings', 'epochs')
    gen_size_list = loader_config.get_nested('settings', 'gen_size')
    preprocessing_list = loader_config.get_nested('settings', 'preprocessing')
    num_walks_list = loader_config.get_nested('settings', 'embedding', 'walks', 'num_walks')
    walk_length_list = loader_config.get_nested('settings', 'embedding', 'walks', 'walk_length')
    alpha_list = loader_config.get_nested('settings', 'embedding', 'walks', 'alpha')
    w2c_epochs_list = loader_config.get_nested('settings', 'embedding', 'word2vec', 'epochs')
    w2v_dimensions_list = loader_config.get_nested('settings', 'embedding', 'word2vec', 'dimensions')
    w2v_min_count_list = loader_config.get_nested('settings', 'embedding', 'word2vec', 'min_count')

    for preprocessing in preprocessing_list:
        log(f"Starting Preprocessing \"{preprocessing}\".", level=LogLevel.INFO)

        preprocessing_directory = get_or_create_directory(
            f"{loader_config.get_nested('preprocessing', 'directory')}" % preprocessing, error=True)

        for batch_size in batch_size_list:
            log(f"Starting Batch Size \"{batch_size}\".", level=LogLevel.INFO)
            for gen_size in gen_size_list:
                log(f"Starting Gen Size \"{gen_size}\".", level=LogLevel.INFO)
                for dimensions in dimensions_list:
                    log(f"Starting Dimensions \"{dimensions}\".", level=LogLevel.INFO)
                    for epochs in epochs_list:
                        log(f"Starting Epochs \"{epochs}\".", level=LogLevel.INFO)
                        for discriminator_steps in discriminator_steps_list:
                            log(f"Starting Discriminator Steps \"{discriminator_steps}\".", level=LogLevel.INFO)
                            for num_walks in num_walks_list:
                                log(f"Starting Num Walks \"{num_walks}\".", level=LogLevel.INFO)

                                for walk_length in walk_length_list:
                                    log(f"Starting Walk Length \"{walk_length}\".", level=LogLevel.INFO)

                                    for alpha in alpha_list:
                                        log(f"Starting Alpha \"{alpha}\".", level=LogLevel.INFO)

                                        for w2v_epochs in w2c_epochs_list:
                                            log(f"Starting W2V epochs \"{w2v_epochs}\".", level=LogLevel.INFO)

                                            for w2v_dimensions in w2v_dimensions_list:
                                                log(f"Starting W2V Dimensions \"{w2v_dimensions}\".",
                                                    level=LogLevel.INFO)

                                                for w2v_min_count in w2v_min_count_list:
                                                    log(f"Starting W2V Min Count \"{w2v_min_count}\".",
                                                        level=LogLevel.INFO)

                                                    config = load_config(loader_config.get_nested('config'))

                                                    # (preprocessing)_(batch_size)_(gen_size)_(dimensions)_(epochs)_(discriminator_steps)
                                                    current_directory_name = f'{preprocessing}_{batch_size}_{gen_size}_{dimensions}_{epochs}_{discriminator_steps}_{num_walks}_{walk_length}_{alpha}_{w2v_epochs}_{w2v_dimensions}_{w2v_min_count}'
                                                    directory = get_or_create_directory(
                                                        config.get_nested('dp_cgans', 'files',
                                                                          'directory') % current_directory_name)
                                                    config.config['dp_cgans']['files']['directory'] = directory

                                                    input_data_path = create_path(preprocessing_directory,
                                                                                  config.get_nested('dp_cgans', 'files',
                                                                                                    'input'),
                                                                                  create=False)
                                                    config.config['dp_cgans']['files']['input'] = input_data_path

                                                    output_data_path = create_path(directory,
                                                                                   config.get_nested('dp_cgans',
                                                                                                     'files', 'output'),
                                                                                   create=True)
                                                    config.config['dp_cgans']['files']['output'] = output_data_path

                                                    embedding_directory = get_or_create_directory(directory,
                                                                                                  config.get_nested(
                                                                                                      'embedding',
                                                                                                      'files',
                                                                                                      'directory'))
                                                    config.config['embedding']['files'][
                                                        'directory'] = embedding_directory

                                                    new_references = []

                                                    for reference in config.get_nested('embedding', 'references'):
                                                        new_references.append(reference % preprocessing)

                                                    config.config['embedding']['references'] = new_references
                                                    config.config['dp_cgans']['files']['use_root'] = False
                                                    config.config['dp_cgans']['batch_size'] = batch_size
                                                    config.config['dp_cgans']['gen_size'] = gen_size
                                                    config.config['dp_cgans']['dimensions'] = dimensions
                                                    config.config['dp_cgans']['epochs'] = epochs
                                                    config.config['dp_cgans'][
                                                        'discriminator_steps'] = discriminator_steps

                                                    # Onto Specific
                                                    config.config['embedding']['random_walks']['num_walks'] = num_walks
                                                    config.config['embedding']['random_walks'][
                                                        'walk_length'] = walk_length
                                                    config.config['embedding']['random_walks']['alpha'] = alpha
                                                    config.config['embedding']['word2vec']['epochs'] = w2v_epochs
                                                    config.config['embedding']['word2vec'][
                                                        'dimensions'] = w2v_dimensions
                                                    config.config['embedding']['word2vec']['min_count'] = w2v_min_count

                                                    iteration_information = f"preprocessing: {preprocessing}\nbatch_size: {batch_size}\ngen_size: {gen_size}\ndimensions: {dimensions}\nepochs: {epochs}\ndiscriminator_steps: {discriminator_steps}\nnum_walks: {num_walks}\nwalk_length: {walk_length}\nalpha: {alpha}\nw2v_epochs: {w2v_epochs}\nw2v_dimensions: {w2v_dimensions}\nw2v_min_count: {w2v_min_count}"

                                                    with open(os.path.join(directory, 'info.txt'), "w") as f:
                                                        f.write(iteration_information)

                                                    cli_onto(config)

                                                    log(f"Finished W2V Min count \"{w2v_min_count}\".",
                                                        level=LogLevel.OK)
                                                log(f"Finished W2V Dimensions \"{w2v_dimensions}\".", level=LogLevel.OK)
                                            log(f"Finished W2V Epochs \"{w2v_epochs}\".", level=LogLevel.OK)
                                        log(f"Finished Alpha \"{alpha}\".", level=LogLevel.OK)
                                    log(f"Finished Walk Length \"{walk_length}\".", level=LogLevel.OK)
                                log(f"Finished Num Walks \"{num_walks}\".", level=LogLevel.OK)
                            log(f"Finished Discriminator Steps \"{discriminator_steps}\".", level=LogLevel.OK)
                        log(f"Finished Epochs \"{epochs}\".", level=LogLevel.OK)
                    log(f"Finished Dimensions \"{dimensions}\".", level=LogLevel.INFO)
                log(f"Finished Gen Size \"{gen_size}\".", level=LogLevel.INFO)
            log(f"Finished Batch Size \"{batch_size}\".", level=LogLevel.INFO)
        log(f"Finished Preprocessing \"{preprocessing}\".", level=LogLevel.INFO)
    log(f"All experiments have been completed!", level=LogLevel.INFO)


if __name__ == "__main__":
    cli()
