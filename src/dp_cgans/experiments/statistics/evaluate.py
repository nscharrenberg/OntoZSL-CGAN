import warnings

# As provided by Chang Sun
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sdv.evaluation import evaluate

from dp_cgans.utils.config import Config
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import get_or_create_directory, create_path

warnings.filterwarnings("ignore")


def _stat_eva_for_all(syn_list, exp_data) -> dict:
    """
    Perform statistical evaluation of synthetic data compared to experimental data.

    This function calculates evaluation metrics for each synthetic dataset in `syn_list`
    and returns a dictionary containing the evaluation results.

    Args:
        syn_list (dict): A dictionary where the keys represent the names of the synthetic datasets,
            and the values are the corresponding synthetic data as pandas DataFrames.
        exp_data (pandas DataFrame): The experimental data for comparison.

    Returns:
        dict: A dictionary where the keys are the names of the synthetic datasets, and the values
            are the evaluation results as pandas DataFrames. The evaluation results include metrics
            such as 'DiscreteKLDivergence' and 'CSTest'.
    """
    evaluation_results = {}

    for synthetic_data in syn_list:
        evaluation_results[synthetic_data] = evaluate(syn_list[synthetic_data], exp_data, metrics=['DiscreteKLDivergence', 'CSTest'], aggregate=False)

    return evaluation_results


def statistical_evaluation(config: str or Config) -> None:
    """
    Perform statistical evaluation of synthetic data against experimental data using various evaluation metrics.

    This function loads the configuration, reads the data from files, performs statistical evaluation,
    and saves the results to output files.

    Args:
        config (str or Config): Either a string representing the path to the configuration file or
            a `Config` object containing the configuration settings.

    Returns:
        None
    """
    _config = load_config(config)

    label = _config.get_nested('class_header')
    real_file_path = _config.get_nested('real_path')
    synthetic_file_paths = _config.get_nested('synthetic_paths')

    real_df = pd.read_csv(real_file_path).drop(label, axis=1).astype(bool)
    synthetic_dfs = {}

    for name, path in synthetic_file_paths.items():
        synthetic_df = pd.read_csv(path)

        synthetic_dfs[name] = synthetic_df.drop(label, axis=1).astype(bool)

    stat_evaluation_results = _stat_eva_for_all(synthetic_dfs, real_df)
    output_directory = get_or_create_directory(_config.get_nested('output_directory'))

    for key, results in stat_evaluation_results.items():
        print(key)
        results.to_csv(create_path(output_directory, f"{key}.csv"), index=False)

    correlation_scores = {}

    for name, fake_df in synthetic_dfs.items():
        if len(real_df) <= len(fake_df):
            shorter_df = real_df
            longer_df = fake_df
        else:
            shorter_df = fake_df
            longer_df = real_df

        sampled_longer_df = longer_df.sample(n=len(shorter_df))
        point_biserial_coef_matrix = pd.DataFrame(index=shorter_df.columns, columns=sampled_longer_df.columns)

        phi_scores = []
        for shorter_col in shorter_df.columns:
            for longer_col in sampled_longer_df.columns:
                point_biserial_coef, _ = pointbiserialr(shorter_df[shorter_col], sampled_longer_df[longer_col])
                point_biserial_coef_matrix.loc[shorter_col, longer_col] = point_biserial_coef
                phi_scores.append(point_biserial_coef)

        point_biserial_coef_matrix.to_csv(create_path(output_directory, f"correlation-{name}.csv"), index=True)
        correlation_scores[name] = {
            "score": np.mean(phi_scores)
        }

        pd.DataFrame.from_dict(correlation_scores).to_csv(create_path(output_directory, f"correlations.csv"),
                                                          index=False)



