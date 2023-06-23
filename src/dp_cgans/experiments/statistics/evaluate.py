import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

from scipy.stats import pointbiserialr

from dp_cgans.utils import Config
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import get_or_create_directory, create_path

warnings.filterwarnings('ignore')

from sdv.evaluation import evaluate


# check this page for more evaluation: https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary


def _stat_eva_for_all(syn_list, exp_data):  # syn_list is a list of synthetic data, exp_data is real/original data
    evaluation_results = {}
    for each_syn_data in syn_list:
        evaluation_results[each_syn_data] = evaluate(syn_list[each_syn_data], exp_data,
                                                     metrics=['DiscreteKLDivergence', 'CSTest'], aggregate=False)
    return evaluation_results


def statistical_evaluation(config: str or Config):
    _config = load_config(config)

    label = _config.get_nested('class_header')
    real_file_path = _config.get_nested('real_path')
    synthethic_file_paths = _config.get_nested('synthetic_paths')

    real_df = pd.read_csv(real_file_path).drop(label, axis=1).astype(bool)
    synthethic_dfs = {

    }

    for name, path in synthethic_file_paths.items():
        synthetic_df = pd.read_csv(path)

        synthethic_dfs[name] = synthetic_df.drop(label, axis=1).astype(bool)

    stat_evaluation_results = _stat_eva_for_all(synthethic_dfs, real_df)

    output_directory = get_or_create_directory(_config.get_nested('output_directory'))

    for key, results in stat_evaluation_results.items():
        print(key)
        results.to_csv(create_path(output_directory, f"{key}.csv"), index=False)

    correlation_scores = {

    }

    for name, fake_df in synthethic_dfs.items():
        if len(real_df) <= len(fake_df):
            shorter_df = real_df
            longer_df = fake_df
        else:
            shorter_df = fake_df
            longer_df = real_df

        sampled_longer_df = longer_df.sample(n=len(shorter_df), random_state=42)

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

    pd.DataFrame.from_dict(correlation_scores).to_csv(create_path(output_directory, f"correlations.csv"), index=False)
