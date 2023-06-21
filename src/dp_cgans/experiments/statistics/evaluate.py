import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from dp_cgans.utils import Config
from dp_cgans.utils.files import get_or_create_directory, create_path

warnings.filterwarnings('ignore')

from sdv.evaluation import evaluate


# check this page for more evaluation: https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary


def _stat_eva_for_all(syn_list, exp_data):  # syn_list is a list of synthetic data, exp_data is real/original data
    evaluation_results = {}
    for each_syn_data in syn_list:
        evaluation_results[each_syn_data] = evaluate(syn_list[each_syn_data], exp_data,
                                                     metrics=['DiscreteKLDivergence', 'ContinuousKLDivergence',
                                                              'CSTest', 'KSTest'], aggregate=False)
    return evaluation_results


def statistical_evaluation_noah(config: str or Config):
    if isinstance(config, str):
        _config = Config(config)
    elif isinstance(config, Config):
        _config = config
    else:
        raise Exception("Configuration could not be read.")

    real_file_path = _config.get_nested('real_path')
    synthethic_file_paths = _config.get_nested('synthetic_paths')
    label = _config.get_nested('class_header')

    real_df = pd.read_csv(real_file_path)
    real_labels = real_df[label]
    real_binaries = real_df.drop(label, axis=1)

    synthethic_dfs = {

    }

    results = {

    }

    for name, path in synthethic_file_paths.items():
        synthetic_df = pd.read_csv(path)
        synthethic_dfs[name] = synthetic_df

    real_contingency_table = pd.crosstab(real_labels, real_binaries) + 0.1
    real_chi2, real_chi2_p_value, _, _ = chi2_contingency(real_contingency_table)

    results['real']['chi2'] = {
        'chi2': real_chi2,
        'p': real_chi2_p_value
    }

    for name, data in synthethic_dfs.items():
        fake_labels = data[label]
        fake_binaries = data.drop(label, axis=1)

        fake_contingency_table = pd.crosstab(fake_labels, fake_binaries) + 0.1
        fake_chi2, fake_chi2_p_value, _, _ = chi2_contingency(fake_contingency_table)

        t_statistic, t_p_value = ttest_ind(real_binaries, fake_binaries)
        f_statistic, f_p_value = f_oneway(real_binaries, fake_binaries)

        results[name]['chi2'] = {
            'chi2': fake_chi2,
            'p': fake_chi2_p_value
        }

        results[name]['t'] = {
            't': t_statistic,
            'p': t_p_value
        }

        results[name]['anova'] = {
            'f': f_statistic,
            'p': f_p_value
        }

    output_directory = get_or_create_directory(_config.get_nested('output_directory'))
    pd.DataFrame.from_dict(results).to_csv(create_path(output_directory, "results.csv"), index=False)


def statistical_evaluation(config: str or Config):
    if isinstance(config, str):
        _config = Config(config)
    elif isinstance(config, Config):
        _config = config
    else:
        raise Exception("Configuration could not be read.")

    label = _config.get_nested('class_header')
    real_file_path = _config.get_nested('real_path')
    synthethic_file_paths = _config.get_nested('synthetic_paths')

    real_df = pd.read_csv(real_file_path)
    synthethic_dfs = {

    }

    for name, path in synthethic_file_paths.items():
        synthetic_df = pd.read_csv(path)

        synthethic_dfs[name] = synthetic_df.drop(label, axis=1).astype(bool)

    stat_evaluation_results = _stat_eva_for_all(synthethic_dfs, real_df.drop(label, axis=1).astype(bool))

    output_directory = get_or_create_directory(_config.get_nested('output_directory'))

    for key, results in stat_evaluation_results.items():
        print(key)
        results.to_csv(create_path(output_directory, f"{key}.csv"), index=False)


