import numpy as np
import pandas as pd
import seaborn as sns
import sdv
import matplotlib.pyplot as plt

import warnings

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


def statistical_evaluation(config: str or Config):
    if isinstance(config, str):
        _config = Config(config)
    elif isinstance(config, Config):
        _config = config
    else:
        raise Exception("Configuration could not be read.")

    real_file_path = _config.get_nested('real_path')
    synthethic_file_paths = _config.get_nested('synthetic_paths')

    real_df = pd.read_csv(real_file_path)
    synthethic_dfs = {

    }

    for name, path in synthethic_file_paths.items():
        synthetic_df = pd.read_csv(path)

        synthethic_dfs[name] = synthetic_df

    stat_evaluation_results = _stat_eva_for_all(synthethic_dfs, real_df)

    output_directory = get_or_create_directory(_config.get_nested('output_directory'))

    for key, results in stat_evaluation_results.items():
        print(key)
        results.to_csv(create_path(output_directory, f"{key}.csv"), index=False)


