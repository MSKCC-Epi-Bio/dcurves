"""
This module contains tests for the harm incorporation in the DCA function.
"""

import pandas as pd

from dcurves import dca
from .load_test_data import load_r_case3_results


def test_simple_binary_harms_1():
    """
    Test simple binary harms calculation and compare with R benchmark results.
    """
    r_case3_benchmark_results = load_r_case3_results()
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca_harm_simple_df = dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["marker"],
        thresholds=[i / 100 for i in range(36)],
        harm={"marker": 0.0333},
        models_to_prob=["marker"],
    )

    for model in ["all", "none", "marker"]:
        for stat in ["net_benefit", "net_intervention_avoided", "tp_rate", "fp_rate"]:
            p_model_stat_df = dca_harm_simple_df.loc[dca_harm_simple_df.model == model, stat]
            p_model_stat_df = p_model_stat_df.round(decimals=6)
            p_model_stat_df = p_model_stat_df.reset_index(drop=True)

            r_model_stat_df = r_case3_benchmark_results.loc[
                r_case3_benchmark_results.variable == model, stat
            ]
            r_model_stat_df = r_model_stat_df.round(decimals=6)
            r_model_stat_df = r_model_stat_df.reset_index(drop=True)

            assert p_model_stat_df.equals(r_model_stat_df)
