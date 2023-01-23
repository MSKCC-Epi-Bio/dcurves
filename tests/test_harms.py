from dcurves import dca, plot_graphs
from .load_test_data import load_r_case3_results
import pandas as pd
import numpy as np

def test_simple_binary_harms_1():

    r_case3_benchmark_results = load_r_case3_results()
    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')

    dca_harm_simple_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            thresholds=np.arange(0, 0.36, 0.01),
            harm={'marker': 0.0333},
            models_to_prob=['marker']
        )

    # plot_graphs(
    #     plot_df=dca_harm_simple_df,
    #     graph_type='net_benefit',
    #     color_names=['red', 'blue', 'green'],
    #     y_limits=[-0.05, 0.20]
    # )

    for model in ['all', 'none', 'marker']:
        for stat in ['net_benefit', 'net_intervention_avoided', 'tp_rate', 'fp_rate']:
            assert dca_harm_simple_df[dca_harm_simple_df.model == model][
                stat].round(
                decimals=6).reset_index(
                drop=True).equals(
                r_case3_benchmark_results[
                    r_case3_benchmark_results.variable == model][
                    stat].round(decimals=6).reset_index(drop=True))
