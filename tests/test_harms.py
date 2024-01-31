from dcurves import dca, plot_graphs
from .load_test_data import load_r_case3_results
import pandas as pd


def test_simple_binary_harms_1():
    r_case3_benchmark_results = load_r_case3_results()
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca_harm_simple_df = dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["marker"],
        thresholds=[i / 100 for i in range(0, 36)],
        harm={"marker": 0.0333},
        models_to_prob=["marker"],
    )

    # plot_graphs(
    #     plot_df=dca_harm_simple_df,
    #     graph_type='net_benefit',
    #     color_names=['red', 'blue', 'green'],
    #     y_limits=[-0.05, 0.20]
    # )

    for model in ["all", "none", "marker"]:
        for stat in ["net_benefit", "net_intervention_avoided", "tp_rate", "fp_rate"]:
            p_model_stat_df = dca_harm_simple_df.loc[
                dca_harm_simple_df.model == model, stat
            ]
            p_model_stat_df = p_model_stat_df.round(decimals=6)
            p_model_stat_df = p_model_stat_df.reset_index(drop=True)

            r_model_stat_df = r_case3_benchmark_results.loc[
                r_case3_benchmark_results.variable == model, stat
            ]
            r_model_stat_df = r_model_stat_df.round(decimals=6)
            r_model_stat_df = r_model_stat_df.reset_index(drop=True)

            assert p_model_stat_df.equals(r_model_stat_df)
