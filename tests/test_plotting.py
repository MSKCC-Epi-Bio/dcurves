# Load Data
from .load_test_data import load_binary_df

# Load DCA function(s)
from dcurves.dca import dca
from dcurves.plot_graphs import _plot_net_benefit, _plot_net_intervention_avoided, plot_graphs

# Load Libraries Used in Testing
import pandas as pd
import numpy as np

def test_2_case1_plot_net_benefit():

    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')

    dca_famhistory_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=[i/100 for i in range(0, 46)]
        )

    # plot_graphs(
    #     plot_df=dca_famhistory_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.15],
    #     color_names=['blue', 'red', 'green']
    # )


def test_case1_plot_net_benefit():

    data = load_binary_df()
    dca_results = \
        dca(
            data=data,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=np.arange(0, 0.5, 0.01)
        )

    plot_df = dca_results

    # _plot_net_benefit(
    #     plot_df=plot_df,
    #     color_names=['blue', 'red', 'green'],
    #     y_limits=[-0.05, 0.3]
    # )

def test_case1_plot_net_intervention_avoided():

    data = load_binary_df()
    y_limits = [-0.05, 1]

    dca_results = \
        dca(
            data=data,
            outcome='cancer',
            modelnames=['famhistory']
        )

    plot_df = dca_results

    # _plot_net_intervention_avoided(
    #     plot_df=plot_df,
    #     y_limits=y_limits,
    #     color_names=['blue', 'red', 'green']
    # )

def test_plot_rough_dca():

    pass