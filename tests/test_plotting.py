# Load Data
from .load_test_data import load_binary_df, load_survival_df

# Load DCA function(s)
from dcurves.dca import dca
from dcurves.plot_graphs import _plot_net_benefit, _plot_net_intervention_avoided, plot_graphs

# Load Libraries Used in Testing
import matplotlib as mpl
mpl.use('tkagg') # Sets backend for mpl to TkAgggraphical: GUI library for Python based on Tk GUI toolkit
# Above solves issues with plt.show()
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

def test_2_case1_plot_net_benefit():

    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
    # y_limits = [-0.05, 0.2]
    color_names = None

    dca_famhistory_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=np.arange(0, 0.45, 0.01)
        )

    # plot_graphs(
    #     plot_df=dca_famhistory_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.15],
    #     color_names=['blue', 'red', 'green']
    # )


def test_case1_plot_net_benefit():

    data = load_binary_df()
    y_limits = [-0.05, 0.2]
    color_names = None
    dca_results = \
        dca(
            data=data,
            outcome='cancer',
            modelnames=['famhistory']
        )

    plot_df = dca_results

    # _plot_net_benefit(
    #     plot_df=plot_df
    # )

def test_case1_plot_net_intervention_avoided():

    data = load_binary_df()
    y_limits = [-0.05, 1]
    color_names = None
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