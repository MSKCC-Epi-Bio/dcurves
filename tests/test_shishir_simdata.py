# Load Data
from .load_test_data import load_shishir_simdata

# Load _calc_prevalence, other necessary functions
from dcurves import dca, plot_graphs

# Load Tools
import pandas as pd


def test_shishir_simdata():

    data = load_shishir_simdata()

    # print(
    #     '\n',
    #     data.model.value_counts()
    # )



    # for i in data.model.value_counts().index:
    #     print(i)

    # print(data.to_string())

    # print(data.to_string())

    # dca_result = \
    #     dca(
    #         data=data,
    #         outcome='e',
    #         modelnames=['p'],
    #         time_to_outcome_col='t',
    #         time=121
    #     )

    # plot_graphs(
    #     plot_df=dca_result,
    #     graph_type='net_benefit'
    # )

    # print(data.to_string())