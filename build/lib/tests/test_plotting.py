# Load Data
from dcurves.load_test_data import load_binary_df, load_survival_df

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

def test_2_case1_plot_net_benefit():

    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
    # y_limits = [-0.05, 0.2]
    color_names = None

    dca_famhistory_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory']
        )

    # plot_graphs(
    #     plot_df=dca_famhistory_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.2]
    # )


# def test_case1_plot_net_benefit():
#
#     data = load_binary_df()
#     y_limits = [-0.05, 0.2]
#     dca_results = \
#         dca(
#             data=data,
#             outcome='cancer',
#             modelnames=['famhistory']
#         )
#
#     plot_df = dca_results
#
#     modelnames = plot_df['model'].value_counts().index
#
#     get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
#     colors = get_colors(len(modelnames))
#     # get_colors(3)  # sample return:  ['#8af5da', '#fbc08c', '#b741d0']
#
#     for modelname, color_name in zip(modelnames, colors):
#         single_model_df = plot_df[plot_df['model'] == modelname]
#         plt.plot(single_model_df['threshold'], single_model_df['net_benefit'], color=color_name)
#         plt.ylim(y_limits)
#         plt.legend(modelnames)
#         plt.grid(b=True, which='both', axis='both')
#         plt.xlabel('Threshold Values')
#         plt.ylabel('Calculated Net Benefit')
#     plt.show()
