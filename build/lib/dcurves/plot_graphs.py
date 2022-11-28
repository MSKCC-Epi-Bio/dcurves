import pandas as pd
import matplotlib.pyplot as plt
from dcurves import _validate
from beartype import beartype

@beartype
def plot_net_benefit(
        after_dca_df: pd.DataFrame,
        y_limits: list = [-0.05, 0.2],
        color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:

    _validate._plot_graphs_input_checks(after_dca_df=after_dca_df,
                                        y_limits=y_limits,
                                        color_names=color_names)

    predictor_names = after_dca_df['predictor'].value_counts().index
    # color_names = ['blue', 'purple','red',
    #                'green', 'hotpink', 'orange',
    #                'saddlebrown', 'lime', 'magenta']

    for predictor_name, color_name in zip(predictor_names, color_names):
        single_pred_df = after_dca_df[after_dca_df['predictor'] == predictor_name]
        x_vals = single_pred_df['threshold']
        y_vals = single_pred_df['net_benefit']
        plt.plot(x_vals, y_vals, color=color_name)

        plt.ylim(y_limits)
        plt.legend(predictor_names)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel('Threshold Values')
        plt.ylabel('Calculated Net Benefit')

@beartype
def plot_net_intervention_avoided(after_dca_df: pd.DataFrame,
                y_limits: list = [-0.05, 0.2],
                color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:
    _validate._plot_graphs_input_checks(after_dca_df=after_dca_df,
                                        y_limits=y_limits,
                                        color_names=color_names)

    # Don't want to plot 'all'/'none' for net_intervention_avoided

    cleaned_after_dca_df = after_dca_df[~(after_dca_df["predictor"].isin(['all', 'none']))]

    predictor_names = cleaned_after_dca_df['predictor'].value_counts().index

    for predictor_name, color_name in zip(predictor_names, color_names):
        single_pred_df = cleaned_after_dca_df[cleaned_after_dca_df['predictor'] == predictor_name]
        x_vals = single_pred_df['threshold']
        y_vals = single_pred_df['net_intervention_avoided']
        plt.plot(x_vals, y_vals, color=color_name)

        plt.ylim(y_limits)
        plt.legend(predictor_names)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel('Threshold Values')
        plt.ylabel('Calculated Net Interventions Avoided')


# def plot_graphs(after_dca_df: pd.DataFrame,
#                 graph_type: str = 'net_benefit',
#                 y_limits: list = [-0.05, 0.2],
#                 color_names: list = ['blue', 'purple', 'red',
#                                      'green', 'hotpink', 'orange',
#                                      'saddlebrown', 'lime', 'magenta']
#                 ) -> None:
#
#
#     _validate._plot_graphs_input_checks(after_dca_df=after_dca_df,
#                                         y_limits=y_limits,
#                                         color_names=color_names,
#                                         graph_type=graph_type)
#
#     _validate._plot_net_intervention_input_checks(after_dca_df=after_dca_df,
#                                                   graph_type=graph_type)
#
#     if graph_type == 'net_benefit':
#
#         predictor_names = after_dca_df['predictor'].value_counts().index
#         # color_names = ['blue', 'purple','red',
#         #                'green', 'hotpink', 'orange',
#         #                'saddlebrown', 'lime', 'magenta']
#
#         for predictor_name, color_name in zip(predictor_names, color_names):
#             single_pred_df = after_dca_df[after_dca_df['predictor'] == predictor_name]
#             x_vals = single_pred_df['threshold']
#             y_vals = single_pred_df['net_benefit']
#             plt.plot(x_vals, y_vals, color=color_name)
#
#             plt.ylim(y_limits)
#             plt.legend(predictor_names)
#             plt.grid(b=True, which='both', axis='both')
#             plt.xlabel('Threshold Values')
#             plt.ylabel('Calculated Net Benefit')
#
#     elif graph_type == 'net_intervention_avoided':
#
#         # Don't want to plot 'all'/'none' for net_intervention_avoided
#
#         cleaned_after_dca_df = after_dca_df[~(after_dca_df["predictor"].isin(['all', 'none']))]
#
#         predictor_names = cleaned_after_dca_df['predictor'].value_counts().index
#
#         for predictor_name, color_name in zip(predictor_names, color_names):
#             single_pred_df = cleaned_after_dca_df[cleaned_after_dca_df['predictor'] == predictor_name]
#             x_vals = single_pred_df['threshold']
#             y_vals = single_pred_df['net_intervention_avoided']
#             plt.plot(x_vals, y_vals, color=color_name)
#
#             plt.ylim(y_limits)
#             plt.legend(predictor_names)
#             plt.grid(b=True, which='both', axis='both')
#             plt.xlabel('Threshold Values')
#             plt.ylabel('Calculated Net Interventions Avoided')
#
#     return

plot_net_benefit.__doc__ = """

    |

    Plot net benefit values from binaru_dca/surv_dca outputted DataFrames.

    |

    Specifically, this function
    will plot the calculated net benefit values for each threshold value from those
    indicated in the dca() function.

    Examples
    ________

    |

    Simple plot binary DCA example. Load binary outcome dataframe, run DCA, plot calculated predictor net benefit values.

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
    >>> binary_dca_results = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )
    >>> plot_graphs(after_dca_df = binary_dca_results)

    |

    Simple binary DCA example with plotting net interventions avoided. Load binary outcome dataframe, run DCA, run
    net_intervention_avoided(), plot outputted dataframe

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
    >>> after_dca_df = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )
    >>> after_net_interventions_avoided_df = net_intervention_avoided(after_dca_df=after_dca_df)
    >>> plot_graphs(after_dca_df = after_net_interventions_avoided_df,
    ...     graph_type='net_intervention_avoided',
    ...     y_limits=[-10,100],
    ...     color_names=['red','teal']
    ... )

    Parameters
    __________
    after_dca_df : pandas.DataFrame
        dataframe outputted by dca function in the dcurves library
    graph_type : str
        type of graph outputted, either 'net_benefit' or 'net_intervention_avoided' (defaults to 'net_benefit')
    y_limits : list[float]
        list of float that control graph lower and upper y-axis limits
        (defaults to [-0.05, 0.2] for graph_type == net_benefit. Change values for
         graph_type == net_intervention_avoided)
    color_names : list[str]
        list of colors specified by user (defaults to ['blue', 'purple', 'red',
        'green', 'hotpink', 'orange', 'saddlebrown', 'lime', 'magenta']

    Returns
    _______
    None

    """


plot_net_intervention_avoided.__doc__ = """

    |

    Plot the number of net interventions avoided in using the specified marker.

    |

    Specifically, this function
    will plot the calculated net interventions avoided for each threshold value.

    
    """


