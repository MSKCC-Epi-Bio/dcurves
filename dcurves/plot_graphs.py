import pandas as pd
import matplotlib.pyplot as plt
from beartype import beartype

@beartype
def _plot_net_benefit(
        plot_df: pd.DataFrame,
        y_limits: list = [-0.05, 0.2],
        color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:

    modelnames = plot_df['model'].value_counts().index

    if len(color_names) < len(modelnames):
        ValueError('More predictors than color_names, please enter more color names in color_names list and try again')

    if 'model' not in plot_df.columns:
        ValueError('Column name containing model names is not a column in inputted dataframe, \
                   please make sure model_name_colname exists in dataframe')

    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = plot_df[plot_df['model'] == modelname]
        x_vals = single_model_df['threshold']
        y_vals = single_model_df['net_benefit']
        plt.plot(x_vals, y_vals, color=color_name)

        plt.ylim(y_limits)
        plt.legend(modelnames)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel('Threshold Values')
        plt.ylabel('Calculated Net Benefit')
    plt.show()

@beartype
def _plot_net_intervention_avoided(
                plot_df: pd.DataFrame,
                y_limits: list = [-0.05, 0.2],
                color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:

    # Don't want to plot 'all'/'none' for net_intervention_avoided

    cleaned_plot_df = plot_df[~(plot_df["model"].isin(['all', 'none']))]

    modelnames = cleaned_plot_df['model'].value_counts().index

    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = cleaned_plot_df[cleaned_plot_df['model'] == modelname]
        x_vals = single_model_df['threshold']
        y_vals = single_model_df['net_intervention_avoided']
        plt.plot(x_vals, y_vals, color=color_name)

        plt.ylim(y_limits)
        plt.legend(modelnames)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel('Threshold Values')
        plt.ylabel('Calculated Net Reduction of Interventions Per 100 Patients')
    plt.show()

def plot_graphs(plot_df: pd.DataFrame,
                graph_type: str = 'net_benefit',
                y_limits: list = [-0.05, 0.5],
                color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:

    if graph_type not in ['net_benefit', 'net_intervention_avoided']:
        ValueError('graph_type must be one of 2 strings: net_benefit, net_intervention_avoided')

    if graph_type == 'net_benefit':

        _plot_net_benefit(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names
        )

    elif graph_type == 'net_intervention_avoided':

        _plot_net_intervention_avoided(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names
        )


    return

_plot_net_benefit.__doc__ = """

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


_plot_net_intervention_avoided.__doc__ = """

    |

    Plot the number of net interventions avoided in using the specified marker.

    |

    Specifically, this function
    will plot the calculated net interventions avoided for each threshold value.


    """


