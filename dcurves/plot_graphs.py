import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from beartype import beartype
import random
from typing import Optional

@beartype
def _plot_net_benefit(
        plot_df: pd.DataFrame,
        y_limits: list = [-0.05, 0.2],
        color_names: Optional[list] = None,
                ) -> None:
    """
    Plot net benefit values against threshold values.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold values, model columns of net benefit scores to be plotted
    y_limits : list[float]
        2 floats, lower and upper bounds for y-axis
    color_names
        Colors to render each model (if n models supplied, then need n+2 colors, since 'all' and 'none' models will be
        included by default
    Returns
    -------
    None
    """

    modelnames = plot_df['model'].value_counts().index
    if color_names is None:
        get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
        color_names = get_colors(len(modelnames))
        # get_colors(3)  # sample return:  ['#8af5da', '#fbc08c', '#b741d0']
    elif color_names is not None:
        if len(color_names) < len(modelnames):
            ValueError(
                'More predictors than color_names, please enter more color names in color_names list and try again')
    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = plot_df[plot_df['model'] == modelname]
        plt.plot(single_model_df['threshold'], single_model_df['net_benefit'], color=color_name)
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
                color_names: Optional[list] = None
                ) -> None:
    """
    Plot net interventions avoided values against threshold values.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold values, model columns of net intervention scores to be plotted
    y_limits : list[float]
        2 floats, lower and upper bounds for y-axis
    color_names
        Colors to render each model (if n models supplied, then need n+2 colors, since 'all' and 'none' models will be
        included by default
    Returns
    -------
    None
    """

    # Don't want to plot 'all'/'none' for net_intervention_avoided

    # plot_df = plot_df[~(plot_df["model"].isin(['all', 'none']))]

    modelnames = plot_df['model'].value_counts().index
    if color_names is None:
        get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
        color_names = get_colors(len(modelnames))
    elif color_names is not None:
        if len(color_names) < len(modelnames):
            ValueError(
                'More predictors than color_names, please enter more color names in color_names list and try again')
    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = plot_df[plot_df['model'] == modelname]
        plt.plot(single_model_df['threshold'], single_model_df['net_intervention_avoided'], color=color_name)

        plt.ylim(y_limits)
        plt.legend(modelnames)
        plt.grid(b=True, which='both', axis='both')
        plt.xlabel('Threshold Values')
        plt.ylabel('Calculated Net Reduction of Interventions')
    plt.show()

@beartype
def plot_graphs(plot_df: pd.DataFrame,
                graph_type: str = 'net_benefit',
                y_limits: list = [-0.05, 1],
                color_names: Optional[list] = None
                ) -> None:
    """
    Plot either net benefit or interventions avoided per threshold.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold values, model columns of net benefit/intervention scores to be plotted
    graph_type : str (default: 'net_benefit')
        Type of plot (either 'net_benefit' or 'net_intervention_avoided')
    y_limits : list[float]
        2 floats, lower and upper bounds for y-axis
    color_names : list[str]
        Colors to render each model (if n models supplied, then need n+2 colors, since 'all' and 'none' models will be
        included by default

    Returns
    -------
    None
    """

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
