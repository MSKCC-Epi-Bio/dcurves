"""
This module houses plotting functions used in the user-facing plot_graphs() 
function to plot net-benefit scores and net interventions avoided.
"""
from typing import Optional, Iterable
import random
import matplotlib.pyplot as plt
import pandas as pd


def _get_colors(num_colors=None):
    """
    Generate a random tuple of colors of length num_colors

    Parameters
    ----------
    num_colors : int
        Number of colors to be outputted in tuple form

    Returns
    -------
    tuple
    """
    return [f"#{format(random.randint(0, 0xFFFFFF), '06x')}" for _ in range(num_colors)]


def _plot_net_benefit(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Iterable = None,
    show_grid: bool = True,
) -> None:
    """
    Plot net benefit values against threshold probability values.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values, model columns of net benefit scores
        to be plotted
    y_limits : list[float]
        2 floats, lower and upper bounds for y-axis
    color_names : list[str]
        Colors to render each model (if n models supplied, then need n+2 colors,
        since 'all' and 'none' models will be included by default
    show_grid : bool, optional
        Flag to enable or disable gridlines in the plot. Default is True, displaying gridlines. Set to False to hide gridlines.

    Returns
    -------
    None
    """

    modelnames = plot_df["model"].value_counts().index
    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = plot_df[plot_df["model"] == modelname]
        plt.plot(
            single_model_df["threshold"],
            single_model_df["net_benefit"],
            color=color_name,
        )
        plt.ylim(y_limits)
        plt.legend(modelnames)
        if show_grid:
            plt.grid(color="black", which="both", axis="both", linewidth="0.3")
        else:
            plt.grid(False)
        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
    plt.show()


def _plot_net_intervention_avoided(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Iterable = None,
    show_grid: bool = True,
) -> None:
    """
    Plot net interventions avoided values against threshold probability values.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values, model columns of net intervention
        scores to be plotted
    y_limits : list[float]
        2 floats, lower and upper bounds for y-axis
    color_names
        Colors to render each model (if n models supplied, then need n+2 colors,
        since 'all' and 'none' models will be included by default
    show_grid : bool, optional
        Flag to enable or disable gridlines in the plot. Default is True, displaying gridlines. Set to False to hide gridlines.

    Returns
    -------
    None
    """

    modelnames = plot_df["model"].value_counts().index
    for modelname, color_name in zip(modelnames, color_names):
        single_model_df = plot_df[plot_df["model"] == modelname]
        plt.plot(
            single_model_df["threshold"],
            single_model_df["net_intervention_avoided"],
            color=color_name,
        )

        plt.ylim(y_limits)
        plt.legend(modelnames)
        if show_grid:
            plt.grid(color="black", which="both", axis="both", linewidth="0.3")
        else:
            plt.grid(False)
        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Reduction of Interventions")
    plt.show()


def plot_graphs(
    plot_df: pd.DataFrame,
    graph_type: str = "net_benefit",
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable] = None,
    show_grid: bool = True,
) -> None:
    """
    Plot either net benefit or interventions avoided per threshold.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values, model columns of net benefit/intervention
        scores to be plotted
    graph_type : str
        Type of plot (either 'net_benefit' or 'net_intervention_avoided')
    y_limits : Iterable[Lower Bound, Upper Bound]
        2 floats, lower and upper bounds for y-axis
    color_names : Iterable[str]
        Colors to render each model (if n models supplied, then need n+2 colors,
        since 'all' and 'none' models will be included by default
    show_grid : bool, optional
        Flag to enable or disable gridlines in the plot. Default is True, displaying gridlines. Set to False to hide gridlines.

    Returns
    -------
    None
    """

    if graph_type not in ["net_benefit", "net_intervention_avoided"]:
        raise ValueError(
            "graph_type must be one of 2 strings: net_benefit,"
            " net_intervention_avoided"
        )

    modelnames = plot_df["model"].value_counts().index
    if color_names is None:
        color_names = _get_colors(num_colors=len(modelnames))
    elif color_names is not None:
        if len(color_names) < len(modelnames):
            raise ValueError(
                "More predictors than color_names, please enter more color names"
                " in color_names list and try again"
            )

    if graph_type == "net_benefit":
        _plot_net_benefit(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names,
            show_grid=show_grid,
        )
    elif graph_type == "net_intervention_avoided":
        _plot_net_intervention_avoided(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names,
            show_grid=show_grid,
        )
