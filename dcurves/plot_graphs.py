"""
This module houses plotting functions used in the user-facing plot_graphs() 
function to plot net-benefit scores and net interventions avoided.
"""
from typing import Optional, Iterable
import random
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from numpy import ndarray


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
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None,  # Corrected parameter
) -> None:
    """
    Plot net benefit values against threshold probability values. Can use pre-computed smoothed data if provided.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and model columns of net benefit scores to be plotted.
    y_limits : Iterable[float], optional
        Tuple or list with two floats specifying the y-axis limits.
    color_names : Iterable[str], optional
        List of colors for each model line. Must match the number of unique models if provided.
    show_grid : bool, optional
        If True, display grid lines on the plot. Default is True.
    show_legend : bool, optional
        If True, display the legend on the plot. Default is True.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model. Keys are model names, and values are arrays with smoothed points.

    Raises
    ------
    ValueError
        If the input dataframe does not contain the required columns or if y_limits or color_names are incorrectly formatted.

    Returns
    -------
    None
    """

    # Validate input dataframe
    required_columns = ["threshold", "model", "net_benefit"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    # Validate y_limits
    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError("y_limits must contain two floats where the first is less than the second")

    # Validate color_names
    modelnames = plot_df["model"].unique()
    if color_names and len(color_names) != len(modelnames):
        raise ValueError("The length of color_names must match the number of unique models")

    # Plotting
    for idx, modelname in enumerate(plot_df["model"].unique()):
        color = color_names[idx]  # Directly use color from color_names by index
        model_df = plot_df[plot_df["model"] == modelname]
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if not isinstance(smoothed, ndarray):
                raise ValueError(f"Smoothed data for '{modelname}' must be a NumPy array.")
            plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, label=modelname)
        else:
            plt.plot(model_df["threshold"], model_df["net_benefit"], color=color, label=modelname)

    plt.ylim(y_limits)
    if show_legend:
        plt.legend()
    if show_grid:
        plt.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        plt.grid(False)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")


def _plot_net_intervention_avoided(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Iterable = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None  # Updated to accept smoothed data
) -> None:
    """
    Plot net interventions avoided values against threshold probability values. Can use pre-computed smoothed data if provided.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and model columns of net interventions avoided scores to be plotted.
    y_limits : Iterable[float]
        Tuple or list with two floats specifying the y-axis limits.
    color_names : Iterable[str]
        List of colors for each model line. Must match the number of unique models if provided.
    show_grid : bool
        If True, display grid lines on the plot. Default is True.
    show_legend : bool
        If True, display the legend on the plot. Default is True.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model. Keys are model names, and values are arrays with smoothed points.

    Raises
    ------
    ValueError
        If the input dataframe does not contain the required columns or if y_limits or color_names are incorrectly formatted.

    Returns
    -------
    None
    """

    # Validate input dataframe
    required_columns = ["threshold", "model", "net_intervention_avoided"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    # Validate y_limits
    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError("y_limits must contain two floats where the first is less than the second")

    # Validate color_names
    modelnames = plot_df["model"].unique()
    if color_names and len(color_names) != len(modelnames):
        raise ValueError("The length of color_names must match the number of unique models")

    # Plotting
    for idx, modelname in enumerate(plot_df["model"].unique()):
        color = color_names[idx]  # Directly use color from color_names by index
        model_df = plot_df[plot_df["model"] == modelname]
        if model_df.empty:  # Skip plotting for empty DataFrames
            continue
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if smoothed_data and modelname in smoothed_data:
                if not isinstance(smoothed, ndarray):
                    raise ValueError(f"Smoothed data for '{modelname}' must be a NumPy array.")
            plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, label=modelname)
        else:
            plt.plot(model_df["threshold"], model_df["net_intervention_avoided"], color=color, label=modelname)

    plt.ylim(y_limits)
    if show_legend:
        plt.legend()
    if show_grid:
        plt.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        plt.grid(False)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Reduction of Interventions")


def plot_graphs(
    plot_df: pd.DataFrame,
    graph_type: str = "net_benefit",
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smooth_frac: float = 0.0,  # Default to 0, indicating no smoothing unless specified
    file_name: Optional[str] = None,
    dpi: int = 100
) -> None:
    """
    Plot specified graph type for the given data, either net benefit or net interventions avoided,
    against threshold probabilities. Applies LOWESS smoothing if `smooth_frac` is greater than 0,
    excluding 'all' and 'none' models from smoothing. The smoothing will be more sensitive to local variations,
    keeping the smoothed lines closer to the original data points if `smooth_frac` is specified.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame containing 'threshold', 'model', and either 'net_benefit' or 'net_intervention_avoided' columns.
    graph_type : str, optional
        Specifies the type of plot to generate. Valid options are 'net_benefit' or 'net_intervention_avoided'.
    y_limits : Iterable[float], optional
        Two-element iterable specifying the lower and upper bounds of the y-axis.
    color_names : Iterable[str], optional
        List of colors to use for each line in the plot. Must match the number of models in `plot_df`.
    show_grid : bool, optional
        If True, display grid lines on the plot. Default is True.
    show_legend : bool, optional
        If True, display the legend on the plot. Default is True.
    smooth_frac : float, optional
        Fraction of data points used when estimating each y-value in the smoothed line,
        making the smoothing more sensitive to local variations. Set to 0 for no smoothing. Default is 0.
    file_name : str, optional
        Path and file name where the figure will be saved. If None, the figure is not saved.
    dpi : int, optional
        Resolution of the saved figure in dots per inch.

    Raises
    ------
    ValueError
        If `graph_type` is not recognized.
        If `y_limits` does not contain exactly two elements or if the lower limit is not less than the upper limit.
        If `color_names` is provided but does not match the number of models in `plot_df`.
        If `smooth_frac` is not within the 0 to 1 range.
        If the input DataFrame is empty.

    Returns
    -------
    None
    """

    if plot_df.empty:
        raise ValueError("The input DataFrame is empty.")

    if graph_type not in ["net_benefit", "net_intervention_avoided"]:
        raise ValueError("graph_type must be 'net_benefit' or 'net_intervention_avoided'")

    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError("y_limits must contain two floats where the first is less than the second")

    if not 0 <= smooth_frac <= 1:
        raise ValueError("smooth_frac must be between 0 and 1")

    modelnames = plot_df["model"].unique()
    if color_names is None:
        color_names = _get_colors(num_colors=len(modelnames))
    elif len(color_names) < len(modelnames):
        raise ValueError("color_names must match the number of unique models in plot_df")

    smoothed_data = {}
    if smooth_frac > 0:  # Apply smoothing only if smooth_frac is greater than 0
        lowess = sm.nonparametric.lowess
        for modelname in plot_df["model"].unique():
            # Skip 'all' and 'none' models from smoothing
            if modelname.lower() in ['all', 'none']:
                continue

            model_df = plot_df[plot_df["model"] == modelname]
            y_col = "net_benefit" if graph_type == "net_benefit" else "net_intervention_avoided"
            smoothed_data[modelname] = lowess(model_df[y_col], model_df["threshold"], frac=smooth_frac)

    plot_function = _plot_net_benefit if graph_type == "net_benefit" else _plot_net_intervention_avoided
    plot_function(
        plot_df=plot_df,
        y_limits=y_limits,
        color_names=color_names,
        show_grid=show_grid,
        show_legend=show_legend,
        smoothed_data=smoothed_data if smooth_frac > 0 else None,  # Pass smoothed_data only if smoothing was applied
    )

    if file_name:
        try:
            plt.savefig(file_name, dpi=dpi)
        except Exception as e:
            print(f"Error saving figure: {e}")
    plt.show()
