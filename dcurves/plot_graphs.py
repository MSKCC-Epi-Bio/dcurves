"""
This module houses plotting functions used in the user-facing plot_graphs()
function to plot net-benefit scores and net interventions avoided.
"""

from typing import Optional, Iterable, List
import random
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from numpy import ndarray

VALID_MARKERS = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    "|",
    "_",
    "P",
    "X",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]


def _get_colors(modelnames):
    """
    Generate a tuple of colors based on the provided model names.
    'none' is always blue, 'all' is always red, and up to 4 additional colors are predefined.
    Any remaining colors are randomly generated.

    Parameters
    ----------
    modelnames : Iterable
        Iterable of model names for which colors are needed

    Returns
    -------
    tuple
        Tuple of color strings in the same order as the input modelnames

    Example
    -------
    >>> _get_colors(["none", "all", "model1", "model2"])
    ('#0000FF', '#FF0000', '#00FF00', '#800080')
    """
    color_dict = {
        "none": "#0000FF",  # Blue
        "all": "#FF0000",  # Red
    }
    predefined_colors = [
        "#00FF00",
        "#800080",
        "#FFA500",
        "#00FFFF",
    ]  # Green, Purple, Orange, Cyan

    colors = []
    predefined_index = 0

    for model in modelnames:
        if model.lower() in color_dict:
            colors.append(color_dict[model.lower()])
        elif predefined_index < len(predefined_colors):
            colors.append(predefined_colors[predefined_index])
            predefined_index += 1
        else:
            colors.append(f"#{format(random.randint(0, 0xFFFFFF), '06x')}")

    return tuple(colors)


def _validate_markers(markers: List) -> None:
    """
    Validate that all provided markers are valid matplotlib marker styles.

    Parameters
    ----------
    markers : List
        List of marker symbols to validate

    Raises
    ------
    ValueError
        If any marker is not a valid matplotlib marker style
    """
    if markers is None:
        return

    invalid_markers = [marker for marker in markers if marker not in VALID_MARKERS]
    if invalid_markers:
        valid_markers_str = ", ".join(
            [f"'{m}'" if isinstance(m, str) else str(m) for m in VALID_MARKERS]
        )
        raise ValueError(
            f"Invalid marker style(s): {invalid_markers}. "
            f"Please use one of the following valid markers: {valid_markers_str}"
        )


def _plot_net_benefit(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable[str]] = None,
    markers: Optional[Iterable[str]] = None,
    linestyles: Optional[Iterable[str]] = None,
    linewidths: Optional[Iterable[float]] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None,
) -> None:
    """
    Plot net benefit values against threshold probability values.
    Supports custom markers and linestyles.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and net benefit scores.
    y_limits : Iterable[float], optional
        Tuple with two floats specifying the y-axis limits.
    color_names : Iterable[str], optional
        List of colors for each model line.
    markers : Iterable[str], optional
        List of marker symbols for each model.
        Examples: ['o', 's', '^', 'D'] where 'o' is circle, 's' is square,
        '^' is triangle, and 'D' is diamond.
    linestyles : Iterable[str], optional
        List of line styles for each model.
        Examples: ['-', '--', '-.', ':'] where '-' is solid, '--' is dashed,
        '-.' is dash-dot, and ':' is dotted.
    linewidths : Iterable[float], optional
        List specifying the line width for each model. Provide a single value
        to apply the same width to all models, or a list matching the number
        of models.
    show_grid : bool, optional
        If True, display grid lines on the plot.
    show_legend : bool, optional
        If True, display the legend on the plot.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model.

    Raises
    ------
    ValueError
        For invalid input DataFrame, y_limits, or if the length of color_names does
        not match the number of unique models.

    Returns
    -------
    None
    """
    required_columns = ["threshold", "model", "net_benefit"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    modelnames = plot_df["model"].unique()
    if color_names and len(color_names) not in (1, len(modelnames)):
        raise ValueError(
            "color_names must be either a single value or match the number of unique models in plot_df"
        )

    # Validate linewidths length
    if linewidths and len(linewidths) not in (1, len(modelnames)):
        raise ValueError(
            "linewidths must be either a single value or match the number of unique models"
        )

    for idx, modelname in enumerate(modelnames):
        if color_names:
            color = color_names[idx % len(color_names)]
        else:
            color = _get_colors(modelnames)[idx]
        model_df = plot_df[plot_df["model"] == modelname]
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if not isinstance(smoothed, ndarray):
                raise ValueError(
                    f"Smoothed data for '{modelname}' must be a NumPy array."
                )
            x = smoothed[:, 0]
            y = smoothed[:, 1]
        else:
            x = model_df["threshold"]
            y = model_df["net_benefit"]
        plot_kwargs = {"color": color, "label": modelname}
        if markers is not None:
            plot_kwargs["marker"] = markers[idx % len(markers)]
        if linestyles is not None:
            plot_kwargs["linestyle"] = linestyles[idx % len(linestyles)]
        if linewidths is not None:
            plot_kwargs["linewidth"] = linewidths[idx % len(linewidths)]
        plt.plot(x, y, **plot_kwargs)

    plt.ylim(y_limits)
    if show_legend:
        plt.legend()
    if show_grid:
        # Use linewidth as a string to match expected test parameters.
        plt.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        plt.grid(False)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")


def _plot_net_intervention_avoided(
    plot_df: pd.DataFrame,
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable[str]] = None,
    markers: Optional[Iterable[str]] = None,
    linestyles: Optional[Iterable[str]] = None,
    linewidths: Optional[Iterable[float]] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smoothed_data: Optional[dict] = None,
) -> None:
    """
    Plot net interventions avoided values against threshold probability values.
    Supports custom markers and linestyles.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Data containing threshold probability values and net interventions avoided scores.
    y_limits : Iterable[float]
        Tuple with two floats specifying the y-axis limits.
    color_names : Iterable[str]
        List of colors for each model line.
    markers : Iterable[str], optional
        List of marker symbols for each model.
        Examples: ['o', 's', '^', 'D'].
    linestyles : Iterable[str], optional
        List of line styles for each model.
        Examples: ['-', '--', '-.', ':'].
    linewidths : Iterable[float], optional
        List specifying the line width for each model. Provide a single value
        to apply the same width to all models, or a list matching the number
        of models.
    show_grid : bool
        If True, display grid lines on the plot.
    show_legend : bool
        If True, display the legend on the plot.
    smoothed_data : dict, optional
        Pre-computed smoothed data for each model.

    Raises
    ------
    ValueError
        For invalid input DataFrame, y_limits, or if the length of color_names does
        not match the number of unique models.

    Returns
    -------
    None
    """
    required_columns = ["threshold", "model", "net_intervention_avoided"]
    if not all(column in plot_df.columns for column in required_columns):
        raise ValueError(
            f"plot_df must contain the following columns: {', '.join(required_columns)}"
        )

    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    modelnames = plot_df["model"].unique()
    if color_names and len(color_names) not in (1, len(modelnames)):
        raise ValueError(
            "color_names must be either a single value or match the number of unique models in plot_df"
        )

    # Validate linewidths length
    if linewidths and len(linewidths) not in (1, len(modelnames)):
        raise ValueError(
            "linewidths must be either a single value or match the number of unique models"
        )

    for idx, modelname in enumerate(modelnames):
        if color_names:
            color = color_names[idx % len(color_names)]
        else:
            color = _get_colors(modelnames)[idx]
        model_df = plot_df[plot_df["model"] == modelname]
        if model_df.empty:
            continue
        if smoothed_data and modelname in smoothed_data:
            smoothed = smoothed_data[modelname]
            if not isinstance(smoothed, ndarray):
                raise ValueError(
                    f"Smoothed data for '{modelname}' must be a NumPy array."
                )
            x = smoothed[:, 0]
            y = smoothed[:, 1]
        else:
            x = model_df["threshold"]
            y = model_df["net_intervention_avoided"]
        plot_kwargs = {"color": color, "label": modelname}
        if markers is not None:
            plot_kwargs["marker"] = markers[idx % len(markers)]
        if linestyles is not None:
            plot_kwargs["linestyle"] = linestyles[idx % len(linestyles)]
        if linewidths is not None:
            plot_kwargs["linewidth"] = linewidths[idx % len(linewidths)]
        plt.plot(x, y, **plot_kwargs)

    plt.ylim(y_limits)
    if show_legend:
        plt.legend()
    if show_grid:
        # Use linewidth as a string to match expected test parameters.
        plt.grid(color="black", which="both", axis="both", linewidth="0.3")
    else:
        plt.grid(False)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Reduction of Interventions")


def plot_graphs(
    plot_df: pd.DataFrame,
    graph_type: str = "net_benefit",
    y_limits: Iterable = (-0.05, 1),
    color_names: Optional[Iterable[str]] = None,
    markers: Optional[Iterable[str]] = None,
    linestyles: Optional[Iterable[str]] = None,
    linewidths: Optional[Iterable[float]] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    smooth_frac: float = 0.0,
    file_name: Optional[str] = None,
    dpi: int = 100,
) -> None:
    """
    Plot specified graph type for the given data, either net benefit or net interventions avoided,
    against threshold probabilities. Applies LOWESS smoothing if smooth_frac > 0
    (excluding 'all' and 'none' models), and allows custom markers and linestyles.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame containing 'threshold', 'model', and either 'net_benefit' or
        'net_intervention_avoided' columns.
        Specifies the type of plot to generate. Options: 'net_benefit' or 'net_intervention_avoided'.
    y_limits : Iterable[float], optional
        Two-element iterable specifying the lower and upper bounds of the y-axis.
    color_names : Iterable[str], optional
        List of colors to use for each line in the plot.
    linewidths : Iterable[float], optional
        List specifying the line width for each model. Provide a single value
        to apply the same width to all models, or a list matching the number
        of models.
    markers : Iterable[str], optional
        List of marker symbols for each model.
        Examples: ['o', 's', '^', 'v', 'D', 'x', '+', '*'].
    linestyles : Iterable[str], optional
        List of line styles for each model.
        Examples: ['-', '--', ':', '-.'].
    show_grid : bool, optional
        If True, display grid lines on the plot.
    show_legend : bool, optional
        If True, display the legend on the plot.
    smooth_frac : float, optional
        Fraction of data used for LOWESS smoothing. Set to 0 for no smoothing.
    file_name : str, optional
        Path and file name to save the figure. If None, the figure is not saved.
    dpi : int, optional
        Resolution of the saved figure.

    Raises
    ------
    ValueError
        For invalid graph_type, y_limits, smooth_frac, empty input DataFrame, or invalid marker styles.

    Returns
    -------
    None
    """
    if plot_df.empty:
        raise ValueError("The input DataFrame is empty.")

    if graph_type not in ["net_benefit", "net_intervention_avoided"]:
        raise ValueError(
            "graph_type must be 'net_benefit' or 'net_intervention_avoided'"
        )

    if len(y_limits) != 2 or y_limits[0] >= y_limits[1]:
        raise ValueError(
            "y_limits must contain two floats where the first is less than the second"
        )

    if not 0 <= smooth_frac <= 1:
        raise ValueError("smooth_frac must be between 0 and 1")

    # Validate marker styles
    _validate_markers(markers)

    modelnames = plot_df["model"].unique()
    if color_names is None:
        color_names = _get_colors(modelnames)
    elif len(color_names) not in (1, len(modelnames)):
        raise ValueError(
            "color_names must be either a single value or match the number of unique models in plot_df"
        )

    # Validate linewidths length
    if linewidths and len(linewidths) not in (1, len(modelnames)):
        raise ValueError(
            "linewidths must be either a single value or match the number of unique models"
        )

    smoothed_data = {}
    if smooth_frac > 0:
        lowess = sm.nonparametric.lowess
        for modelname in modelnames:
            if modelname.lower() in ["all", "none"]:
                continue
            model_df = plot_df[plot_df["model"] == modelname]
            y_col = (
                "net_benefit"
                if graph_type == "net_benefit"
                else "net_intervention_avoided"
            )
            smoothed_data[modelname] = lowess(
                model_df[y_col], model_df["threshold"], frac=smooth_frac
            )

    if graph_type == "net_benefit":
        _plot_net_benefit(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names,
            markers=markers,
            linestyles=linestyles,
            linewidths=linewidths,
            show_grid=show_grid,
            show_legend=show_legend,
            smoothed_data=smoothed_data if smooth_frac > 0 else None,
        )
    else:
        _plot_net_intervention_avoided(
            plot_df=plot_df,
            y_limits=y_limits,
            color_names=color_names,
            markers=markers,
            linestyles=linestyles,
            linewidths=linewidths,
            show_grid=show_grid,
            show_legend=show_legend,
            smoothed_data=smoothed_data if smooth_frac > 0 else None,
        )

    if file_name:
        try:
            plt.savefig(file_name, dpi=dpi)
        except Exception as e:
            print(f"Error saving figure: {e}")
    plt.show()
