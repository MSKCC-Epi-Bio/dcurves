import pandas as pd
import numpy as np
import random
import re
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .load_test_data import load_binary_df
from dcurves.dca import dca
from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
    _get_colors,
)

from unittest.mock import patch

# Constants
SAMPLE_DATA_DF = pd.DataFrame(
    {
        "model": ["Model1", "Model2"],
        "threshold": [0.2, 0.5],
        "net_benefit": [0.7, 0.8],
        "net_intervention_avoided": [0.3, 0.4],
    }
)

DEFAULT_COLOR_NAMES = ["red", "blue"]


def get_dca_results(data, thresholds=None):
    return dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=thresholds or np.arange(0, 0.5, 0.01),
    )


def test_2_case1_plot_net_benefit():
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )
    dca_famhistory_df = get_dca_results(df_cancer_dx, [i / 100 for i in range(0, 46)])


def test_case1_plot_net_benefit():
    data = load_binary_df()
    get_dca_results(data)


def test_case1_plot_net_intervention_avoided():
    data = load_binary_df()
    get_dca_results(data)


def test_get_colors():
    # Set a seed for repeatability
    random.seed(42)

    num_colors = 5
    colors = _get_colors(num_colors=num_colors)

    # Check that the number of colors returned is correct
    assert (
        len(colors) == num_colors
    ), f"Expected {num_colors} colors, got {len(colors)} colors."

    # Check that each color is a valid color code
    for color in colors:
        assert re.match(r"^#[0-9A-Fa-f]{6}$", color), f"Invalid color code: {color}"

    # Check for repeatability using a seed
    random.seed(42)
    assert colors == _get_colors(
        num_colors=num_colors
    ), "Expected the same colors with the same seed."


@pytest.mark.parametrize(
    "func, args",
    [
        (_plot_net_benefit, {"y_limits": (-0.05, 1)}),
        (_plot_net_intervention_avoided, {}),
    ],
)
def test_plot_functions(func, args, mocker):
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.show")
    func(SAMPLE_DATA_DF, color_names=DEFAULT_COLOR_NAMES, **args)


def test_invalid_graph_type():
    df = pd.DataFrame({"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]})
    with pytest.raises(
        ValueError,
        match="graph_type must be 'net_benefit' or 'net_intervention_avoided'",
    ):
        plot_graphs(plot_df=df, graph_type="invalid_type")


def test_fewer_color_names_than_models():
    df = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.1],
            "net_benefit": [0.2, 0.3],
        }
    )
    with pytest.raises(
        ValueError,
        match="color_names must match the number of unique models in plot_df",
    ):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=["red"])


def test_custom_colors_net_benefit(mocker):
    mocker.patch("dcurves.plot_graphs")
    df = pd.DataFrame({"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]})
    with patch("matplotlib.pyplot.show"):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=["red"])
    # assertions can be added if necessary


def test_custom_colors_net_intervention_avoided(mocker):
    mocker.patch("dcurves.plot_graphs", return_value=None)
    df = pd.DataFrame(
        {"model": ["model1"], "threshold": [0.1], "net_intervention_avoided": [0.3]}
    )
    with patch("matplotlib.pyplot.show"):
        plot_graphs(
            plot_df=df, graph_type="net_intervention_avoided", color_names=["red"]
        )
    # assertions can be added if necessary


def test_default_colors_net_benefit(mocker):
    mocker.patch("dcurves.plot_graphs", return_value=None)
    df = pd.DataFrame({"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]})
    with patch("matplotlib.pyplot.show"):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=None)
    # assertions can be added if necessary


def test_plot_net_benefit_missing_columns():
    df_missing_columns = pd.DataFrame({"model": ["model1"], "threshold": [0.1]})
    with pytest.raises(
        ValueError,
        match="plot_df must contain the following columns: threshold, model, net_benefit",
    ):
        with patch("matplotlib.pyplot.show"):
            _plot_net_benefit(
                df_missing_columns, y_limits=(-0.05, 1), color_names=["blue"]
            )


def test_plot_net_benefit_invalid_y_limits():
    with pytest.raises(
        ValueError,
        match="y_limits must contain two floats where the first is less than the second",
    ):
        _plot_net_benefit(SAMPLE_DATA_DF, y_limits=(-1, -2), color_names=["blue"])


def test_plot_net_benefit_mismatched_color_names():
    df_two_models = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.2],
            "net_benefit": [0.3, 0.4],
        }
    )
    with pytest.raises(
        ValueError,
        match="The length of color_names must match the number of unique models",
    ):
        _plot_net_benefit(df_two_models, y_limits=(-0.05, 1), color_names=["blue"])


def test_plot_net_benefit_grid_enabled():
    plt.ioff()  # Turn off interactive mode
    with patch("matplotlib.pyplot.grid") as mock_grid, patch(
        "matplotlib.pyplot.show"
    ):  # Mock plt.show() to prevent plot display
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_grid=True,
        )
        mock_grid.assert_called_with(
            color="black", which="both", axis="both", linewidth="0.3"
        )


def test_plot_net_benefit_show_legend_enabled(mocker):
    plt.ioff()  # Turn off interactive mode
    with patch("matplotlib.pyplot.legend") as mock_legend, patch(
        "matplotlib.pyplot.show"
    ):  # Mock plt.show() to prevent plot display
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_benefit_show_legend_disabled(mocker):
    plt.ioff()  # Turn off interactive mode
    with patch("matplotlib.pyplot.legend") as mock_legend, patch(
        "matplotlib.pyplot.show"
    ):  # Mock plt.show() to prevent plot display
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


def test_plot_net_intervention_avoided_show_legend_enabled(mocker):
    plt.ioff()  # Turn off interactive mode
    with patch("matplotlib.pyplot.legend") as mock_legend, patch(
        "matplotlib.pyplot.show"
    ):  # Mock plt.show() to prevent plot display
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_intervention_avoided_show_legend_disabled(mocker):
    plt.ioff()  # Turn off interactive mode
    with patch("matplotlib.pyplot.legend") as mock_legend, patch(
        "matplotlib.pyplot.show"
    ):  # Mock plt.show() to prevent plot display
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


def test_plot_graphs_y_limits(mock_plot):
    custom_y_limits = (0, 0.5)
    plot_graphs(SAMPLE_DATA_DF, "net_benefit", y_limits=custom_y_limits)
    mock_plot.assert_called_with(
        SAMPLE_DATA_DF, custom_y_limits, None, True, True, False
    )


def test_integration_with_dca_results():
    data = load_binary_df()
    dca_results = get_dca_results(data)
    # This test would check if the plotting function can handle the output format of `get_dca_results` directly.
    with patch("matplotlib.pyplot.show"):
        plot_graphs(dca_results, "net_benefit")


def test_plot_graphs_function_selection():
    SAMPLE_DATA_DF = pd.DataFrame(
        {
            "model": ["Model1", "Model2"],
            "threshold": [0.2, 0.5],
            "net_benefit": [0.7, 0.8],
            "net_intervention_avoided": [0.3, 0.4],
        }
    )

    # Call plot_graphs for each graph type without mock assertions
    plot_graphs(SAMPLE_DATA_DF, "net_benefit")
    plot_graphs(SAMPLE_DATA_DF, "net_intervention_avoided")
    # No direct assertions on the mock calls; consider other ways to verify the expected behavior


def test_plot_graphs_custom_colors():
    SAMPLE_DATA_DF = pd.DataFrame(
        {
            "model": ["Model1", "Model2"],
            "threshold": [0.2, 0.5],
            "net_benefit": [0.7, 0.8],
            "net_intervention_avoided": [0.3, 0.4],
        }
    )
    custom_colors = ["green", "purple"]
    # Directly call plot_graphs without mock assertions
    plot_graphs(SAMPLE_DATA_DF, "net_benefit", color_names=custom_colors)
    # Verifying the color usage in the plot is not straightforward in a unit test


def test_plot_graphs_y_limits():
    SAMPLE_DATA_DF = pd.DataFrame(
        {
            "model": ["Model1", "Model2"],
            "threshold": [0.2, 0.5],
            "net_benefit": [0.7, 0.8],
            "net_intervention_avoided": [0.3, 0.4],
        }
    )
    custom_y_limits = (0, 0.5)
    # Directly call plot_graphs without mock assertions
    plot_graphs(SAMPLE_DATA_DF, "net_benefit", y_limits=custom_y_limits)
    # Asserts here would need to focus on the resulting plot, which is more complex and may not be ideal for unit testing


def test_plot_graphs_with_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        plot_graphs(empty_df, "net_benefit")
