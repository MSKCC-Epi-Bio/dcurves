"""
This module contains tests for plotting functionality in dcurves.
"""

import random
import re
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

# pylint: disable=C0413
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pylint: enable=C0413

from dcurves.dca import dca
from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
    _get_colors,
    _validate_markers,
)
from .load_test_data import load_binary_df

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
    """Get DCA results for given data and thresholds."""
    return dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=thresholds or np.arange(0, 0.5, 0.01),
    )


def test_2_case1_plot_net_benefit():
    """Test case 1 for plotting net benefit."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )
    get_dca_results(df_cancer_dx, [i / 100 for i in range(0, 46)])


def test_case1_plot_net_benefit():
    """Test case 1 for plotting net benefit."""
    data = load_binary_df()
    get_dca_results(data)


def test_case1_plot_net_intervention_avoided():
    """Test case 1 for plotting net intervention avoided."""
    data = load_binary_df()
    get_dca_results(data)


def test_get_colors():
    """Test color generation function."""
    random.seed(42)
    modelnames = ["Model1", "Model2", "Model3", "Model4", "Model5"]
    colors = _get_colors(modelnames)

    assert len(colors) == len(modelnames)

    for color in colors:
        assert re.match(r"^#[0-9A-Fa-f]{6}$", color), f"Invalid color code: {color}"

    random.seed(42)
    assert colors == _get_colors(modelnames), (
        "Expected the same colors with the same seed."
    )

    # Test that 'none' and 'all' get their specific colors
    special_modelnames = ["none", "all", "Model1", "Model2"]
    special_colors = _get_colors(special_modelnames)
    assert special_colors[0] == "#0000FF", "'none' should be blue"
    assert special_colors[1] == "#FF0000", "'all' should be red"


@pytest.mark.parametrize(
    "func, args",
    [
        (_plot_net_benefit, {"y_limits": (-0.05, 1)}),
        (_plot_net_intervention_avoided, {}),
    ],
)
def test_plot_functions(func, args):
    """Test plot functions."""
    with patch("matplotlib.pyplot.plot"), patch("matplotlib.pyplot.show"):
        func(SAMPLE_DATA_DF, color_names=DEFAULT_COLOR_NAMES, **args)


def test_invalid_graph_type():
    """Test invalid graph type."""
    sample_df = pd.DataFrame(
        {"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]}
    )
    with pytest.raises(
        ValueError,
        match="graph_type must be 'net_benefit' or 'net_intervention_avoided'",
    ):
        plot_graphs(plot_df=sample_df, graph_type="invalid_type")


def test_fewer_color_names_than_models():
    """Test fewer color names than models."""
    sample_df = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.1],
            "net_benefit": [0.2, 0.3],
        }
    )
    # A single color for multiple models is now acceptable; provide an invalid case
    # where the length is neither 1 nor the number of models to ensure a ValueError.
    with pytest.raises(
        ValueError,
        match="color_names must be either a single value or match the number of unique models",
    ):
        plot_graphs(
            plot_df=sample_df,
            graph_type="net_benefit",
            color_names=["red", "blue", "green"],
        )


def test_custom_colors_net_benefit():
    """Test custom colors for net benefit."""
    sample_df = pd.DataFrame(
        {"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]}
    )
    with patch("matplotlib.pyplot.show"), patch("dcurves.plot_graphs"):
        plot_graphs(plot_df=sample_df, graph_type="net_benefit", color_names=["red"])


def test_custom_colors_net_intervention_avoided():
    """Test custom colors for net intervention avoided."""
    sample_df = pd.DataFrame(
        {"model": ["model1"], "threshold": [0.1], "net_intervention_avoided": [0.3]}
    )
    with (
        patch("matplotlib.pyplot.show"),
        patch("dcurves.plot_graphs", return_value=None),
    ):
        plot_graphs(
            plot_df=sample_df,
            graph_type="net_intervention_avoided",
            color_names=["red"],
        )


def test_default_colors_net_benefit():
    """Test default colors for net benefit."""
    sample_df = pd.DataFrame(
        {"model": ["model1"], "threshold": [0.1], "net_benefit": [0.2]}
    )
    with (
        patch("matplotlib.pyplot.show"),
        patch("dcurves.plot_graphs", return_value=None),
    ):
        plot_graphs(plot_df=sample_df, graph_type="net_benefit", color_names=None)


def test_plot_net_benefit_missing_columns():
    """Test plot net benefit with missing columns."""
    df_missing_columns = pd.DataFrame({"model": ["model1"], "threshold": [0.1]})
    with pytest.raises(
        ValueError,
        match="plot_df must contain the following columns: "
        "threshold, model, net_benefit",
    ):
        with patch("matplotlib.pyplot.show"):
            _plot_net_benefit(
                df_missing_columns, y_limits=(-0.05, 1), color_names=["blue"]
            )


def test_plot_net_benefit_invalid_y_limits():
    """Test plot net benefit with invalid y limits."""
    with pytest.raises(
        ValueError,
        match="y_limits must contain two floats where the first "
        "is less than the second",
    ):
        _plot_net_benefit(SAMPLE_DATA_DF, y_limits=(-1, -2), color_names=["blue"])


def test_plot_net_benefit_mismatched_color_names():
    """Test plot net benefit with mismatched color names."""
    df_two_models = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.2],
            "net_benefit": [0.3, 0.4],
        }
    )
    # Expect error when color_names length is invalid (neither 1 nor number of models)
    with pytest.raises(
        ValueError,
        match="color_names must be either a single value or match the number of unique models",
    ):
        _plot_net_benefit(
            df_two_models, y_limits=(-0.05, 1), color_names=["blue", "red", "green"]
        )


def test_plot_net_benefit_grid_enabled():
    """Test plot net benefit with grid enabled."""
    plt.ioff()  # Turn off interactive mode
    with (
        patch("matplotlib.pyplot.grid") as mock_grid,
        patch("matplotlib.pyplot.show"),
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


def test_plot_net_benefit_show_legend_enabled():
    """Test plot net benefit with legend enabled."""
    plt.ioff()  # Turn off interactive mode
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):  # Mock plt.show() to prevent plot display
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_benefit_show_legend_disabled():
    """Test plot net benefit with legend disabled."""
    plt.ioff()  # Turn off interactive mode
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):  # Mock plt.show() to prevent plot display
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


def test_plot_net_intervention_avoided_show_legend_enabled():
    """Test plot net intervention avoided with legend enabled."""
    plt.ioff()  # Turn off interactive mode
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):  # Mock plt.show() to prevent plot display
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_intervention_avoided_show_legend_disabled():
    """Test plot net intervention avoided with legend disabled."""
    plt.ioff()  # Turn off interactive mode
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):  # Mock plt.show() to prevent plot display
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


def test_plot_graphs_y_limits():
    """Test plot graphs with custom y limits."""
    custom_y_limits = (0, 0.5)

    with patch("matplotlib.pyplot.show"):  # Prevent the plot from actually showing
        plot_graphs(SAMPLE_DATA_DF, "net_benefit", y_limits=custom_y_limits)

    # Get the current axes
    current_axes = plt.gca()

    # Check y-limits
    assert current_axes.get_ylim() == custom_y_limits

    # Check that at least the number of unique models are represented in the plot
    unique_models = SAMPLE_DATA_DF["model"].unique()
    assert len(current_axes.lines) >= len(unique_models)

    # Check that each unique model has at least one line in the plot
    line_labels = [line.get_label() for line in current_axes.lines]
    for model in unique_models:
        assert model in line_labels, f"No line found for model {model}"

    # Check x-label and y-label
    assert current_axes.get_xlabel() == "Threshold Probability"
    assert current_axes.get_ylabel() == "Net Benefit"

    # Check if legend is present
    assert current_axes.get_legend() is not None

    # Check if grid is enabled
    assert any(line.get_visible() for line in current_axes.xaxis.get_gridlines()), (
        "X-axis grid is not visible"
    )
    assert any(line.get_visible() for line in current_axes.yaxis.get_gridlines()), (
        "Y-axis grid is not visible"
    )

    plt.close()  # Close the plot to free up memory


def test_integration_with_dca_results():
    """Test integration with DCA results."""
    data = load_binary_df()
    dca_results = get_dca_results(data)
    with patch("matplotlib.pyplot.show"):
        plot_graphs(dca_results, "net_benefit")


def test_plot_graphs_function_selection():
    """Test plot graphs function selection."""
    with patch("matplotlib.pyplot.show"):
        plot_graphs(SAMPLE_DATA_DF, "net_benefit")
        plot_graphs(SAMPLE_DATA_DF, "net_intervention_avoided")


def test_plot_graphs_custom_colors():
    """Test plot graphs with custom colors."""
    custom_colors = ["green", "purple"]
    with patch("matplotlib.pyplot.show"):
        plot_graphs(SAMPLE_DATA_DF, "net_benefit", color_names=custom_colors)


def test_plot_graphs_with_empty_dataframe():
    """Test plot graphs with empty dataframe."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        plot_graphs(empty_df, "net_benefit")


def test_validate_markers_valid():
    """Test that valid markers are accepted."""
    # Test with a subset of valid markers
    valid_subset = ["o", "*", "s", "D"]
    # This should not raise an exception
    _validate_markers(valid_subset)

    # Test with None
    _validate_markers(None)

    # Test with empty list
    _validate_markers([])


def test_validate_markers_invalid():
    """Test that invalid markers raise a ValueError with appropriate message."""
    invalid_markers = ["o", "-", "*"]  # '-' is not a valid marker

    with pytest.raises(ValueError) as excinfo:
        _validate_markers(invalid_markers)

    # Check that the error message contains the invalid marker
    assert "'-'" in str(excinfo.value)
    # Check that the error message mentions valid markers
    assert "valid markers" in str(excinfo.value)
    # Check that some valid markers are mentioned in the error message
    for marker in ["o", "*", "s"]:
        assert f"'{marker}'" in str(excinfo.value)


def test_plot_graphs_with_invalid_markers():
    """Test that plot_graphs raises a ValueError when invalid markers are provided."""
    df = get_dca_results(load_binary_df())

    with pytest.raises(ValueError) as excinfo:
        plot_graphs(plot_df=df, markers=["*", "d", "-"])

    # Check that the error message contains the invalid marker
    assert "'-'" in str(excinfo.value)
    # Check that the error message mentions valid markers
    assert "valid markers" in str(excinfo.value)


def test_plot_graphs_linewidths_single_and_multiple(monkeypatch):
    """Test that linewidths parameter is handled correctly for single and multiple values."""

    sample_df = SAMPLE_DATA_DF.copy()

    # Track calls to plt.plot
    call_args = []

    def fake_plot(*args, **kwargs):
        call_args.append(kwargs)

    monkeypatch.setattr(plt, "plot", fake_plot)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # Case 1: single linewidth applies to all models
    call_args.clear()
    plot_graphs(sample_df, linewidths=[2.0], show_legend=False)
    assert all(kwargs.get("linewidth") == 2.0 for kwargs in call_args), (
        "Single linewidth not applied to all lines"
    )

    # Case 2: list length equals number of models
    call_args.clear()
    plot_graphs(sample_df, linewidths=[1.0, 3.0], show_legend=False)
    assert {kwargs.get("linewidth") for kwargs in call_args} == {1.0, 3.0}, (
        "Line-specific linewidths not applied"
    )

    # Case 3: invalid length should raise error
    with pytest.raises(ValueError):
        plot_graphs(sample_df, linewidths=[1.0, 2.0, 3.0])
