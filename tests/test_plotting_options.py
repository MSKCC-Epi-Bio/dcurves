"""
Unit tests for plotting options: colors, markers, gridlines, legends, linewidths.

These tests validate individual plotting parameters and error handling.
"""

import random
import re
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
    _get_colors,
    _validate_markers,
)

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


# Color tests


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


def test_fewer_color_names_than_models():
    """Test fewer color names than models."""
    sample_df = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.1],
            "net_benefit": [0.2, 0.3],
        }
    )
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


def test_plot_net_benefit_mismatched_color_names():
    """Test plot net benefit with mismatched color names."""
    df_two_models = pd.DataFrame(
        {
            "model": ["model1", "model2"],
            "threshold": [0.1, 0.2],
            "net_benefit": [0.3, 0.4],
        }
    )
    with pytest.raises(
        ValueError,
        match="color_names must be either a single value or match the number of unique models",
    ):
        _plot_net_benefit(
            df_two_models, y_limits=(-0.05, 1), color_names=["blue", "red", "green"]
        )


# Grid tests


def test_plot_net_benefit_grid_enabled():
    """Test plot net benefit with grid enabled."""
    plt.ioff()
    with (
        patch("matplotlib.pyplot.grid") as mock_grid,
        patch("matplotlib.pyplot.show"),
    ):
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_grid=True,
        )
        mock_grid.assert_called_with(
            color="black", which="both", axis="both", linewidth="0.3"
        )


# Legend tests


def test_plot_net_benefit_show_legend_enabled():
    """Test plot net benefit with legend enabled."""
    plt.ioff()
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_benefit_show_legend_disabled():
    """Test plot net benefit with legend disabled."""
    plt.ioff()
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):
        _plot_net_benefit(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


def test_plot_net_intervention_avoided_show_legend_enabled():
    """Test plot net intervention avoided with legend enabled."""
    plt.ioff()
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=True,
        )
        mock_legend.assert_called_once()


def test_plot_net_intervention_avoided_show_legend_disabled():
    """Test plot net intervention avoided with legend disabled."""
    plt.ioff()
    with (
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.show"),
    ):
        _plot_net_intervention_avoided(
            SAMPLE_DATA_DF,
            y_limits=(-0.05, 1),
            color_names=["blue", "red"],
            show_legend=False,
        )
        mock_legend.assert_not_called()


# Validation tests


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


# Marker tests


def test_validate_markers_valid():
    """Test that valid markers are accepted."""
    valid_subset = ["o", "*", "s", "D"]
    _validate_markers(valid_subset)
    _validate_markers(None)
    _validate_markers([])


def test_validate_markers_invalid():
    """Test that invalid markers raise a ValueError with appropriate message."""
    invalid_markers = ["o", "-", "*"]

    with pytest.raises(ValueError) as excinfo:
        _validate_markers(invalid_markers)

    assert "'-'" in str(excinfo.value)
    assert "valid markers" in str(excinfo.value)
    for marker in ["o", "*", "s"]:
        assert f"'{marker}'" in str(excinfo.value)


def test_plot_graphs_with_invalid_markers(df_binary):
    """Test that plot_graphs raises a ValueError when invalid markers are provided."""
    from dcurves import dca

    df = dca(
        data=df_binary,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=np.arange(0, 0.5, 0.01),
    )

    with pytest.raises(ValueError) as excinfo:
        plot_graphs(plot_df=df, markers=["*", "d", "-"])

    assert "'-'" in str(excinfo.value)
    assert "valid markers" in str(excinfo.value)


# Linewidth tests


def test_plot_graphs_linewidths_single_and_multiple(monkeypatch):
    """Test that linewidths parameter is handled correctly for single and multiple values."""

    sample_df = SAMPLE_DATA_DF.copy()

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


# Parametrized tests


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
