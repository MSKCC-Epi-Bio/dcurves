"""
This module contains tests for optional gridlines in plot functions.
"""

import pytest
import pandas as pd
from dcurves.plot_graphs import _plot_net_benefit, _plot_net_intervention_avoided

# pylint: disable=redefined-outer-name

# Sample DataFrame setup for testing
SAMPLE_DATA_DF = pd.DataFrame(
    {
        "model": ["Model1", "Model2"],
        "threshold": [0.2, 0.5],
        "net_benefit": [0.7, 0.8],
        "net_intervention_avoided": [0.3, 0.4],
    }
)

DEFAULT_COLOR_NAMES = ["red", "blue"]


@pytest.fixture
def mock_matplotlib(mocker):
    """Set up mock for matplotlib functions."""
    mocker.patch("matplotlib.pyplot.show")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.ylim")
    mocker.patch("matplotlib.pyplot.legend")
    return mocker.patch("matplotlib.pyplot.grid")


def test_plot_net_benefit_grid_enabled(mock_matplotlib):
    """Test that grid is enabled for net benefit plot."""
    _plot_net_benefit(
        SAMPLE_DATA_DF,
        y_limits=(-0.05, 1),
        color_names=DEFAULT_COLOR_NAMES,
        show_grid=True,
    )
    mock_matplotlib.assert_called_with(color="black", which="both", axis="both", linewidth="0.3")


def test_plot_net_intervention_avoided_grid_enabled(mock_matplotlib):
    """Test that grid is enabled for net intervention avoided plot."""
    _plot_net_intervention_avoided(
        SAMPLE_DATA_DF,
        y_limits=(-0.05, 1),
        color_names=DEFAULT_COLOR_NAMES,
        show_grid=True,
    )
    mock_matplotlib.assert_called_with(color="black", which="both", axis="both", linewidth="0.3")


def test_plot_net_benefit_grid_disabled(mock_matplotlib):
    """Test that grid is disabled for net benefit plot."""
    _plot_net_benefit(
        SAMPLE_DATA_DF,
        y_limits=(-0.05, 1),
        color_names=DEFAULT_COLOR_NAMES,
        show_grid=False,
    )
    mock_matplotlib.assert_called_with(False)


def test_plot_net_intervention_avoided_grid_disabled(mock_matplotlib):
    """Test that grid is disabled for net intervention avoided plot."""
    _plot_net_intervention_avoided(
        SAMPLE_DATA_DF,
        y_limits=(-0.05, 1),
        color_names=DEFAULT_COLOR_NAMES,
        show_grid=False,
    )
    mock_matplotlib.assert_called_with(False)
