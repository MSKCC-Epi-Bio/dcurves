# pylint: disable=redefined-outer-name
"""
This module contains tests for plot smoothing functionality.
"""

# pylint: disable=wrong-import-position,wrong-import-order
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from dcurves.plot_graphs import plot_graphs, _plot_net_benefit

# pylint: enable=wrong-import-position,wrong-import-order


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress specific warnings during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


@pytest.fixture
def sample_data_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["Model1"] * 10 + ["Model2"] * 10,  # 20 data points for 2 models
            "threshold": np.linspace(0, 1, 10).tolist() * 2,  # 10 threshold values for each model
            "net_benefit": np.random.rand(20),  # Random net benefit values
            "net_intervention_avoided": np.random.rand(20),  # Random net interventions avoided
        }
    )


@pytest.fixture
def color_names():
    """Return a list of color names for testing."""
    return ["red", "blue"]


def test_plot_graphs_with_smoothing(sample_data_df, color_names):
    """Test plotting with smoothing enabled."""
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show") as mock_show:
        plot_graphs(sample_data_df, "net_benefit", smooth_frac=0.2, color_names=color_names)
        assert mock_plot.called
        assert mock_show.called


def test_plot_net_benefit_with_smoothing(sample_data_df, color_names):
    """Test plotting net benefit with smoothing."""
    smoothed_data = {
        "Model1": np.array([[0, 0.1], [0.5, 0.5], [1, 0.9]]),
        "Model2": np.array([[0, 0.2], [0.5, 0.6], [1, 0.8]]),
    }

    with patch("matplotlib.pyplot.plot") as mock_plot:
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=smoothed_data,
        )
        expected_calls = len(sample_data_df["model"].unique())
        assert (
            mock_plot.call_count == expected_calls
        ), f"Expected {expected_calls} plot calls, got {mock_plot.call_count}"


def test_plot_with_invalid_smoothed_data(sample_data_df, color_names):
    """Test plotting with invalid smoothed data."""
    invalid_smoothed_data = {"Model1": "invalid_data"}  # Non-array data
    with pytest.raises(ValueError):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=invalid_smoothed_data,
        )


def test_plot_without_smoothed_data(sample_data_df, color_names):
    """Test plotting without smoothed data."""
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        plot_graphs(sample_data_df, "net_benefit", smooth_frac=0.2, color_names=color_names)
        assert mock_plot.called


def test_smoothed_data_with_unmatched_model_names(sample_data_df, color_names):
    """Test plotting with unmatched model names in smoothed data."""
    unmatched_smoothed_data = {"Model3": np.array([[0, 0.1], [0.5, 0.5], [1, 0.9]])}
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=unmatched_smoothed_data,
        )
        assert mock_plot.call_count == len(sample_data_df["model"].unique())


def test_multiple_smoothed_lines_per_model(sample_data_df, color_names):
    """Test plotting with multiple smoothed lines per model."""
    single_line_smoothed_data = {"Model1": np.array([[0, 0.1], [0.5, 0.5], [1, 0.9]])}
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=single_line_smoothed_data,
        )
        assert mock_plot.called


def test_smoothed_data_with_missing_points(sample_data_df, color_names):
    """Test plotting with missing points in smoothed data."""
    missing_points_smoothed_data = {
        "Model1": np.array([[0, 0.1], [1, 0.9]])  # Missing intermediate points
    }
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=missing_points_smoothed_data,
        )
        assert mock_plot.called


def test_empty_smoothed_data(sample_data_df, color_names):
    """Test plotting with empty smoothed data."""
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data={},
        )
        assert mock_plot.call_count == len(sample_data_df["model"].unique())


def test_smooth_frac_validation():
    """Test smooth_frac validation."""
    sample_data = pd.DataFrame(
        {
            "model": ["Model1"],
            "threshold": [0.1],
            "net_benefit": [0.1],
        }
    )

    with pytest.raises(ValueError, match="smooth_frac must be between 0 and 1"):
        plot_graphs(sample_data, "net_benefit", smooth_frac=1.1)

    with pytest.raises(ValueError, match="smooth_frac must be between 0 and 1"):
        plot_graphs(sample_data, "net_benefit", smooth_frac=-0.1)


def test_smooth_frac_range():
    """Test smooth_frac range validation."""
    sample_data = pd.DataFrame(
        {
            "model": ["Model1"],
            "threshold": [0.1],
            "net_benefit": [0.1],
        }
    )

    with pytest.raises(ValueError):
        plot_graphs(sample_data, "net_benefit", smooth_frac=-0.1)

    with pytest.raises(ValueError):
        plot_graphs(sample_data, "net_benefit", smooth_frac=1.1)

    try:
        plot_graphs(sample_data, "net_benefit", smooth_frac=0.5)
    except ValueError:
        pytest.fail("smooth_frac within the valid range raised ValueError")
