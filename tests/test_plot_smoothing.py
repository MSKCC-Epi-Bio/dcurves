import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittest.mock import patch
import pandas as pd
import numpy as np

from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
)


# Sample DataFrame for tests
@pytest.fixture
def sample_data_df():
    return pd.DataFrame(
        {
            "model": ["Model1"] * 10 + ["Model2"] * 10,  # 20 data points for 2 models
            "threshold": np.linspace(0, 1, 10).tolist()
            * 2,  # 10 threshold values for each model
            "net_benefit": np.random.rand(20),  # Random net benefit values
            "net_intervention_avoided": np.random.rand(
                20
            ),  # Random net interventions avoided
        }
    )


# Mocked color names
@pytest.fixture
def color_names():
    return ["red", "blue"]


# Test for plotting with smoothing enabled
def test_plot_graphs_with_smoothing(sample_data_df, color_names):
    with patch("matplotlib.pyplot.plot") as mock_plot, patch(
        "matplotlib.pyplot.show"
    ) as mock_show:
        plot_graphs(
            sample_data_df, "net_benefit", smooth_frac=0.2, color_names=color_names
        )
        # Assertions to verify that the plot functions were called correctly
        assert mock_plot.called
        assert mock_show.called


def test_plot_net_benefit_with_smoothing(sample_data_df, color_names):
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

        # Expect one call per model
        expected_calls = len(sample_data_df["model"].unique())
        assert (
            mock_plot.call_count == expected_calls
        ), f"Expected {expected_calls} plot calls, got {mock_plot.call_count}"


def test_plot_with_invalid_smoothed_data(sample_data_df, color_names):
    invalid_smoothed_data = {"Model1": "invalid_data"}  # Non-array data
    with pytest.raises(ValueError):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=invalid_smoothed_data,
        )


def test_plot_without_smoothed_data(sample_data_df, color_names):
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        plot_graphs(
            sample_data_df, "net_benefit", smooth_frac=0.2, color_names=color_names
        )
        assert mock_plot.called


def test_smoothed_data_with_unmatched_model_names(sample_data_df, color_names):
    unmatched_smoothed_data = {
        "Model3": np.array([[0, 0.1], [0.5, 0.5], [1, 0.9]])
    }  # Model3 does not exist in sample_data_df
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=unmatched_smoothed_data,
        )
        assert mock_plot.call_count == len(
            sample_data_df["model"].unique()
        )  # Only plots for existing models


def test_multiple_smoothed_lines_per_model(sample_data_df, color_names):
    # Combine multiple lines of smoothed data into a single array for each model
    # For simplicity, we're just taking one of the smoothed data sets for this test
    single_line_smoothed_data = {"Model1": np.array([[0, 0.1], [0.5, 0.5], [1, 0.9]])}

    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data=single_line_smoothed_data,
        )
        # Now the function should not raise a ValueError
        assert mock_plot.called


def test_smoothed_data_with_missing_points(sample_data_df, color_names):
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
    with patch("matplotlib.pyplot.plot") as mock_plot, patch("matplotlib.pyplot.show"):
        _plot_net_benefit(
            sample_data_df,
            y_limits=(-0.05, 1),
            color_names=color_names,
            smoothed_data={},
        )
        assert mock_plot.call_count == len(
            sample_data_df["model"].unique()
        )  # Fallback to plotting raw data


def test_smooth_frac_validation():
    # Prepare a sample DataFrame
    sample_data = pd.DataFrame(
        {
            "model": ["Model1"],
            "threshold": [0.1],
            "net_benefit": [0.1],
        }
    )

    # Test smooth_frac outside of valid range
    with pytest.raises(ValueError) as e:
        plot_graphs(sample_data, "net_benefit", smooth_frac=1.1)  # Greater than 1
    assert "smooth_frac must be between 0 and 1" in str(e.value)

    with pytest.raises(ValueError) as e:
        plot_graphs(sample_data, "net_benefit", smooth_frac=-0.1)  # Less than 0
    assert "smooth_frac must be between 0 and 1" in str(e.value)


def test_smooth_frac_range():
    sample_data = pd.DataFrame(
        {
            "model": ["Model1"],
            "threshold": [0.1],
            "net_benefit": [0.1],
        }
    )

    # Test for smooth_frac less than 0
    with pytest.raises(ValueError):
        plot_graphs(sample_data, "net_benefit", smooth_frac=-0.1)

    # Test for smooth_frac greater than 1
    with pytest.raises(ValueError):
        plot_graphs(sample_data, "net_benefit", smooth_frac=1.1)

    # Test for smooth_frac within the valid range
    try:
        plot_graphs(sample_data, "net_benefit", smooth_frac=0.5)
    except ValueError:
        pytest.fail("smooth_frac within the valid range raised ValueError")
