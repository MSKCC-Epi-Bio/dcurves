"""
This module contains tests for plot saving functionality.
"""

# pylint: disable=redefined-outer-name

import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dcurves.plot_graphs import plot_graphs

# Set matplotlib to use a non-GUI backend to prevent plots from showing up
plt.switch_backend("Agg")


@pytest.fixture
def sample_data():
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
def mock_show(mocker):
    """Mock plt.show() to avoid UserWarning"""
    return mocker.patch("matplotlib.pyplot.show")


def test_plot_graphs_saves_file(sample_data, tmp_path, mock_show):
    """Test that plot_graphs function saves the file correctly."""
    file_name = tmp_path / "test_plot.png"
    dpi = 100

    plot_graphs(sample_data, "net_benefit", file_name=str(file_name), dpi=dpi)

    assert file_name.is_file(), "The plot file should exist after calling plot_graphs."
    mock_show.assert_called_once()  # Ensure plt.show() was called
