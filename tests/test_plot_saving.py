import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dcurves.plot_graphs import plot_graphs

# Set matplotlib to use a non-GUI backend to prevent plots from showing up
plt.switch_backend('Agg')

# Sample DataFrame for tests
@pytest.fixture
def sample_data_df():
    return pd.DataFrame({
        "model": ["Model1"] * 10 + ["Model2"] * 10,  # 20 data points for 2 models
        "threshold": np.linspace(0, 1, 10).tolist() * 2,  # 10 threshold values for each model
        "net_benefit": np.random.rand(20),  # Random net benefit values
        "net_intervention_avoided": np.random.rand(20),  # Random net interventions avoided
    })

# Test for saving the plot
def test_plot_graphs_saves_file(sample_data_df, tmp_path):
    file_name = tmp_path / "test_plot.png"
    dpi = 100

    # Call the plot function with the file saving parameters
    plot_graphs(sample_data_df, 'net_benefit', file_name=str(file_name), dpi=dpi)

    # Check if the file was created
    assert file_name.is_file(), "The plot file should exist after calling plot_graphs."

    # Optional: check file size or other properties if needed
