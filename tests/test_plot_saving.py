import os
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Assuming plot_graphs is your function to be tested
from dcurves.plot_graphs import plot_graphs

@pytest.fixture
def sample_data_df():
    # Create a simple DataFrame suitable for plotting
    return pd.DataFrame({
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
        "net_benefit": [0.2, 0.3, 0.5, 0.6, 0.7],
        "model": ["Model1"] * 5
    })

def test_plot_graphs_saves_file(sample_data_df, tmp_path):
    file_name = tmp_path / "test_plot.png"
    dpi = 100

    # Call the plot function with the file saving parameters
    plot_graphs(sample_data_df, 'net_benefit', file_name=str(file_name), dpi=dpi)

    # Check if the file was created
    assert file_name.is_file(), "The plot file should exist after calling plot_graphs."

    # Optional: check file size or other properties if needed
