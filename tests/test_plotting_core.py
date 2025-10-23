"""
Core integration tests for DCA plotting functionality.

These tests validate end-to-end workflows: running DCA and plotting results.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs


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


def test_case1_plot_net_benefit(df_binary):
    """Test case 1 for plotting net benefit."""
    get_dca_results(df_binary)


def test_case1_plot_net_intervention_avoided(df_binary):
    """Test case 1 for plotting net intervention avoided."""
    get_dca_results(df_binary)


def test_plot_graphs_y_limits():
    """Test plot graphs with custom y limits."""
    sample_df = pd.DataFrame(
        {
            "model": ["Model1", "Model2"],
            "threshold": [0.2, 0.5],
            "net_benefit": [0.7, 0.8],
            "net_intervention_avoided": [0.3, 0.4],
        }
    )
    custom_y_limits = (0, 0.5)

    with patch("matplotlib.pyplot.show"):
        plot_graphs(sample_df, "net_benefit", y_limits=custom_y_limits)

    current_axes = plt.gca()

    assert current_axes.get_ylim() == custom_y_limits

    unique_models = sample_df["model"].unique()
    assert len(current_axes.lines) >= len(unique_models)

    line_labels = [line.get_label() for line in current_axes.lines]
    for model in unique_models:
        assert model in line_labels, f"No line found for model {model}"

    assert current_axes.get_xlabel() == "Threshold Probability"
    assert current_axes.get_ylabel() == "Net Benefit"
    assert current_axes.get_legend() is not None

    assert any(line.get_visible() for line in current_axes.xaxis.get_gridlines())
    assert any(line.get_visible() for line in current_axes.yaxis.get_gridlines())

    plt.close()


def test_integration_with_dca_results(df_binary):
    """Test integration with DCA results."""
    dca_results = get_dca_results(df_binary)
    with patch("matplotlib.pyplot.show"):
        plot_graphs(dca_results, "net_benefit")


def test_plot_graphs_function_selection():
    """Test plot graphs function selection."""
    sample_df = pd.DataFrame(
        {
            "model": ["Model1", "Model2"],
            "threshold": [0.2, 0.5],
            "net_benefit": [0.7, 0.8],
            "net_intervention_avoided": [0.3, 0.4],
        }
    )
    with patch("matplotlib.pyplot.show"):
        plot_graphs(sample_df, "net_benefit")
        plot_graphs(sample_df, "net_intervention_avoided")

