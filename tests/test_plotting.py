import pandas as pd
import numpy as np
import random
import re
import pytest
from unittest.mock import patch

from .load_test_data import load_binary_df
from dcurves.dca import dca
from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
    _get_colors,
)

# Constants
SAMPLE_DATA_DF = pd.DataFrame({
    "model": ["Model1", "Model2"],
    "threshold": [0.2, 0.5],
    "net_benefit": [0.7, 0.8],
    "net_intervention_avoided": [0.3, 0.4],
})

DEFAULT_COLOR_NAMES = ["red", "blue"]


def get_dca_results(data, thresholds=None):
    return dca(
        data=data,
        outcome='cancer',
        modelnames=['famhistory'],
        thresholds=thresholds or np.arange(0, 0.5, 0.01)
    )


def test_2_case1_plot_net_benefit():
    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
    dca_famhistory_df = get_dca_results(df_cancer_dx, [i/100 for i in range(0, 46)])


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
    assert len(colors) == num_colors, f"Expected {num_colors} colors, got {len(colors)} colors."

    # Check that each color is a valid color code
    for color in colors:
        assert re.match(r'^#[0-9A-Fa-f]{6}$', color), f"Invalid color code: {color}"

    # Check for repeatability using a seed
    random.seed(42)
    assert colors == _get_colors(num_colors=num_colors), "Expected the same colors with the same seed."


@pytest.mark.parametrize(
    "func, args",
    [
        (_plot_net_benefit, {"y_limits": (-0.05, 1)}),
        (_plot_net_intervention_avoided, {}),
    ]
)
def test_plot_functions(func, args, mocker):
    mocker.patch("matplotlib.pyplot.plot", autospec=True)
    mocker.patch("matplotlib.pyplot.show", autospec=True)
    func(SAMPLE_DATA_DF, color_names=DEFAULT_COLOR_NAMES, **args)


def test_plot_graphs():
    # The plot_graphs functions can be tested together using pytest's parametrize feature
    with patch("dcurves.plot_graphs._plot_net_benefit") as mock_plot_net_benefit:
        plot_graphs(plot_df=SAMPLE_DATA_DF, graph_type="net_benefit", color_names=DEFAULT_COLOR_NAMES)
        mock_plot_net_benefit.assert_called_once_with(plot_df=SAMPLE_DATA_DF, y_limits=(-0.05, 1), color_names=DEFAULT_COLOR_NAMES)

    with patch("dcurves.plot_graphs._plot_net_intervention_avoided") as mock_plot_net_intervention_avoided:
        plot_graphs(plot_df=SAMPLE_DATA_DF, graph_type="net_intervention_avoided", color_names=DEFAULT_COLOR_NAMES)
        mock_plot_net_intervention_avoided.assert_called_once_with(plot_df=SAMPLE_DATA_DF, y_limits=(-0.05, 1), color_names=DEFAULT_COLOR_NAMES)


def test_plot_graphs_exceptions():
    with pytest.raises(ValueError):
        plot_graphs(plot_df=SAMPLE_DATA_DF, graph_type="invalid_type")

    with pytest.raises(ValueError):
        plot_graphs(plot_df=SAMPLE_DATA_DF, graph_type="net_benefit", color_names=["red"])

    with patch("dcurves.plot_graphs._get_colors", return_value=DEFAULT_COLOR_NAMES) as mock_get_colors:
        with patch("dcurves.plot_graphs._plot_net_benefit") as mock_plot_net_benefit:
            plot_graphs(plot_df=SAMPLE_DATA_DF, graph_type="net_benefit")
            mock_get_colors.assert_called_once_with(num_colors=2)
            mock_plot_net_benefit.assert_called_once_with(plot_df=SAMPLE_DATA_DF, y_limits=(-0.05, 1), color_names=DEFAULT_COLOR_NAMES)
