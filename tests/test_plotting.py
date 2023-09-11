

import pandas as pd
import numpy as np
import random
import re
import pytest

from .load_test_data import load_binary_df
from dcurves.dca import dca
from dcurves.plot_graphs import (
    plot_graphs,
    _plot_net_benefit,
    _plot_net_intervention_avoided,
    _get_colors,
)

from unittest.mock import patch

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
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.show")
    func(SAMPLE_DATA_DF, color_names=DEFAULT_COLOR_NAMES, **args)

def test_invalid_graph_type():
    import pytest
    import pandas as pd
    df = pd.DataFrame({'model': ['model1'], 'threshold': [0.1], 'net_benefit': [0.2]})
    with pytest.raises(ValueError, match="graph_type must be one of 2 strings: net_benefit, net_intervention_avoided"):
        plot_graphs(plot_df=df, graph_type="invalid_type")

def test_fewer_color_names_than_models():
    df = pd.DataFrame({
        'model': ['model1', 'model2'],
        'threshold': [0.1, 0.1],
        'net_benefit': [0.2, 0.3]
    })
    with pytest.raises(ValueError, match="More predictors than color_names, please enter more color names in color_names list and try again"):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=["red"])


def test_custom_colors_net_benefit(mocker):

    mocker.patch("dcurves.plot_graphs")
    df = pd.DataFrame({'model': ['model1'], 'threshold': [0.1], 'net_benefit': [0.2]})
    with patch("matplotlib.pyplot.show"):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=["red"])
    # assertions can be added if necessary

def test_custom_colors_net_intervention_avoided(mocker):
    mocker.patch("dcurves.plot_graphs", return_value=None)
    df = pd.DataFrame({'model': ['model1'], 'threshold': [0.1], 'net_intervention_avoided': [0.3]})
    with patch("matplotlib.pyplot.show"):
        plot_graphs(plot_df=df, graph_type="net_intervention_avoided", color_names=["red"])
    # assertions can be added if necessary


def test_default_colors_net_benefit(mocker):
    mocker.patch("dcurves.plot_graphs", return_value=None)
    df = pd.DataFrame({'model': ['model1'], 'threshold': [0.1], 'net_benefit': [0.2]})
    with patch("matplotlib.pyplot.show"):
        plot_graphs(plot_df=df, graph_type="net_benefit", color_names=None)
    # assertions can be added if necessary