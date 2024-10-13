from dcurves import dca
from .load_test_data import load_data
import numpy as np
import pandas as pd  # Assuming the output is a DataFrame
import pytest


# Test for df_binary.csv
def test_load_data_binary():
    data = load_data("df_binary.csv")

    # Check for DataFrame type
    assert isinstance(data, pd.DataFrame), "Data loaded is not a DataFrame"

    # Check for Non-Empty Data
    assert not data.empty, "Loaded data is empty"

    # Check for the existence of columns given to modelnames
    modelnames = ["famhistory"]
    for column in modelnames:
        assert column in data.columns, f"Column '{column}' not found in the dataframe"

    # Check for outcome column
    assert "cancer" in data.columns, "'cancer' column not found in the dataframe"


def test_dca_binary():
    data = load_data("df_binary.csv")
    dca_results = dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=[i / 100 for i in range(0, 46)],
    )

    # Check if the output is a DataFrame
    assert isinstance(dca_results, pd.DataFrame), "DCA results are not a DataFrame"


# Test for df_surv.csv
def test_load_data_surv():
    data = load_data("df_surv.csv")

    # Check for DataFrame type
    assert isinstance(data, pd.DataFrame), "Data loaded is not a DataFrame"

    # Check for Non-Empty Data
    assert not data.empty, "Loaded data is empty"

    # Check for the existence of columns given to modelnames
    modelnames = ["famhistory", "marker", "cancerpredmarker"]
    for column in modelnames:
        assert column in data.columns, f"Column '{column}' not found in the dataframe"

    # Check for outcome column
    assert "cancer" in data.columns, "'cancer' column not found in the dataframe"

    # Check models_to_prob
    for col in ["marker"]:
        assert col in data.columns, f"Column '{col}' not found in dataframe"
        assert np.issubdtype(
            data[col].dtype, np.number
        ), f"Expected numerical data type for '{col}'"


def test_dca_surv():
    data = load_data("df_surv.csv")
    dca_results = dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory", "marker", "cancerpredmarker"],
        models_to_prob=["marker"],
        thresholds=[i / 100 for i in range(0, 46)],
        time_to_outcome_col="ttcancer",
        time=1
    )

    # Check if the output is a DataFrame
    assert isinstance(dca_results, pd.DataFrame), "DCA results are not a DataFrame"


def test_error_handling():
    data = load_data("df_binary.csv")

    # Test for erroneous inputs
    with pytest.raises(
        Exception
    ):  # Replace Exception with the specific exception you expect
        dca(data=data, outcome="nonexistent_column", modelnames=["famhistory"])
