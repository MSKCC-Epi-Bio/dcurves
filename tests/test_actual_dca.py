"""Unit tests for the dca function and related data loading."""

import numpy as np
import pandas as pd
import pytest
from dcurves import dca
from .load_test_data import load_data


def test_load_data_binary():
    """Test loading binary data from CSV file."""
    data = load_data("df_binary.csv")

    assert isinstance(data, pd.DataFrame), "Data loaded is not a DataFrame"
    assert not data.empty, "Loaded data is empty"

    modelnames = ["famhistory"]
    for column in modelnames:
        assert column in data.columns, f"Column '{column}' not found in the dataframe"

    assert "cancer" in data.columns, "'cancer' column not found in the dataframe"


def test_dca_binary():
    """Test DCA function with binary data."""
    data = load_data("df_binary.csv")
    dca_results = dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=[i / 100 for i in range(0, 46)],
    )

    assert isinstance(dca_results, pd.DataFrame), "DCA results are not a DataFrame"


def test_load_data_surv():
    """Test loading survival data from CSV file."""
    data = load_data("df_surv.csv")

    assert isinstance(data, pd.DataFrame), "Data loaded is not a DataFrame"
    assert not data.empty, "Loaded data is empty"

    modelnames = ["famhistory", "marker", "cancerpredmarker"]
    for column in modelnames:
        assert column in data.columns, f"Column '{column}' not found in the dataframe"

    assert "cancer" in data.columns, "'cancer' column not found in the dataframe"

    for col in ["marker"]:
        assert col in data.columns, f"Column '{col}' not found in dataframe"
        assert np.issubdtype(
            data[col].dtype, np.number
        ), f"Expected numerical data type for '{col}'"


def test_dca_surv():
    """Test DCA function with survival data."""
    data = load_data("df_surv.csv")
    dca_results = dca(
        data=data,
        outcome="cancer",
        modelnames=["famhistory", "marker", "cancerpredmarker"],
        models_to_prob=["marker"],
        thresholds=[i / 100 for i in range(0, 46)],
        time_to_outcome_col="ttcancer",
        time=1,
    )

    assert isinstance(dca_results, pd.DataFrame), "DCA results are not a DataFrame"


def test_error_handling():
    """Test error handling for invalid inputs in DCA function."""
    data = load_data("df_binary.csv")

    with pytest.raises(KeyError):
        dca(data=data, outcome="nonexistent_column", modelnames=["famhistory"])
