"""
This module contains tests for the data loading functions in the dcurves package.
"""

import pandas as pd
from dcurves.load_test_data import (
    load_binary_df,
    load_survival_df,
    load_case_control_df,
)


def test_load_binary_df():
    """Test loading of binary data."""
    result = load_binary_df()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_load_survival_df():
    """Test loading of survival data."""
    result = load_survival_df()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_load_case_control_df():
    """Test loading of case-control data."""
    result = load_case_control_df()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
