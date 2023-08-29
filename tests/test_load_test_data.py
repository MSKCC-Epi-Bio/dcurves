# tests/test_data_loading.py

import pytest
import pandas as pd
import dcurves

def test_load_binary_df():
    # Act

    from dcurves.load_test_data import load_binary_df
    result = load_binary_df()


    # Assert
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Optionally, you can add more specific assertions on the shape, columns, or other properties of the DataFrame.

def test_load_survival_df():
    # Act
    from dcurves.load_test_data import load_survival_df

    result = load_survival_df()

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Optionally, add more specific assertions

def test_load_case_control_df():
    # Act
    from dcurves.load_test_data import load_case_control_df

    result = load_case_control_df()

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Optionally, add more specific assertions

# Running this will allow you to check the coverage.
# pytest --cov=dcurves tests/
