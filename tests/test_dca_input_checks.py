"""
This module contains tests for input validation in the dca function.
"""

import pytest
import numpy as np

from dcurves.dca import dca


def test_dca_data_argument_check():
    """
    Test that the dca function raises a ValueError when given incorrect data type.
    """
    # Using a numpy array (or any other type) instead of a pandas DataFrame
    incorrect_data_type = np.array([[1, 2, 3], [4, 5, 6]])

    # Other arguments for dca function
    outcome = "some_outcome"
    modelnames = ["model1", "model2"]
    thresholds = [0.1, 0.2, 0.3]
    harm = 0.05
    models_to_prob = None
    prevalence = 0.1
    time = 5
    time_to_outcome_col = "time_to_event"
    nper = 1

    # Expecting a ValueError when incorrect data type is passed to dca function
    with pytest.raises(ValueError) as exc_info:
        dca(
            data=incorrect_data_type,
            outcome=outcome,
            modelnames=modelnames,
            thresholds=thresholds,
            harm=harm,
            models_to_prob=models_to_prob,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col,
            nper=nper,
        )

    # Asserting that the error message is as expected
    assert "'data' must be a pandas DataFrame" in str(exc_info.value)
