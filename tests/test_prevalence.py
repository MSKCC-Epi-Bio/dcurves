"""
Tests for prevalence calculation in binary and survival cases.
"""

# Third-party imports
import pytest

# Load _calc_prevalence, other necessary functions
from dcurves.prevalence import _calc_prevalence

# Load Data
from .load_test_data import load_binary_df, load_survival_df
from .load_test_data import load_r_case1_results


def test_binary_prevalence():
    """
    Test prevalence calculation for binary outcome.
    """
    data = load_binary_df()
    r_benchmark_results = load_r_case1_results()
    outcome = "cancer"
    time = 1
    time_to_outcome_col = None
    prevalence = None

    prevalence_value = _calc_prevalence(
        risks_df=data,
        outcome=outcome,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    assert prevalence_value == r_benchmark_results.pos_rate[0]

    prevalence_value = _calc_prevalence(
        risks_df=data,
        outcome=outcome,
        prevalence=0.5,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    assert prevalence_value == 0.5


def test_survival_prevalence():
    """
    Test prevalence calculation for survival outcome.
    """
    surv_df = load_survival_df()

    local_prevalence_calc = _calc_prevalence(
        risks_df=surv_df,
        outcome="cancer",
        prevalence=None,
        time=1,
        time_to_outcome_col="ttcancer",
    )

    assert round(number=float(local_prevalence_calc), ndigits=6) == 0.147287


def test_prevalence_in_survival_case():
    """
    Test that providing prevalence for survival outcomes raises an error.
    """
    surv_df = load_survival_df()

    # This should raise an error since prevalence is supplied for survival outcomes
    with pytest.raises(ValueError, match="In survival outcomes, prevalence should not be supplied"):
        _calc_prevalence(
            risks_df=surv_df,
            outcome="cancer",
            prevalence=0.5,  # Providing prevalence for a survival outcome
            time=1,
            time_to_outcome_col="ttcancer",  # Indicating this is a survival case
        )
