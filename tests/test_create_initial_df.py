"""
This module contains tests for the _create_initial_df function and related functionality.
"""

# Third-party imports
import pandas as pd
import pytest

# Local imports
from dcurves.dca import _create_initial_df, _calc_prevalence
from dcurves.risks import _create_risks_df
from dcurves.dca import _rectify_model_risk_boundaries


def test_create_initial_df():
    """
    Test the _create_initial_df function with default parameters.
    """
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    data = df_cancer_dx
    outcome = "cancer"
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    modelnames = ["famhistory"]
    prevalence = None
    thresholds = [i / 100 for i in range(0, 100)]
    harm = None

    risks_df = _create_risks_df(
        data=data,
        outcome=outcome,
        models_to_prob=models_to_prob,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=outcome,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
        harm=harm,
    )

    assert len(initial_df) % 3 == 0
    assert set(["all", "none"]).issubset(set(initial_df["model"].unique()))
    assert len(risks_df) == initial_df["n"][0]
    assert prevalence_value == initial_df["prevalence"][0]
    assert initial_df["harm"][0] == 0


def test_create_initial_df_harms():
    """
    Test that a non-dict non-None harm raises the correct exception.
    """
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    data = df_cancer_dx
    outcome = "cancer"
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    modelnames = ["famhistory"]
    prevalence = None
    thresholds = [i / 100 for i in range(0, 100)]
    harm = "a"

    risks_df = _create_risks_df(
        data=data,
        outcome=outcome,
        models_to_prob=models_to_prob,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=outcome,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    with pytest.raises(ValueError, match="Harm should be either None or dict"):
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(rectified_risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm,
        )


# TODO: Add tests for when harm is specified to make sure each model has a different associated harm
