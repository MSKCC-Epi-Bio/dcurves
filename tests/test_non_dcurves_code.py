"""
This module contains tests for non-dcurves code, specifically the Cox Proportional Hazards model.
"""

import pandas as pd
import lifelines

from .load_test_data import load_tutorial_coxph_pr_failure18_vals

# pylint: enable=duplicate-code


def test_tutorial_python_coxph():
    """
    Test the Cox Proportional Hazards model implementation against R benchmark results.
    """
    r_coxph_pr_failure18_series = load_tutorial_coxph_pr_failure18_vals()["pr_failure18"]

    df_time_to_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/"
        "data/df_time_to_cancer_dx.csv"
    )
    cph = lifelines.CoxPHFitter()
    cph.fit(
        df=df_time_to_cancer_dx,
        duration_col="ttcancer",
        event_col="cancer",
        formula="age + famhistory + marker",
    )
    cph_pred_vals = cph.predict_survival_function(
        df_time_to_cancer_dx[["age", "famhistory", "marker"]], times=[1.5]
    )
    df_time_to_cancer_dx["pr_failure18"] = 1 - cph_pred_vals.iloc[0, :]

    assert (
        df_time_to_cancer_dx["pr_failure18"]
        .round(decimals=4)
        .equals(r_coxph_pr_failure18_series.round(decimals=4))
    )


# pylint: enable=duplicate-code
