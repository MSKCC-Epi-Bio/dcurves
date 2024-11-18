"""
This module contains tests for the risk calculation functions in the dcurves package.
"""

import sys

import pandas as pd

from dcurves.risks import (
    _calc_binary_risks,
    _calc_surv_risks,
    _create_risks_df,
    _rectify_model_risk_boundaries,
)
from .load_test_data import (
    load_test_surv_risk_test_df,
    load_binary_df,
    load_survival_df,
    load_tutorial_bin_marker_risks_list,
)


def test_bin_dca_risks_calc():
    """
    Test binary DCA risk calculation against known results.
    """
    r_marker_risks_df = load_tutorial_bin_marker_risks_list().copy()
    r_marker_risks_df = sorted(r_marker_risks_df["marker_risk"].tolist()).copy()
    r_marker_risks = [round(x, 10) for x in r_marker_risks_df].copy()

    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    binary_risks = sorted(_calc_binary_risks(data=df_cancer_dx, outcome="cancer", model="marker"))
    p_marker_risks = [round(x, 10) for x in binary_risks]

    assert r_marker_risks == p_marker_risks


def test_surv_dca_risks_calc():
    """
    Test survival DCA risk calculation against known results.
    """
    r_surv_risk_test_df = load_test_surv_risk_test_df()

    surv_df = load_survival_df()

    surv_marker_calc_list = _calc_surv_risks(
        data=surv_df,
        outcome="cancer",
        model="marker",
        time=2,
        time_to_outcome_col="ttcancer",
    )

    comp_df = (
        pd.DataFrame(
            {
                "r_marker_risks": r_surv_risk_test_df["marker"].tolist(),
                "p_marker_risks": surv_marker_calc_list,
            }
        )
        .round(decimals=4)
        .copy()
    )

    assert comp_df["r_marker_risks"].equals(comp_df["p_marker_risks"])


def test_rectify_model_risk_boundaries():
    """
    Test the rectification of model risk boundaries.
    """
    data = load_binary_df()
    modelnames = ["famhistory"]

    risks_df = _create_risks_df(data=data, outcome="cancer")

    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)

    machine_epsilon = sys.float_info.epsilon

    assert rectified_risks_df["all"][0] == 1 + machine_epsilon
    assert not rectified_risks_df["all"][0] == 1
    assert rectified_risks_df["none"][0] == 0 - machine_epsilon
    assert not rectified_risks_df["none"][0] == 0
