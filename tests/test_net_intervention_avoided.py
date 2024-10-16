"""
This module contains tests for the net intervention avoided calculations in the DCA function.
"""

from dcurves.dca import dca
from .load_test_data import (
    load_binary_df,
    load_survival_df,
    load_r_case1_results,
    load_r_case2_results,
)


def test_case1_binary_net_interventions_avoided():
    """Test net interventions avoided for binary case."""
    data = load_binary_df()
    outcome = "cancer"
    modelnames = ["famhistory"]

    dca_results_df = dca(data=data, outcome=outcome, modelnames=modelnames)

    r_results_df = load_r_case1_results()

    for model in ["all", "none", "famhistory"]:
        dca_model_nia = (
            dca_results_df[dca_results_df.model == model]["net_intervention_avoided"]
            .round(decimals=6)
            .reset_index(drop=True)
        )
        r_model_nia = (
            r_results_df[r_results_df.variable == model]["net_intervention_avoided"]
            .round(decimals=6)
            .reset_index(drop=True)
        )
        assert dca_model_nia.equals(r_model_nia)


def test_case2_surv_net_interventions_avoided():
    """Test net interventions avoided for survival case."""
    data = load_survival_df()
    outcome = "cancer"
    modelnames = ["cancerpredmarker"]
    time = 1
    time_to_outcome_col = "ttcancer"

    dca_results_df = dca(
        data=data,
        outcome=outcome,
        modelnames=modelnames,
        time_to_outcome_col=time_to_outcome_col,
        time=time,
    )

    r_results_df = load_r_case2_results()

    for model in ["all", "none", "cancerpredmarker"]:
        dca_model_nia = (
            dca_results_df[dca_results_df.model == model]["net_intervention_avoided"]
            .round(decimals=6)
            .reset_index(drop=True)
        )
        r_model_nia = (
            r_results_df[r_results_df.variable == model]["net_intervention_avoided"]
            .round(decimals=6)
            .reset_index(drop=True)
        )
        assert dca_model_nia.equals(r_model_nia)
