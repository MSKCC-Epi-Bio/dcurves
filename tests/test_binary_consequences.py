"""
This module contains tests for binary consequences in the dcurves package.
It compares the results of various calculations against R benchmark results.
"""

import pandas as pd

from dcurves.dca import (
    _calc_tp_rate,
    _calc_fp_rate,
    _calc_prevalence,
    _create_initial_df,
    _calc_initial_stats,
    _calc_more_stats,
    _rectify_model_risk_boundaries,
)
from dcurves.risks import _create_risks_df
from .load_test_data import load_r_case1_results, load_binary_df


def _setup_common_variables():
    """Set up common variables used across multiple tests."""
    data = load_binary_df()
    outcome = "cancer"
    modelnames = ["famhistory"]
    thresholds = [i / 100 for i in range(0, 100)]
    return data, outcome, modelnames, thresholds


def test_case1_binary_test_pos_rate():
    """Test the test positive rate calculation for binary case 1."""
    data, outcome, modelnames, thresholds = _setup_common_variables()
    r_benchmark_results = load_r_case1_results()

    r_benchmark_test_pos_rates_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "famhistory": r_benchmark_results[r_benchmark_results["variable"] == "famhistory"][
                "test_pos_rate"
            ].reset_index(drop=True),
            "all": r_benchmark_results[r_benchmark_results["variable"] == "all"][
                "test_pos_rate"
            ].reset_index(drop=True),
            "none": r_benchmark_results[r_benchmark_results["variable"] == "none"][
                "test_pos_rate"
            ].reset_index(drop=True),
        }
    )

    risks_df = _create_risks_df(data=data, outcome=outcome)
    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)
    prevalence_value = _calc_prevalence(risks_df=rectified_risks_df, outcome=outcome)
    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
    )
    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        outcome=outcome,
        prevalence_value=prevalence_value,
    )

    p_test_pos_rate_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "famhistory": initial_stats_df[initial_stats_df["model"] == "famhistory"][
                "test_pos_rate"
            ].reset_index(drop=True),
            "all": initial_stats_df[initial_stats_df["model"] == "all"][
                "test_pos_rate"
            ].reset_index(drop=True),
            "none": initial_stats_df[initial_stats_df["model"] == "none"][
                "test_pos_rate"
            ].reset_index(drop=True),
        }
    )

    for model in ["famhistory", "all", "none"]:
        assert (
            r_benchmark_test_pos_rates_df[model]
            .round(decimals=6)
            .equals(p_test_pos_rate_df[model].round(decimals=6))
        )


def test_case1_binary_tp_rate():
    """Test the true positive rate calculation for binary case 1."""
    data, outcome, modelnames, thresholds = _setup_common_variables()
    r_benchmark_results_df = load_r_case1_results()

    risks_df = _create_risks_df(data=data, outcome=outcome)
    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)
    prevalence_value = _calc_prevalence(risks_df=rectified_risks_df, outcome=outcome)

    p_tp_rate = _calc_tp_rate(
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        model=modelnames[0],
        outcome=outcome,
        prevalence_value=prevalence_value,
    )

    bm_tp_rate = r_benchmark_results_df[
        r_benchmark_results_df.variable == "famhistory"
    ].tp_rate.reset_index(drop=True)

    assert bm_tp_rate.round(decimals=6).equals(p_tp_rate.round(decimals=6))


def test_case1_binary_fp_rate():
    """Test the false positive rate calculation for binary case 1."""
    data, outcome, modelnames, thresholds = _setup_common_variables()
    r_benchmark_results_df = load_r_case1_results()

    risks_df = _create_risks_df(data=data, outcome=outcome)
    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)
    prevalence_value = _calc_prevalence(risks_df=rectified_risks_df, outcome=outcome)

    p_fp_rate = _calc_fp_rate(
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        model=modelnames[0],
        outcome=outcome,
        prevalence_value=prevalence_value,
    )

    bm_fp_rate = r_benchmark_results_df[
        r_benchmark_results_df.variable == "famhistory"
    ].fp_rate.reset_index(drop=True)

    assert bm_fp_rate.round(decimals=6).equals(p_fp_rate.round(decimals=6))


def test_case1_binary_calc_initial_stats():
    """Test the initial statistics calculation for binary case 1."""
    data, outcome, modelnames, thresholds = _setup_common_variables()
    r_benchmark_results_df = load_r_case1_results()

    risks_df = _create_risks_df(data=data, outcome=outcome)
    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)
    prevalence_value = _calc_prevalence(risks_df=rectified_risks_df, outcome=outcome)
    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
    )
    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        outcome=outcome,
        prevalence_value=prevalence_value,
    )

    for model in ["all", "none", "famhistory"]:
        for stat in ["test_pos_rate", "tp_rate", "fp_rate"]:
            model_df = (
                initial_stats_df[initial_stats_df.model == model][stat]
                .round(decimals=6)
                .reset_index(drop=True)
            )
            benchmark_df = (
                r_benchmark_results_df[r_benchmark_results_df.variable == model][stat]
                .round(decimals=6)
                .reset_index(drop=True)
            )
            assert model_df.equals(benchmark_df)


def test_case1_binary_calc_more_stats():
    """Test the calculation of additional statistics for binary case 1."""
    data, outcome, modelnames, thresholds = _setup_common_variables()
    r_benchmark_results_df = load_r_case1_results()

    risks_df = _create_risks_df(data=data, outcome=outcome)
    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)
    prevalence_value = _calc_prevalence(risks_df=rectified_risks_df, outcome=outcome)
    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
    )
    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        outcome=outcome,
        prevalence_value=prevalence_value,
    )
    final_dca_df = _calc_more_stats(initial_stats_df=initial_stats_df)

    for model in ["all", "none", "famhistory"]:
        assert (
            final_dca_df[final_dca_df.model == model]["net_benefit"]
            .round(decimals=6)
            .reset_index(drop=True)
            .equals(
                r_benchmark_results_df[r_benchmark_results_df.variable == model]["net_benefit"]
                .round(decimals=6)
                .reset_index(drop=True)
            )
        )
