"""
Unit tests for survival analysis functions related to risk rate among test positive cases.
These tests validate calculations against benchmark data and ensure correct behavior for
survival data.
"""

# Load Tools
import pandas as pd

# Load Functions To Test/Needed For Testing
from dcurves.dca import _calc_risk_rate_among_test_pos
from dcurves.risks import _create_risks_df
from dcurves.dca import (
    _calc_prevalence,
    _create_initial_df,
    _calc_initial_stats,
    _calc_more_stats,
)
from dcurves.dca import _rectify_model_risk_boundaries

# Load Data for Testing
from .load_test_data import load_r_case2_results
from .load_test_data import load_survival_df
from .load_test_data import load_tutorial_r_stdca_coxph_pr_failure18_test_consequences

from .common_test_utils import create_coxph_model, predict_survival

# Test cases for the survival consequences


def test_case2_surv_risk_rate_among_test_positive():
    """Test the calculation of risk rate among test positive for case 2 survival data."""

    # Load the data and constants
    data = load_survival_df()
    params = {
        "outcome": "cancer",
        "modelnames": ["cancerpredmarker"],
        "time": 1,
        "time_to_outcome_col": "ttcancer",
        "thresholds": [i / 100 for i in range(0, 100)],
    }

    # Create risk and rectified dataframes
    risks_df = _create_risks_df(
        data=data,
        outcome=params["outcome"],
        models_to_prob=None,
        time=params["time"],
        time_to_outcome_col=params["time_to_outcome_col"],
    )

    rectified_risks_df = _rectify_model_risk_boundaries(
        risks_df=risks_df, modelnames=params["modelnames"]
    )

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=params["outcome"],
        prevalence=None,
        time=params["time"],
        time_to_outcome_col=params["time_to_outcome_col"],
    )

    # Create initial dataframe
    _create_initial_df(
        thresholds=params["thresholds"],
        modelnames=params["modelnames"],
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
        harm=None,
    )

    # Helper function for calculating risk rates
    def calculate_risk_rate_for_model(model):
        return _calc_risk_rate_among_test_pos(
            risks_df=rectified_risks_df,
            outcome=params["outcome"],
            model=model,
            thresholds=params["thresholds"],
            time_to_outcome_col=params["time_to_outcome_col"],
            time=params["time"],
        ).reset_index(drop=1)

    # Calculate risk rate for each model
    model_rratp_dict = {
        model: calculate_risk_rate_for_model(model) for model in ["all", "none", "cancerpredmarker"]
    }

    # Load and prepare benchmark results
    r_benchmark_results = load_r_case2_results()

    r_benchmark_results["risk_rate_among_test_pos"] = (
        r_benchmark_results.tp_rate / r_benchmark_results.test_pos_rate
    )
    r_benchmark_results.loc[r_benchmark_results.variable == "none", "risk_rate_among_test_pos"] = (
        float(0)
    )

    # Validate results against benchmarks
    for model in ["all", "none", "cancerpredmarker"]:
        assert (
            model_rratp_dict[model]
            .reset_index(drop=True)
            .round(decimals=5)
            .equals(
                r_benchmark_results[r_benchmark_results.variable == model][
                    "risk_rate_among_test_pos"
                ]
                .reset_index(drop=True)
                .round(decimals=5)
            )
        )


def test_risk_rate_among_test_pos_max_time_less_than_time():
    """Test the calculation when max time is less than the provided time."""
    sample_data = {
        "cancerpredmarker": [0.1, 0.5, 0.8, 0.3, 0.7],
        "cancer": [1, 0, 1, 0, 0],
        "ttcancer": [1, 2, 3, 4, 4.5],  # All times are below 5
    }
    risks_df = pd.DataFrame(sample_data)
    outcome = "cancer"
    model = "cancerpredmarker"
    thresholds = [0.2, 0.4, 0.6]
    time = 5  # This is greater than all times in our sample data
    time_to_outcome_col = "ttcancer"

    result = _calc_risk_rate_among_test_pos(
        risks_df, outcome, model, thresholds, time, time_to_outcome_col
    )

    expected_result = [None, None, None]
    assert result == expected_result, f"Expected {expected_result} but got {result}"


def test_case2_all_stats():
    """Test all statistics for case 2 data."""
    r_benchmark_results = load_r_case2_results()

    outcome = "cancer"
    time = 1
    time_to_outcome_col = "ttcancer"

    risks_df = _create_risks_df(
        data=load_survival_df(),
        outcome=outcome,
        models_to_prob=None,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    rectified_risks_df = _rectify_model_risk_boundaries(
        risks_df=risks_df, modelnames=["cancerpredmarker"]
    )

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=outcome,
        prevalence=None,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    initial_df = _create_initial_df(
        thresholds=[i / 100 for i in range(0, 100)],
        modelnames=["cancerpredmarker"],
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
        harm=None,
    )

    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=[i / 100 for i in range(0, 100)],
        outcome=outcome,
        prevalence_value=prevalence_value,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    final_dca_df = _calc_more_stats(initial_stats_df=initial_stats_df)

    for model in ["all", "none", "cancerpredmarker"]:
        for stat in ["tp_rate", "fp_rate", "net_benefit"]:
            round_dec_num = 6
            p_tp_rate_series = (
                final_dca_df[final_dca_df.model == model][stat]
                .reset_index(drop=True)
                .round(decimals=round_dec_num)
            )

            r_tp_rate_series = (
                r_benchmark_results[r_benchmark_results.variable == model][stat]
                .reset_index(drop=1)
                .round(decimals=round_dec_num)
            )

            assert p_tp_rate_series.equals(r_tp_rate_series)


def test_tut_pr_failure18_tp_rate():
    """Test tutorial pr_failure18 tp rate calculation."""
    # Load the data
    df_time_to_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main"
        "/data/df_time_to_cancer_dx.csv"
    )

    # Create and fit the CoxPH model
    # pylint: disable=duplicate-code
    cph = create_coxph_model(df_time_to_cancer_dx)
    # pylint: enable=duplicate-code

    # Predict survival and add to dataframe
    df_time_to_cancer_dx["pr_failure18"] = predict_survival(cph, df_time_to_cancer_dx, time=1.5)

    # Set up parameters for DCA calculation
    outcome = "cancer"
    time = 1.5
    time_to_outcome_col = "ttcancer"

    # Create risks dataframe
    risks_df = _create_risks_df(
        data=df_time_to_cancer_dx,
        outcome=outcome,
        models_to_prob=None,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    # Calculate prevalence
    prevalence_value = _calc_prevalence(
        risks_df=risks_df,
        outcome=outcome,
        prevalence=None,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    # Create initial dataframe
    initial_df = _create_initial_df(
        thresholds=[i / 100 for i in range(0, 51)],
        modelnames=["pr_failure18"],
        input_df_rownum=len(risks_df.index),
        prevalence_value=prevalence_value,
        harm=None,
    )

    # Calculate initial stats
    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=risks_df,
        thresholds=[i / 100 for i in range(0, 51)],
        outcome=outcome,
        prevalence_value=prevalence_value,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    # Calculate final stats
    final_dca_df = _calc_more_stats(initial_stats_df=initial_stats_df)

    # Load benchmark data
    benchmark_df = load_tutorial_r_stdca_coxph_pr_failure18_test_consequences()

    # Compare results
    for stat in ["tp_rate"]:
        p_series = (
            final_dca_df[final_dca_df.model == "pr_failure18"][stat]
            .reset_index(drop=True)
            .round(decimals=5)
        )

        r_series = benchmark_df[stat].reset_index(drop=True).round(decimals=5)

        assert p_series.equals(r_series), f"Mismatch in {stat} calculation"
