"""Shared fixtures for dcurves tests.

All commonly used test data is loaded once and cached at module scope.
"""

import pytest
from .load_test_data import (
    load_binary_df,
    load_survival_df,
    load_case_control_df,
    load_r_case1_results,
    load_r_case2_results,
    load_r_case3_results,
    load_shishir_simdata,
    load_tutorial_bin_marker_risks_list,
    load_tutorial_coxph_pr_failure18_vals,
    load_test_surv_risk_test_df,
    load_tutorial_r_stdca_coxph_df,
    load_tutorial_r_stdca_coxph_pr_failure18_test_consequences,
    load_r_simple_surv_dca_result_df,
    load_r_simple_surv_tpfp_calc_df,
    load_r_simple_binary_dca_result_df,
    load_r_dca_famhistory,
    load_r_df_cancer_dx,
    load_r_df_cancer_dx2,
)


# --- Primary Test Datasets ---


@pytest.fixture(scope="session")
def df_binary():
    """Binary endpoint simulation data (loaded once per test session)."""
    return load_binary_df()


@pytest.fixture(scope="session")
def df_surv():
    """Survival endpoint simulation data (loaded once per test session)."""
    return load_survival_df()


@pytest.fixture(scope="session")
def df_case_control():
    """Case-control simulation data (loaded once per test session)."""
    return load_case_control_df()


# --- R Benchmark Results ---


@pytest.fixture(scope="session")
def r_case1_results():
    """R benchmark results for binary case: cancer ~ famhistory."""
    return load_r_case1_results()


@pytest.fixture(scope="session")
def r_case2_results():
    """R benchmark results for survival case: Cancer ~ Cancerpredmarker."""
    return load_r_case2_results()


@pytest.fixture(scope="session")
def r_case3_results():
    """R benchmark results for binary case: Cancer ~ marker."""
    return load_r_case3_results()


# --- Other Test Data ---


@pytest.fixture(scope="session")
def df_shishir_simdata():
    """Shishir Rao survival simulation data."""
    return load_shishir_simdata()


@pytest.fixture(scope="session")
def tutorial_bin_marker_risks():
    """Tutorial binary marker risks."""
    return load_tutorial_bin_marker_risks_list()


@pytest.fixture(scope="session")
def tutorial_coxph_pr_failure18():
    """Tutorial CoxPH failure values."""
    return load_tutorial_coxph_pr_failure18_vals()


@pytest.fixture(scope="session")
def surv_risk_test_df():
    """Risk scores for marker convert_to_risk in survival case."""
    return load_test_surv_risk_test_df()


@pytest.fixture(scope="session")
def tutorial_r_stdca_coxph_df():
    """Tutorial R stdca CoxPH dataframe."""
    return load_tutorial_r_stdca_coxph_df()


@pytest.fixture(scope="session")
def tutorial_r_stdca_coxph_consequences():
    """Tutorial R stdca CoxPH failure consequences."""
    return load_tutorial_r_stdca_coxph_pr_failure18_test_consequences()


@pytest.fixture(scope="session")
def r_simple_surv_dca_results():
    """R survival DCA results for simple testing."""
    return load_r_simple_surv_dca_result_df()


@pytest.fixture(scope="session")
def r_simple_surv_tpfp_calc():
    """TP/FP rate calculation columns from R survival DCA."""
    return load_r_simple_surv_tpfp_calc_df()


@pytest.fixture(scope="session")
def r_simple_binary_dca_results():
    """R simple binary DCA results."""
    return load_r_simple_binary_dca_result_df()


@pytest.fixture(scope="session")
def r_dca_famhistory():
    """R DCA family history data."""
    return load_r_dca_famhistory()


@pytest.fixture(scope="session")
def df_cancer_dx():
    """Cancer diagnosis data from dca-tutorial GitHub."""
    return load_r_df_cancer_dx()


@pytest.fixture(scope="session")
def df_cancer_dx2():
    """Cancer diagnosis data with model predictions."""
    return load_r_df_cancer_dx2()

