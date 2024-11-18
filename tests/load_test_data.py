"""
This module provides functions to load various test data files for the dcurves package.

It includes functions to load binary, survival, and case-control data,
as well as benchmark data for quality control on local functions.
"""

import os
import pandas as pd

# File descriptions dictionary
DESCRIPTIONS = {
    "df_binary.csv": "Simulation Data For Binary Endpoints DCA",
    "df_surv.csv": "Simulation Data For Survival Endpoints DCA",
    "df_case_control.csv": (
        "Simulation Data For Binary Endpoints DCA " "With User-Specified Outcome Prevalence"
    ),
    "shishir_simdata.csv": "Simulation survival data from Shishir Rao, Oxford Doctoral Student",
    "r_case1_results.csv": "Binary, cancer ~ famhistory",
    "r_case2_results.csv": "Survival, Cancer ~ Cancerpredmarker",
    "r_case3_results.csv": "Binary, Cancer ~ marker",
    "dca_tut_bin_int_marker_risks.csv": "Tutorial binary marker risks",
    "dca_tut_coxph_pr_failure18_vals.csv": "Tutorial CoxPH failure values",
    "surv_risk_test_df.csv": "Risk scores for marker convert_to_risk in survival case",
    "dca_tut_r_stdca_coxph_df.csv": "Tutorial R stdca CoxPH dataframe",
    "dca_tut_r_stdca_coxph_pr_failure18_test_consequences.csv": (
        "Tutorial R stdca CoxPH failure consequences"
    ),
    "r_simple_surv_dca_result_df.csv": "R survival DCA results for simple testing",
    "r_simple_surv_tpfp_calc_df.csv": "Columns for TP/FP rate calculation from R survival DCA",
    "r_simple_binary_dca_result_df.csv": "R simple binary DCA results",
    "r_dca_famhistory.csv": "R DCA family history data",
    "df_cancer_dx.csv": "Cancer diagnosis data from dca-tutorial GitHub",
    "df_cancer_dx2.csv": "Cancer diagnosis data with model predictions",
}


def load_data(filename):
    """
    Load a data file based on the provided filename.

    Args:
        filename (str): Name of the file to be loaded.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If the filename is unknown.
    """
    if filename not in DESCRIPTIONS:
        raise ValueError(f"Unknown filename: {filename}")

    file_path = os.path.join(os.path.dirname(__file__), f"benchmark_data/{filename}")
    return pd.read_csv(file_path, encoding="latin-1")


def load_binary_df():
    """Load Simulation Data For Binary Endpoints DCA."""
    return load_data("df_binary.csv")


def load_survival_df():
    """Load Simulation Data For Survival Endpoints DCA."""
    return load_data("df_surv.csv")


def load_case_control_df():
    """Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome Prevalence."""
    return load_data("df_case_control.csv")


def load_shishir_simdata():
    """Load simulation survival data from Shishir Rao, Oxford Doctoral Student."""
    return load_data("shishir_simdata.csv")


def load_r_case1_results():
    """
    Load R results for Binary case: cancer ~ famhistory.

    Analogous Python Settings:
    data = load_binary_df()
    thresholds = [i/100 for i in range(0, 100)]
    outcome = 'cancer'
    modelnames = ['famhistory']
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    prevalence = None
    harm = None
    """
    return load_data("r_case1_results.csv")


def load_r_case2_results():
    """
    Load R results for Survival case: Cancer ~ Cancerpredmarker.

    Analogous Python Settings:
    data = load_surv_df()
    thresholds = [i/100 for i in range(0, 100)]
    outcome = 'cancer'
    modelnames = ['cancerpredmarker']
    models_to_prob = None
    time = 1
    time_to_outcome_col = 'ttcancer'
    prevalence = None
    harm = None
    """
    return load_data("r_case2_results.csv")


def load_r_case3_results():
    """
    Load R results for Binary case: Cancer ~ marker.

    Analogous Python Settings:
    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main"
        "/data/df_cancer_dx.csv"
    )
    data = df_cancer_dx
    thresholds = [i/100 for i in range(0, 36)]
    outcome = 'cancer'
    modelnames = ['marker']
    models_to_prob = ['marker']
    time = 1
    time_to_outcome_col = 'ttcancer'
    prevalence = None
    harm = {'marker': 0.0333}
    """
    return load_data("r_case3_results.csv")


def load_tutorial_bin_marker_risks_list():
    """Load tutorial binary marker risks list."""
    return load_data("dca_tut_bin_int_marker_risks.csv")


def load_tutorial_coxph_pr_failure18_vals():
    """Load tutorial CoxPH failure values."""
    return load_data("dca_tut_coxph_pr_failure18_vals.csv")


def load_test_surv_risk_test_df():
    """
    Load risk scores for marker convert_to_risk in survival case.

    Settings:
    outcome='cancer'
    models=['marker']
    data=load_survival_df()
    time=2
    time_to_outcome_col='ttcancer'
    """
    return load_data("surv_risk_test_df.csv")


def load_tutorial_r_stdca_coxph_df():
    """Load tutorial R stdca CoxPH dataframe."""
    return load_data("dca_tut_r_stdca_coxph_df.csv")


def load_tutorial_r_stdca_coxph_pr_failure18_test_consequences():
    """Load tutorial R stdca CoxPH failure consequences."""
    return load_data("dca_tut_r_stdca_coxph_pr_failure18_test_consequences.csv")


def load_r_simple_surv_dca_result_df():
    """
    Load R survival DCA results for simple testing.

    Settings:
    outcome='cancer'
    models=['famhistory']
    thresholds=[i/100 for i in range(0, 46)]
    time_to_outcome_col='ttcancer'
    time=1.5
    """
    return load_data("r_simple_surv_dca_result_df.csv")


def load_r_simple_surv_tpfp_calc_df():
    """
    Load columns for TP/FP rate calculation from R survival DCA.

    Settings:
    outcome='cancer'
    models=['famhistory']
    thresholds=[i/100 for i in range(0, 46)]
    time_to_outcome_col='ttcancer'
    time=1.5
    """
    return load_data("r_simple_surv_tpfp_calc_df.csv")


def load_r_simple_binary_dca_result_df():
    """Load R simple binary DCA results."""
    return load_data("r_simple_binary_dca_result_df.csv")


def load_r_dca_famhistory():
    """Load R DCA family history data."""
    return load_data("r_dca_famhistory.csv")


def load_r_df_cancer_dx():
    """
    Load cancer diagnosis data from dca-tutorial GitHub.

    This data can also be obtained from:
    https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv
    """
    return load_data("df_cancer_dx.csv")


def load_r_df_cancer_dx2():
    """Load cancer diagnosis data with model predictions."""
    return load_data("df_cancer_dx2.csv")
