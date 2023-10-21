import os
import pathlib

import pandas as pd

def load_data(filename):
    """
    Unified function to load a data file based on the provided filename.
    :param filename: Name of the file to be loaded.
    :return: pd.DataFrame with the loaded data.
    """

    # Define a mapping of filenames to their corresponding descriptions
    descriptions = {
        'df_binary.csv': 'Load Simulation Data For Binary Endpoints DCA',
        'df_surv.csv': 'Load Simulation Data For Survival Endpoints DCA',
        'df_case_control.csv': 'Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome Prevalence (Case-Control)',
        'shishir_simdata.csv': 'Load simulation survival data from Shishir Rao, Oxford Doctoral Student, and check that my DCA functions work correctly',
        'r_case1_results.csv': 'Binary, cancer ~ famhistory',
        'r_case2_results.csv': 'Survival, Cancer ~ Cancerpredmarker',
        'r_case3_results.csv': 'Binary, Cancer ~ marker',
        'dca_tut_bin_int_marker_risks.csv': '',
        'dca_tut_coxph_pr_failure18_vals.csv': '',
        'surv_risk_test_df.csv': 'These are risk scores calculated for marker convert_to_risk in survival case',
        'dca_tut_r_stdca_coxph_df.csv': '',
        'dca_tut_r_stdca_coxph_pr_failure18_test_consequences.csv': '',
        'r_simple_surv_dca_result_df.csv': "Dataframe containing results from r survival dca case for simple testing",
        'r_simple_surv_tpfp_calc_df.csv': "Dataframe containing columns used to calculate tp/fp rate from r survival dca case for simple testing",
        'r_simple_binary_dca_result_df.csv': '',
        'r_dca_famhistory.csv': '',
        'df_cancer_dx.csv': 'Load data that can also be gotten from this link: "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"',
        'df_cancer_dx2.csv': 'Load data that is same as df_cancer_dx but with predictions from model as in dca-tutorial'
    }

    # Check if the filename is in the mapping
    if filename not in descriptions:
        raise ValueError(f"Unknown filename: {filename}")

    # Load the file
    file_path = os.path.join(os.path.dirname(__file__), f'benchmark_data/{filename}')
    with open(file_path, 'r') as f:
        return pd.read_csv(f, encoding='latin-1')


def load_data_file(filename):
    file_path = os.path.join(os.path.dirname(__file__), f'benchmark_data/{filename}')
    with open(file_path, 'r') as f:
        return pd.read_csv(f, encoding='latin-1')

def load_binary_df():
    '''
    Load Simulation Data For Binary Endpoints DCA
    :return pd.DataFrame that contains binary data
    '''
    return load_data_file('df_binary.csv')

def load_survival_df():
    """
    Load Simulation Data For Survival Endpoints DCA
    :return pd.DataFrame that contains survival data
    """
    return load_data_file('df_surv.csv')

def load_case_control_df():
    """
    Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome Prevalence (Case-Control)
    :return pd.DataFrame that contains binary data
    """
    return load_data_file('df_case_control.csv')


def load_shishir_simdata():
    """
    Load simulation survival data from Shishir Rao, Oxford Doctoral Student, and check that my DCA functions work
    correctly
    Returns
    -------
    pd.DataFrame
    """
    return load_data_file('shishir_simdata.csv')

# Load Benchmarking Data For QC On Local Functions

# Case 1 R Results: Simple Binary case

def load_r_case1_results():
    '''
    Title: Binary, cancer ~ famhistory

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
    '''
    return load_data_file('r_case1_results.csv')

# Case 2 R Results: Simple survival Case

def load_r_case2_results():
    '''

    Title: Survival, Cancer ~ Cancerpredmarker

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
    '''

    return load_data_file('r_case2_results.csv')


def load_r_case3_results():
    '''
    Title: Binary, Cancer ~ marker

    Analogous Python Settings:

    df_cancer_dx = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')

    data = df_cancer_dx
    thresholds = [i/100 for i in range(0, 36)]
    outcome = 'cancer'
    modelnames = ['marker']
    models_to_prob = ['marker']
    time = 1
    time_to_outcome_col = 'ttcancer'
    prevalence = None
    harm = {'marker': 0.0333}
    '''
    return load_data_file('r_case3_results.csv')

# def load_r_case4_results():
#     """
#     Title: Survival,
#     Returns
#     -------
#
#     """
#
#     pass

# TUTORIAL BENCHMARKING

def load_tutorial_bin_marker_risks_list():
    return load_data_file('dca_tut_bin_int_marker_risks.csv')

def load_tutorial_coxph_pr_failure18_vals():
    return load_data_file('dca_tut_coxph_pr_failure18_vals.csv')

def load_test_surv_risk_test_df():
    """
     These are risk scores calculated for marker convert_to_risk in survival case:
     outcome='cancer'
     models=['marker']
     data=load_survival_df() (from above)
     time=2,
     time_to_outcome_col='ttcancer'
    """
    return load_data_file('surv_risk_test_df.csv')

def load_tutorial_r_stdca_coxph_df():
    return load_data_file('dca_tut_r_stdca_coxph_df.csv')

def load_tutorial_r_stdca_coxph_pr_failure18_test_consequences():
    return load_data_file('dca_tut_r_stdca_coxph_pr_failure18_test_consequences.csv')

def load_r_simple_surv_dca_result_df():
    """
    Dataframe containing results from r survival dca case for simple testing
    outcome='cancer'
    models=['famhistory']
    thresholds=[i/100 for i in range(0, 46)]
    time_to_outcome_col='ttcancer'
    time=1.5
    :return:
    """
    return load_data_file('r_simple_surv_dca_result_df.csv')

def load_r_simple_surv_tpfp_calc_df():
    """
    Dataframe containing columns used to calculate tp/fp rate from r survival dca case for simple testing
    outcome='cancer'
    models=['famhistory']
    thresholds=[i/100 for i in range(0, 46)]
    time_to_outcome_col='ttcancer'
    time=1.5
    :return:
    """
    return load_data_file('r_simple_surv_tpfp_calc_df.csv')

def load_r_simple_binary_dca_result_df():
    return load_data_file('r_simple_binary_dca_result_df.csv')

# DCA Tutorial Benchmarking files

def load_r_dca_famhistory():
    return load_data_file('r_dca_famhistory.csv')

def load_r_df_cancer_dx():
    '''
    Load data that can also be gotten from this link:
    "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
    '''
    return load_data_file('df_cancer_dx.csv')

def load_r_df_cancer_dx2():
    ''''
    Load data that is same as df_cancer_dx but with predictions from model as in dca-tutorial
    '''
    return load_data_file('df_cancer_dx2.csv')