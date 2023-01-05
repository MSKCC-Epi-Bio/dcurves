import pandas as pd
import pkg_resources
from os import path

#useful directories
# root_test_dir = path.dirname(path.realpath(__file__))
# resources_dir = path.join(root_test_dir, 'data')

# simdata_dir = path.join(resources_dir, 'sim_data')
# r_results_dir = path.join(resources_dir, 'r_results')

# Note: after speaking with Dan, can host data online on dropbox/s3/googledrive and pull using python script

# Load Simulation Data
def load_binary_df():
    '''
    Load Simulation Data For Binary Endpoints DCA
    :return pd.DataFrame that contains binary data
    '''

    stream = pkg_resources.resource_stream(__name__, 'data/df_binary.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_survival_df():
    """
    Load Simulation Data For Survival Endpoints DCA
    :return pd.DataFrame that contains survival data
    """
    stream = pkg_resources.resource_stream(__name__, 'data/df_surv.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_case_control_df():
    """
    Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome Prevalence (Case-Control)
    :return pd.DataFrame that contains binary data
    """
    stream = pkg_resources.resource_stream(__name__, 'data/df_case_control.csv')
    return pd.read_csv(stream, encoding='latin-1')

# Load Benchmarking Data For QC On Local Functions

def load_tutorial_bin_interventions_df():
    stream = pkg_resources.resource_stream(__name__, 'data/dca_tut_bin_int_df.csv')
    return pd.read_csv(stream, encoding='latin-1')
def load_tutorial_bin_marker_risks_list():
    stream = pkg_resources.resource_stream(__name__, 'data/dca_tut_bin_int_marker_risks.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_tutorial_coxph_pr_failure18_vals():
    stream = pkg_resources.resource_stream(__name__, 'data/dca_tut_coxph_pr_failure18_vals.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_test_surv_risk_test_df():
    """
     These are risk scores calculated for marker convert_to_risk in survival case:
     outcome='cancer'
     models=['marker']
     data=load_survival_df() (from above)
     time=2,
     time_to_outcome_col='ttcancer'
    """
    stream = pkg_resources.resource_stream(__name__, 'data/surv_risk_test_df.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_tutorial_r_stdca_coxph_df():
    stream = pkg_resources.resource_stream(__name__, 'data/dca_tut_r_stdca_coxph_df.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_tutorial_r_stdca_coxph_pr_failure18_test_consequences():
    stream = pkg_resources.resource_stream(__name__, 'data/dca_tut_r_stdca_coxph_pr_failure18_test_consequences.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_r_simple_surv_dca_result_df():
    """
    Dataframe containing results from r survival dca case for simple testing
    outcome='cancer'
    models=['famhistory']
    thresholds=np.arange(0, 1.0, 0.01)
    time_to_outcome_col='ttcancer'
    time=1.5
    :return:
    """
    stream = pkg_resources.resource_stream(__name__, 'data/r_simple_surv_dca_result_df.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_r_simple_surv_tpfp_calc_df():
    """
    Dataframe containing columns used to calculate tp/fp rate from r survival dca case for simple testing
    outcome='cancer'
    models=['famhistory']
    thresholds=np.arange(0, 1.0, 0.01)
    time_to_outcome_col='ttcancer'
    time=1.5
    :return:
    """
    stream = pkg_resources.resource_stream(__name__, 'data/r_simple_surv_tpfp_calc_df.csv')
    return pd.read_csv(stream, encoding='latin-1')
