import pandas as pd
import pkg_resources

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

# Case 1 R Results: Simple Binary case

def load_r_case1_results():
    '''
    Title: Binary, cancer ~ famhistory

    Analogous Python Settings:

    data = load_binary_df()
    thresholds = np.arange(0, 1.0, 0.01)
    outcome = 'cancer'
    modelnames = ['famhistory']
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    prevalence = None
    harm = None
    '''
    stream = pkg_resources.resource_stream(__name__, 'data/r_case1_results.csv')
    return pd.read_csv(stream, encoding='latin-1')

# Case 2 R Results:

def load_r_case2_results():
    '''

    Title: Survival, Cancer ~ Cancerpredmarker

    Analogous Python Settings:

    data = load_surv_df()
    thresholds = np.arange(0, 1.0, 0.01)
    outcome = 'cancer'
    modelnames = ['cancerpredmarker']
    models_to_prob = None
    time = 1
    time_to_outcome_col = 'ttcancer'
    prevalence = None
    harm = None
    '''

    stream = pkg_resources.resource_stream(__name__, 'data/r_case2_results.csv')
    return pd.read_csv(stream, encoding='latin-1')

# TUTORIAL BENCHMARKING

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

def load_r_simple_binary_dca_result_df():
    stream = pkg_resources.resource_stream(__name__, 'data/r_simple_binary_dca_result_df.csv')
    return pd.read_csv(stream, encoding='latin-1')


# DCA Tutorial Benchmarking files

def load_r_dca_famhistory():
    stream = pkg_resources.resource_stream(__name__, 'data/r_dca_famhistory.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_r_df_cancer_dx():
    '''
    Load data that can also be gotten from this link:
    "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
    '''
    stream = pkg_resources.resource_stream(__name__, 'data/df_cancer_dx.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_r_df_cancer_dx2():
    ''''
    Load data that is same as df_cancer_dx but with predictions from model as in dca-tutorial
    '''
    stream = pkg_resources.resource_stream(__name__, 'data/df_cancer_dx2.csv')
    return pd.read_csv(stream, encoding='latin-1')





