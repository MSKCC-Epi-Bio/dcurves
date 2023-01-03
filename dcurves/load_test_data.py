import pandas as pd
import pkg_resources
from os import path

#useful directories
# root_test_dir = path.dirname(path.realpath(__file__))
# resources_dir = path.join(root_test_dir, 'data')

# path.join(resources_dir, 'df_surv.csv')
# path.join(resources_dir, 'df_case_control.csv')

# simdata_dir = path.join(resources_dir, 'sim_data')
# r_results_dir = path.join(resources_dir, 'r_results')

# Note: after speaking with Dan, can host data online on dropbox/s3/googledrive and pull using python script

# Do try/except on pulling data if it's not in expected location

def load_binary_df():
    """Return a dataframe containing a binary simulation dataset

    Parameters
    ----------
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    # with pkg_resources.resource_stream(__name__, 'data/df_binary.csv') as stream:
    #     # stream path:
    #     # '/Users/ShaunPorwal/Documents/GitHub/python_packages/dcurves/dcurves/data/df_binary.csv'
    #
    #     data = pd.read_csv(stream, encoding='latin-

    # data_path = path.join(resources_dir, 'df_binary.csv')
    # data = pd.read_csv(data_path)

    stream = pkg_resources.resource_stream(__name__, 'data/df_binary.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_survival_df():
    """Return a dataframe containing a simulation survival data
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    # with pkg_resources.resource_stream(__name__, 'data/df_surv.csv') as stream:
    #     data = pd.read_csv(stream, encoding='latin-1')
    stream = pkg_resources.resource_stream(__name__, 'data/df_surv.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_case_control_df():
    """Return a dataframe containing a case-control dataset
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    # with pkg_resources.resource_stream(__name__, 'data/df_case_control.csv') as stream:
    #     data = pd.read_csv(stream, encoding='latin-1')

    stream = pkg_resources.resource_stream(__name__, 'data/df_case_control.csv')
    return pd.read_csv(stream, encoding='latin-1')

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
     outcome = 'cancer'
     model = 'marker'
     data = load_survival_df() (from above)
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