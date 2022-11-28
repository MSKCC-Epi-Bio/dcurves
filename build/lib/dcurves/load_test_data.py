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

def load_r_bin_dca_result_1():
    """For binary case, return a dataframe containing calculated net benefit values for default settings
    model_frame: load_binary_df() from load_test_data
    outcome: cancer
    predictor(s): cancerpredmarker, marker
    thresholds: 0.01 to 0.99
    harm: None
    probabilities = [False, True] # False for cancerpredmarker, True for marker

    :return:
    """

    stream = pkg_resources.resource_stream(__name__, 'data/r_bin_dca_result_1.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_r_bctr_result_1():

    stream = pkg_resources.resource_stream(__name__, 'data/r_bin_dca_ctr_result_1.csv')
    return pd.read_csv(stream, encoding='latin-1')






