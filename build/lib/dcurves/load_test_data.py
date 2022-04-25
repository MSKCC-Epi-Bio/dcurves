import pandas as pd
import pkg_resources
from os import path

#useful directories
root_test_dir = path.dirname(path.realpath(__file__))
resources_dir = path.join(root_test_dir, 'data')

path.join(resources_dir, 'df_surv.csv')
path.join(resources_dir, 'df_case_control.csv')

# simdata_dir = path.join(resources_dir, 'sim_data')
# r_results_dir = path.join(resources_dir, 'r_results')

def load_binary_df():
    """Return a dataframe containing the binary data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    # with pkg_resources.resource_stream(__name__, 'data/df_binary.csv') as stream:
    #     # stream path:
    #     # '/Users/ShaunPorwal/Documents/GitHub/python_packages/dcurves/dcurves/data/df_binary.csv'
    #
    #     data = pd.read_csv(stream, encoding='latin-

    data_path = path.join(resources_dir, 'df_binary.csv')
    data = pd.read_csv(data_path)
    return data
    # return print(stream)

def load_survival_df():
    """Return a dataframe containing the survival data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    # with pkg_resources.resource_stream(__name__, 'data/df_surv.csv') as stream:
    #     data = pd.read_csv(stream, encoding='latin-1')

    data_path = path.join(resources_dir, 'df_surv.csv')
    data = pd.read_csv(data_path)
    return data

def load_case_control_df():
    """Return a dataframe containing the case-control data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    # with pkg_resources.resource_stream(__name__, 'data/df_case_control.csv') as stream:
    #     data = pd.read_csv(stream, encoding='latin-1')

    data_path = path.join(resources_dir, 'df_case_control.csv')
    data = pd.read_csv(data_path)
    return data
