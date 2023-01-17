

import pandas as pd
from os import path
# import load_test_data

#useful directories
root_test_dir = path.dirname(path.realpath(__file__))
resources_dir = path.join(root_test_dir, 'resources')
simdata_dir = path.join(resources_dir, 'sim_data')

r_results_dir = path.join(resources_dir, 'r_results')

# def load_binary_data():
#
#     csv_path = path.join(simdata_dir, 'df_binary.csv')
#     return pd.read_csv(csv_path)
#
# def load_survival_data():
#     csv_path = path.join(simdata_dir, 'df_surv.csv')
#     return pd.read_csv(csv_path)
#
# def load_case_control_data():
#     csv_path = path.join(simdata_dir, 'df_case_control.csv')
#     return pd.read_csv(csv_path)


# def load_r_results(analysis_name):
#     """Loads the net_benefit and interventions_avoided results for an R analysis
#
#     Parameters
#     ----------
#     analysis_name : str
#         the name to give to the analysis (this will be the name used by the folder
#         created where the output csv's are saved)
#
#     Returns
#     -------
#     tuple(pd.DataFrame, pd.DataFrame)
#         `net benefit` and `interventions avoided` dataframes
#     """
#     analysis_dir = path.join(r_results_dir, analysis_name)
#     r_nb = pd.read_csv(path.join(analysis_dir, 'net_benefit.csv'))
#     r_ia = pd.read_csv(path.join(analysis_dir, 'interventions_avoided.csv'))
#     return r_nb, r_ia



# ---------------------------------------




# import pandas as pd
# import pkg_resources
#
# def load_binary_df():
#     """Return a dataframe containing the binary data
#     """
#
#     # This is a stream-like object. If you want the actual info, call
#     # stream.read()
#     stream = pkg_resources.resource_stream(__name__, 'resources/sim_data/df_binary.csv')
#     return pd.read_csv(stream, encoding='latin-1')
#
# def load_survival_df():
#     """Return a dataframe containing the survival data
#     """
#
#     # This is a stream-like object. If you want the actual info, call
#     # stream.read()
#
#     stream = pkg_resources.resource_stream(__name__, 'resources/sim_data/df_surv.csv')
#     return pd.read_csv(stream, encoding='latin-1')
#
# def load_case_control_df():
#     """Return a dataframe containing the case-control data
#     """
#
#     # This is a stream-like object. If you want the actual info, call
#     # stream.read()
#
#     stream = pkg_resources.resource_stream(__name__, 'resources/sim_data/df_case_control.csv')
#     return pd.read_csv(stream, encoding='latin-1')


