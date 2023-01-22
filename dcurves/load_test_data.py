import pandas as pd
import pkg_resources
# import importlib_resources

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





