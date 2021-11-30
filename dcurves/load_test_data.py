import pandas as pd
import pkg_resources

def load_binary_df():
    """Return a dataframe containing the binary data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/df_binary.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_survival_df():
    """Return a dataframe containing the survival data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    stream = pkg_resources.resource_stream(__name__, 'data/df_surv.csv')
    return pd.read_csv(stream, encoding='latin-1')

def load_case_control_df():
    """Return a dataframe containing the case-control data
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()

    stream = pkg_resources.resource_stream(__name__, 'data/df_case_control.csv')
    return pd.read_csv(stream, encoding='latin-1')

