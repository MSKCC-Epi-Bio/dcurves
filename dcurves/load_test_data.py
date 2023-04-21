"""
This module houses user-facing simulation data-retrieval functions for end-users.
These data were also used for testing.
"""
from importlib.resources import files
import pandas as pd
from dcurves import data

def load_binary_df():
    """
    Load Simulation Data For Binary Endpoints DCA
    :return pd.DataFrame that contains binary data
    """
    file_path = files(data).joinpath("df_binary.csv")
    return pd.read_csv(file_path)

def load_survival_df():
    """
    Load Simulation Data For Survival Endpoints DCA
    :return pd.DataFrame that contains survival data
    """
    file_path = files(data).joinpath("df_surv.csv")
    return pd.read_csv(file_path)


def load_case_control_df():
    """
    Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome
    Prevalence (Case-Control)
    :return pd.DataFrame that contains binary data
    """
    file_path = files(data).joinpath("df_case_control.csv")
    return pd.read_csv(file_path)
