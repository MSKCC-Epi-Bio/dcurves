"""
This module houses user-facing simulation data-retrieval functions for end-users.
These data were also used for testing.
"""
import os
import pandas as pd


def _load_data_from_local(filename):
    """
    Load data from the local 'data' directory.
    :param filename: Name of the file to be loaded.
    :return: pd.DataFrame containing the data from the file.
    """
    # Assuming data directory is in the same level as this file
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_dir, filename)
    return pd.read_csv(file_path)


def load_binary_df():
    """
    Load Simulation Data For Binary Endpoints DCA
    :return pd.DataFrame that contains binary data
    """
    return _load_data_from_local("df_binary.csv")


def load_survival_df():
    """
    Load Simulation Data For Survival Endpoints DCA
    :return pd.DataFrame that contains survival data
    """
    return _load_data_from_local("df_surv.csv")


def load_case_control_df():
    """
    Load Simulation Data For Binary Endpoints DCA With User-Specified Outcome
    Prevalence (Case-Control)
    :return pd.DataFrame that contains binary data
    """
    return _load_data_from_local("df_case_control.csv")
