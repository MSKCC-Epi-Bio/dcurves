"""
This module houses the functions used to calculate prevalence. It is
kept in a separate module from dca.py to disperse the code and make
clear its individual dependencies.
"""

from typing import Optional, Union
import pandas as pd
import lifelines


def _calc_prevalence(
    risks_df: pd.DataFrame,
    outcome: str,
    prevalence: Optional[Union[int, float]] = None,
    time: Optional[Union[int, float]] = None,
    time_to_outcome_col: Optional[str] = None,
) -> float:
    """
    Calculate prevalence value when not supplied for binary and survival DCA cases,
    and set prevalence value for binary case when supplied (case control). Case
    control prevalence is not value for survival cases.

    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging from
        0 to 1) and outcome of interest
    outcome : str
        Column name of outcome of interest in risks_df
    prevalence : int or float
        Value that indicates the prevalence among the population, only to be
        specified in case-control situations
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name of time of interest in risks_df

    Returns
    -------
    float
        Either calculated prevalence or inputted prevalence depending on
        whether input prevalence was supplied
    """

    if time_to_outcome_col is None:
        if prevalence is None:
            prevalence = sum(risks_df[outcome]) / len(risks_df[outcome])
    else:
        if prevalence is not None:
            raise ValueError("In survival outcomes, prevalence should not be supplied")

        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(risks_df[time_to_outcome_col], risks_df[outcome] * 1)
        prevalence = 1 - kmf.survival_function_at_times(time).iloc[0]
    return float(prevalence)
