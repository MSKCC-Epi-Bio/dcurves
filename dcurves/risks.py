from beartype import beartype
import pandas as pd
import numpy as np
from typing import Optional, Union
import statsmodels.api as sm
import lifelines

@beartype
def _calc_binary_risks(
        data: pd.DataFrame,
        outcome: str,
        model: str
) -> list:
    """
    Calculate Risks For a Model Column for binary DCA.

    Parameters
    ----------
    data : pd.DataFrame
        Initial raw data ideally containing risk scores (scores ranging from 0 to 1), or else predictor/model
        values, and outcome of interest
    outcome : str
        Column name of outcome of interest in risks_df
    model
        Model column name in risks_df

    Returns
    -------
    list[float]
        list of predicted risk scores for a model column
    """
    predicted_vals = sm.formula.glm(outcome + '~' + model, family=sm.families.Binomial(),
                                    data=data).fit().predict()
    return [val for val in predicted_vals]

@beartype
def _calc_surv_risks(
        data: pd.DataFrame,
        outcome: str,
        model: str,
        time: Union[int, float],
        time_to_outcome_col: str
) -> list:
    """
    Calculate Risks For a Model Column for survival DCA.

    Parameters
    ----------
    data : pd.DataFrame
        Initial raw data ideally containing risk scores (scores ranging from 0 to 1), or else predictor/model
        values, and outcome of interest
    outcome : str
        Column name of outcome of interest in risks_df
    model : str
        Model column name in risks_df
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name containing time to outcome values, used in Survival DCA

    Returns
    -------
    list
        list of coxph-predicted risk scores for a model column

    """
    cph_df = data[[time_to_outcome_col, outcome, model]]
    cph = lifelines.CoxPHFitter()
    cph.fit(cph_df, time_to_outcome_col, outcome)
    cph_df[time_to_outcome_col] = [time for i in range(0, len(cph_df))]
    predicted_vals = cph.predict_survival_function(cph_df, times=time).values[0]
    return [val for val in predicted_vals]

@beartype
def _create_risks_df(
        data: pd.DataFrame,
        outcome: str,
        models_to_prob: Optional[list] = None,
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : pd.DataFrame
        Initial raw data ideally containing risk scores (scores ranging from 0 to 1), or else predictor/model
        values, and outcome of interest
    outcome : str
        Column name of outcome of interest in risks_df
    models_to_prob : list[str] (default: None)
        Columns that need to be converted to risk scores from 0 to 1
    time : int or float (default: None)
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str (default: None)
        Column name in data containing time to outcome values, used in Survival DCA

    Returns
    -------
    pd.DataFrame
        Data with non-risk score model columns converted to risk scores between 0 and 1
    """
    if models_to_prob is None:
        pass
    elif time_to_outcome_col is None:
        for model in models_to_prob:
            data[model] = \
                _calc_binary_risks(
                    data=data,
                    outcome=outcome,
                    model=model
                )
    elif time_to_outcome_col is not None:
        for model in models_to_prob:
            data[model] = \
                _calc_surv_risks(
                    data=data,
                    outcome=outcome,
                    model=model,
                    time=time,
                    time_to_outcome_col=time_to_outcome_col
                )

    data['all'] = [1 for i in range(0, len(data.index))]
    data['none'] = [0 for i in range(0, len(data.index))]

    return data

@beartype
def _rectify_model_risk_boundaries(
        risks_df: pd.DataFrame,
        modelnames: list
) -> pd.DataFrame:
    """
    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging from 0 to 1) and outcome of interest
    modelnames : list[str]
        Column names from risks_df that contain model risk scores

    Returns
    -------
    pd.DataFrame
        Data with model column risk scores 0 and 1 changed to 0 - e and 1 + e, respectively, to evaluate as less/
        greater than 0 and 1 thresholds for correct tp/fp evaluations
    """

    machine_epsilon = np.finfo(float).eps
    for modelname in np.append(modelnames, ['all', 'none']):
        risks_df[modelname].replace(to_replace=0, value=0 - machine_epsilon, inplace=True)
        risks_df[modelname].replace(to_replace=1, value=1 + machine_epsilon, inplace=True)

    return risks_df
