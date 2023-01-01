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
):
    predicted_vals = sm.formula.glm(outcome + '~' + model, family=sm.families.Binomial(),
                                    data=data).fit().predict()
    # return [(1 - val) for val in predicted_vals]
    return [val for val in predicted_vals]

@beartype
def _calc_surv_risks(
        data: pd.DataFrame,
        outcome: str,
        model: str,
        time: Union[int, float],
        time_to_outcome_col: str
) -> list:
    cph_df = data[[time_to_outcome_col, outcome, model]]
    cph = lifelines.CoxPHFitter()
    cph.fit(cph_df, time_to_outcome_col, outcome)
    cph_df[time_to_outcome_col] = [time for i in range(0, len(cph_df))]
    return cph.predict_survival_function(cph_df, times=time).values[0]

@beartype
def _create_risks_df(
        data: pd.DataFrame,
        outcome: str,
        models_to_prob: Optional[list] = None,
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None
) -> pd.DataFrame:
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

    machine_epsilon = np.finfo(float).eps

    data['all'] = [1 - machine_epsilon for i in range(0, len(data.index))]
    data['none'] = [0 + machine_epsilon for i in range(0, len(data.index))]

    return data

