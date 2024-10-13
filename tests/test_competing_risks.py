
from dcurves.risks import _calc_surv_risks
from .load_test_data import load_data
import numpy as np
import pandas as pd  # Assuming the output is a DataFrame
import pytest

from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.api import CumIncidence

def test_regular_surv_case_new_code():
    
    
    data = load_data("df_surv.csv")
    
    
    
    pass


def test_competing_risks():
    
    data = load_data("df_surv.csv")
    
    # dca_results = dca(
    #     data=data,
    #     outcome="cancer",
    #     modelnames=["famhistory", "marker", "cancerpredmarker"],
    #     models_to_prob=["marker"],
    #     thresholds=[i / 100 for i in range(0, 46)],
    #     time_to_outcome_col="ttcancer",
    #     time=1
    # )
    
    model = 'marker' # to convert to prob
    outcome = 'cancer'
    time = 1
    time_to_outcome_col = "ttcancer"


    cph_df = data[[time_to_outcome_col, outcome, model]].copy()
    cph = lifelines.CoxPHFitter()
    cph.fit(cph_df, time_to_outcome_col, outcome)
    cph_df = cph_df.assign(**{time_to_outcome_col: time}).copy()
    predicted_vals = cph.predict_survival_function(cph_df, times=time).values[0]
    
    
    data = data.assign(**{model: predicted_vals}).copy()

    data = data.assign(**{"all": 1}).copy()
    data = data.assign(**{"none": 0}).copy()
    
    print(data)
