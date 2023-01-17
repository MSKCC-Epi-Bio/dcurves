
# Load Data for Testing
from dcurves.load_test_data import load_tutorial_coxph_pr_failure18_vals

# Load Tools
import pandas as pd
import numpy as np

# Load Statistics Libraries
import lifelines


def test_tutorial_python_coxph():

    r_coxph_pr_failure18_series = load_tutorial_coxph_pr_failure18_vals()['pr_failure18']

    df_time_to_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
        )
    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')
    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5]
        )
    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]
    # Shows that up to 4 decimal places the values calculated in Python and R are the same
    assert df_time_to_cancer_dx['pr_failure18'].round(decimals=4).equals(r_coxph_pr_failure18_series.round(decimals=4))