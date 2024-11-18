"""
Common utility functions for tests.
"""

import pandas as pd
import numpy as np
import lifelines


def create_coxph_model(df_time_to_cancer_dx):
    """
    Create and fit a CoxPHFitter model.
    """
    # pylint: disable=duplicate-code
    cph = lifelines.CoxPHFitter()
    cph.fit(
        df=df_time_to_cancer_dx,
        duration_col="ttcancer",
        event_col="cancer",
        formula="age + famhistory + marker",
    )
    # pylint: enable=duplicate-code

    return cph


def predict_survival(cph, df_time_to_cancer_dx, time=1.5):
    """
    Predict survival using the CoxPHFitter model.
    """
    cph_pred_vals = cph.predict_survival_function(
        df_time_to_cancer_dx[["age", "famhistory", "marker"]], times=[time]
    )
    return 1 - cph_pred_vals.iloc[0, :]


def create_sample_plot_data():
    """
    Create a sample DataFrame for plot testing.
    """
    return pd.DataFrame(
        {
            "model": ["Model1"] * 10 + ["Model2"] * 10,  # 20 data points for 2 models
            "threshold": np.linspace(0, 1, 10).tolist() * 2,  # 10 threshold values for each model
            "net_benefit": np.random.rand(20),  # Random net benefit values
            "net_intervention_avoided": np.random.rand(20),  # Random net interventions avoided
        }
    )


def create_dca_params(df_cancer_dx):
    """
    Create common DCA parameters.
    """
    return {
        "data": df_cancer_dx,
        "outcome": "cancer",
        "modelnames": ["marker"],
        "thresholds": [i / 100 for i in range(36)],
        "harm": {"marker": 0.0333},
        "models_to_prob": ["marker"],
    }
