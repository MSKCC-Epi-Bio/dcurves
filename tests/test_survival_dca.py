import pytest
import numpy as np
import pandas as pd
from dcurves.surv_dca import _surv_convert_to_risk, _surv_calculate_test_consequences, surv_dca
from dcurves.load_test_data import load_survival_df

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm

from dcurves.plot_graphs import plot_net_benefit


def test_scratch1():
    surv_df = load_survival_df()
    time = 1
    tto_col = 'ttcancer'

    surv_calcs_df = \
        surv_dca(
            data=surv_df,
            outcome='cancer',
            predictors=['marker', 'cancerpredmarker', 'famhistory'],
            predictors_to_prob=['marker'],
            prevalence=0.5,
            harm={'cancerpredmarker': 0.3},
            time=1.0,
            time_to_outcome_col='ttcancer'
        )

    plot_net_benefit(
        data=surv_calcs_df
    )


    # _surv_convert_to_risk(
    #     data=surv_df,
    #     outcome='cancer',
    #     predictors_to_prob=''
    # )




# def test_scratch1():
#     pass
#
# def test_scratch1():
#     pass
# def test_scratch1():
#     pass
# def test_scratch1():
#     pass
# def test_scratch1():
#     pass
# def test_scratch1():
#     pass


