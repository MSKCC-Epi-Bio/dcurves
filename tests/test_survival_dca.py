import pytest
import numpy as np
import pandas as pd
from dcurves.surv_dca import _surv_convert_to_risk, _surv_calculate_test_consequences, surv_dca
from dcurves.load_test_data import load_survival_df

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm


def test_scratch1():
    surv_df = load_survival_df()
    time = 1
    tto_col = 'ttcancer'
    print(' ')
    print(
        surv_df[['ttcancer',
                 'age',
                 'famhistory',
                 'marker',
                 'cancerpredmarker',
                 'cancer_cr']].describe(),
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


