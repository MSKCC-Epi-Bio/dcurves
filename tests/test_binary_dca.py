import pytest
import numpy as np
import pandas as pd
from dcurves.binary_dca import _binary_convert_to_risk, _binary_calculate_test_consequences, binary_dca
from dcurves.load_test_data import load_binary_df
from dcurves.load_test_data import load_r_bctr_result_1, load_r_bin_dca_result_1
from dcurves.load_test_data import load_r_bctr_result_2, load_r_bin_dca_result_2

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm

def test_bctr_1():

    # Load simulation data
    binary_df = load_binary_df()
    # Data loaded from R dcurves calculation of risk scores
    r_bctr_result_1_df = load_r_bctr_result_1()
    p_bctr_result_1_df = \
        _binary_convert_to_risk(
            model_frame=binary_df,
            outcome='cancer',
            predictors_to_prob=['marker']  # 'marker'
        )

    r_bctr_result_1_df = r_bctr_result_1_df.round(12)
    p_bctr_result_1_df = p_bctr_result_1_df.round(12)

    for i in r_bctr_result_1_df['marker'].values:
        assert i in set(p_bctr_result_1_df['marker'].values)

def test_b_dca_result_1():

    df_binary = load_binary_df()
    r_b_dca_result_df = load_r_bin_dca_result_1()

    p_b_dca_result_df = \
        binary_dca(
            data=df_binary,
            outcome='cancer',
            predictors=['cancerpredmarker','marker'],
            probabilities=[False, True]
        )

    del r_b_dca_result_df['label']

    r_b_dca_result_df = r_b_dca_result_df.round(13)
    p_b_dca_result_df = p_b_dca_result_df.round(13)

    p_b_dca_result_df = p_b_dca_result_df.rename(columns={'fpr': 'fp_rate',
                                                          'tpr': 'tp_rate'})

    r_b_dca_result_df = r_b_dca_result_df.sort_values(by=['variable', 'threshold'])
    p_b_dca_result_df = p_b_dca_result_df.sort_values(by=['variable', 'threshold'])

    p_b_dca_result_df = p_b_dca_result_df[r_b_dca_result_df.columns]

    # Compare values between r & p binary DCA result

    # test_pos_rate
    for i in range(0, len(r_b_dca_result_df['test_pos_rate'].values)):
        assert r_b_dca_result_df['tp_rate'].values[i] == p_b_dca_result_df['tp_rate'].values[i]

    # true positive rate
    for i in range(0, len(r_b_dca_result_df['tp_rate'].values)):
        assert r_b_dca_result_df['tp_rate'].values[i] == p_b_dca_result_df['tp_rate'].values[i]

    # false positive rate
    for i in range(0, len(r_b_dca_result_df['fp_rate'].values)):
        assert r_b_dca_result_df['fp_rate'].values[i] == p_b_dca_result_df['fp_rate'].values[i]

    # net benefit
    for i in range(0, len(r_b_dca_result_df['net_benefit'].values)):
        assert r_b_dca_result_df['net_benefit'].values[i] == p_b_dca_result_df['net_benefit'].values[i]


    # for i in range(0, len(r_b_dca_result_df['tp_rate'].values)):
    #     print('r and p tpr values: ' +
    #           str(r_b_dca_result_df['tp_rate'].values[i]) +
    #           ' and ' +
    #           str(p_b_dca_result_df['tp_rate'].values[i]) +
    #           ' at index: ' +
    #           str(i))
    #     assert r_b_dca_result_df['tp_rate'].values[i] == p_b_dca_result_df['tp_rate'].values[i]

# def test_b_dca_prev():
#
#     df_binary = load_binary_df()
#     r_b_dca_result_df = load_r_bin_dca_result_2()
#
#     p_b_dca_result_df = \
#         binary_dca(
#             data=df_binary,
#             outcome='cancer',
#             predictors=['age'],
#             probabilities=[True],
#             prevalence=[0.5]
#         )
#
#     del r_b_dca_result_df['label']
#
#     # Rounding to 13 digits to smooth out minute differences
#     # between R and Python calcs
#     r_b_dca_result_df = r_b_dca_result_df.round(13)
#     p_b_dca_result_df = p_b_dca_result_df.round(13)
#
#     p_b_dca_result_df = p_b_dca_result_df.rename(columns={'fpr': 'fp_rate',
#                                                           'tpr': 'tp_rate'})
#
#     r_b_dca_result_df = r_b_dca_result_df.sort_values(by=['variable', 'threshold'])
#     p_b_dca_result_df = p_b_dca_result_df.sort_values(by=['variable', 'threshold'])
#
#     p_b_dca_result_df = p_b_dca_result_df[r_b_dca_result_df.columns]
#
#     # Compare values between r & p binary DCA result
#
#     # test_pos_rate
#     for i in range(0, len(r_b_dca_result_df['test_pos_rate'].values)):
#         assert r_b_dca_result_df['tp_rate'].values[i] == p_b_dca_result_df['tp_rate'].values[i]
#
#     # true positive rate
#     for i in range(0, len(r_b_dca_result_df['tp_rate'].values)):
#         assert r_b_dca_result_df['tp_rate'].values[i] == p_b_dca_result_df['tp_rate'].values[i]
#
#     # false positive rate
#     for i in range(0, len(r_b_dca_result_df['fp_rate'].values)):
#         assert r_b_dca_result_df['fp_rate'].values[i] == p_b_dca_result_df['fp_rate'].values[i]
#
#     # net benefit
#     for i in range(0, len(r_b_dca_result_df['net_benefit'].values)):
#         assert r_b_dca_result_df['net_benefit'].values[i] == p_b_dca_result_df['net_benefit'].values[i]


# def test_load_newdata():
#     asdf = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
#     print(asdf)
#
#     return
