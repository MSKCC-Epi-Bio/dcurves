import pytest
import numpy as np
import pandas as pd
from dcurves.binary_dca import _binary_convert_to_risk, _binary_calculate_test_consequences, binary_dca
from dcurves.load_test_data import load_binary_df
from dcurves.plot_graphs import plot_net_benefit
from dcurves.load_test_data import load_r_bctr_result_1, load_r_bin_dca_result_1
from dcurves.load_test_data import load_r_bctr_result_2, load_r_bin_dca_result_2

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm

import matplotlib

# def test_bctr_1():
#
#     # Load simulation data
#     binary_df = load_binary_df()
#     # Data loaded from R dcurves calculation of risk scores
#     r_bctr_result_1_df = load_r_bctr_result_1()
#     p_bctr_result_1_df = \
#         _binary_convert_to_risk(
#             model_frame=binary_df,
#             outcome='cancer',
#             predictors_to_prob=['marker']  # 'marker'
#         )
#
#     r_bctr_result_1_df = r_bctr_result_1_df.round(12)
#     p_bctr_result_1_df = p_bctr_result_1_df.round(12)
#
#     for i in r_bctr_result_1_df['marker'].values:
#         assert i in set(p_bctr_result_1_df['marker'].values)
#
#
# # def test_scratch():
# #
# #     print(
# #         pd.DataFrame({'asdf': 1,
# #                       'threshold': [np.arange(start=0,
# #                                              stop=1.00,
# #                                              step=0.01)]})
# #     )
#
# def test_scratch2():
#     """
#     Here we learned that using np.append is the best
#     1. Creates copy of first and second inputs
#     2. Flattens both
#     3. Combines them into a single list
#     :return:
#     """
#
#     list1 = ['cancerpredmarker', 'marker', 'famhistory']
#
#     covariate_names = np.append(list1, ['all', 'none'])
#     # print(' ')
#     # print(covariate_names)
#     # print(list1)
#     # for i in covariate_names:
#     #     print(i)

def test_scratch3():

    binary_df = load_binary_df()

    model_frame = binary_df
    outcome = 'cancer'
    predictors = ['marker', 'cancerpredmarker', 'famhistory']
    predictors_to_prob = ['marker']
    thresholds = np.arange(0.00, 1.01, 0.01)
    prevalence = None

    model_frame = \
        _binary_convert_to_risk(
            model_frame=binary_df,
            outcome=outcome,
            predictors_to_prob=predictors_to_prob
        )

    machine_epsilon = np.finfo(float).eps

    model_frame['all'] = [1 - machine_epsilon for i in range(0, len(model_frame.index))]
    model_frame['none'] = [0 + machine_epsilon for i in range(0, len(model_frame.index))]

    thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
    thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)

    covariate_names = np.append(predictors, ['all', 'none'])



    # testcons_list = []

    # all_covariates_df = pd.concat([_binary_calculate_test_consequences(model_frame=model_frame,
    #                                                      outcome=outcome,
    #                                                      predictor=predictor,
    #                                                      thresholds=thresholds,
    #                                                      prevalence=prevalence) for predictor in covariate_names])



    # print(model_frame.columns)


    # outcome in model_frame subset

    # n = 'n': [n] * len(thresholds),
    # df = pd.DataFrame({'predictor': predictor,
    #                    'threshold': thresholds,
    #                    'n': [n] * len(thresholds),
    #                    'prevalence': prevalence_values})


def test_scratch4():
    data = load_binary_df()

    bin_calcs_df = binary_dca(
        data=data,
        thresholds=np.linspace(0, 1.0, 101),
        outcome='cancer',
        predictors=['marker', 'cancerpredmarker', 'famhistory'],
        predictors_to_prob=['marker'],
        prevalence=0.5,
        harm={'cancerpredmarker': 0.3}
    )

    print(' ')
    # print(calcs_df[['threshold', 'net_benefit', 'predictor']])

    plot_net_benefit(
        after_dca_df=bin_calcs_df[['threshold', 'net_benefit', 'predictor']],
        y_limits=[-.05, 1]
    )

    # matplotlib.pyplot.plot(calcs_df['thresholds'])

def test_famhistory_tut():
    import dcurves
    dcurves.binary_dca.




# def test_load_newdata():
#     asdf = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
#     print(asdf)
#
#     return

