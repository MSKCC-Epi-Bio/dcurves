import pytest
import numpy as np
import pandas as pd
from dcurves.binary_dca import _binary_convert_to_risk, _binary_calculate_test_consequences, binary_dca
from dcurves.load_test_data import load_binary_df, load_r_bctr_result_1, load_r_bin_dca_result_1
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
            predictor='marker'  # 'marker'
        )

    r_bctr_result_1_df = r_bctr_result_1_df.round(12)
    p_bctr_result_1_df = p_bctr_result_1_df.round(12)

    for i in r_bctr_result_1_df['marker'].values:
        assert i in set(p_bctr_result_1_df['marker'].values)

def test_b_dca_result_df():

    df_binary = load_binary_df()
    r_b_dca_result_df = load_r_bin_dca_result_1()

    p_b_dca_result_df = \
        binary_dca(
            data=df_binary,
            outcome='cancer',
            predictors=['cancerpredmarker','marker'],
            probabilities=[False, True]
        )

    print(np.sort(r_b_dca_result_df.columns))
    print(np.sort(p_b_dca_result_df.columns))

    # assert np.array_equal(np.sort(r_b_dca_result_df.columns), np.sort(p_b_dca_result_df.columns))

    # print(r_bctr_result_1_df.to_string())

    # assert np.array_equal(post_bctr_df['marker'].values, model_frame['marker'].values)



# def test_bin_ctr_samecols():
#     '''Check to see if _binary_convert_to_risk function maintains original columns'''
#
#     binary_df = load_binary_df()
#     # List of columns to subset from inputted dataframe for convert_to_risk() function
#     # Make dataframe of just predictor(s) and outcome columns
#     model_frame = binary_df[['cancer', 'marker']]
#     # Run function
#     post_bctr_df = \
#         _binary_convert_to_risk(
#             model_frame=model_frame,
#             outcome='cancer',
#             predictor='marker'  # 'marker'
#         )
#
#     assert np.array_equal(post_bctr_df.columns, np.array(['cancer','marker']))




# def test_initial_simdata_same_values

    #
    # print(local_binary_df.columns)
    # print(dan_binary_df.columns)

    # assert local_binary_df.equals(other=dan_binary_df)
#
# def test_binary_dca():
#
#
#
#     binary_nb_1 = load_bin_dca_1()
#
#     local_bindca_calc1 = \
#         binary_dca(
#             data=binary_df,
#             outcome='cancer',
#             predictors=['cancerpredmarker', 'marker'],
#             probabilities=[True, False],
#
#         )
#
#
#     # print()
#     # print(pd.DataFrame([local_bindca_calc1['net_benefit'].values, binary_nb_1['net_benefit'].values]).transpose().to_string())
#
#
#
#     shaun_set = set(local_bindca_calc1['net_benefit'].values)
#     dan_set = set(binary_nb_1['net_benefit'].values)
#
#     tf_in_set = []
#     for i in shaun_set:
#         if i in dan_set:
#             boolval = True
#         else:
#             boolval = False
#
#         tf_in_set.append(boolval)
#
#     print(pd.DataFrame(tf_in_set).to_string())

    #.to_csv('../dcurves/data/check_1.csv')



    # assert(np.array_equal(local_bindca_calc1['net_benefit'].values, binary_nb_1['net_benefit'].values))


    # for i in range(0,len(binary_nb_1)):
    #     print(i)
    #     print('dan_calc: ' + binary_nb_1)
    #     print()
    #

    # local_bindca_nb_calc1 = pd.DataFrame(local_bindca_calc1['net_benefit'])



    # binary_nb_1['nb2'] = local_bindca_nb_calc1['net_benefit']
    # print(binary_nb_1)


    # assert(np.array_equal(binary_nb_1['net_benefit'], local_bindca_nb_calc1['net_benefit']))





    # print(local_bindca_calc1.columns())

    # binary_dca(
    #     data=binary_df,
    #     outcome='cancer',
    #     predictors=[''],
    #
    # )

    # print(
    #     binary_dca(
    #         data=binary_df,
    #         outcome='cancer',
    #         predictors=['marker','cancerpredmarker'],
    #         probabilities=[True,False]
    #     )
    # )
#
#
#     #print(binary_nb_1)

