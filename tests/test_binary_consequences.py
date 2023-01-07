
# Load Functions To Test/Needed For Testing
from dcurves.dca import _calc_tp_rate, _calc_fp_rate, _calc_test_pos_rate
from dcurves.dca import _calc_risk_rate_among_test_pos
from dcurves.risks import _create_risks_df
from dcurves.dca import _calc_prevalence, _create_initial_df, _calc_modelspecific_stats
from dcurves.dca import _calc_nonspecific_stats
from dcurves.dca import dca

# Load Data for Testing
from dcurves.load_test_data import load_binary_df
from dcurves.load_test_data import load_r_simple_binary_dca_result_df

# Load Tools
import pandas as pd
import numpy as np

# Load Statistics Libraries
import lifelines

# def test_binary_tp_rate():
#     data = load_binary_df()
#     outcome = 'cancer'
#     prevalence = None
#     time = None
#     time_to_outcome_col = None
#     models_to_prob = ['age']
#     modelnames = ['age']
#     thresholds = np.arange(0, 1.0, 0.01)
#     model = modelnames[0]
#     harm = None
#
#     risks_df = \
#         _create_risks_df(
#             data=data,
#             outcome=outcome,
#             models_to_prob=models_to_prob,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#
#     # 3. calculate prevalences
#
#     prevalence_value = \
#         _calc_prevalence(
#             risks_df=risks_df,
#             outcome=outcome,
#             prevalence=prevalence,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#
#     # 4. Create initial dataframe for binary/survival cases
#
#     initial_df = \
#         _create_initial_df(
#             thresholds=thresholds,
#             modelnames=modelnames,
#             input_df_rownum=len(risks_df.index),
#             prevalence_value=prevalence_value,
#             harm=harm
#         )
#
#     # 5. Calculate model-specific consequences
#
#     initial_stats_df = \
#         _calc_modelspecific_stats(
#             initial_df=initial_df,
#             risks_df=risks_df,
#             thresholds=thresholds,
#             outcome=outcome,
#             prevalence_value=prevalence_value,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#
#     # 6. Generate DCA-ready df with full list of calculated statistics
#     final_dca_df = \
#         _calc_nonspecific_stats(
#             initial_stats_df=initial_stats_df
#         )

    # r_dca_results = load_r_simple_binary_dca_result_df()

    # round_dec_num = 6
    # assert r_dca_results[r_dca_results['variable']=='age'][
    #     'tp_rate'].round(decimals=round_dec_num).equals(tp_rate.round(decimals=round_dec_num))

    # print(
    #     '\n', r_dca_results[r_dca_results['variable'] == 'age']['tp_rate'].to_string()
    # )
    

    # print('\n',
    #       pd.concat([
    #           r_dca_results[r_dca_results['variable'] == 'age']['tp_rate'].reset_index(drop=1),
    #           tp_rate
    #       ], axis=1).to_string())


    # true_outcome = risks_df[risks_df[outcome] == True][[modelnames[0]]]
    # tp_rate = []
    # for threshold in thresholds:
    #     try:
    #         tp_rate.append(
    #             pd.Series(true_outcome[modelnames[0]] >= threshold).value_counts()[1] / len(true_outcome[modelnames[0]]) * (
    #                 prevalence_value))
    #     except KeyError:
    #         tp_rate.append(0 / len(true_outcome[modelnames[0]]) * prevalence_value)
    #
    # print(pd.Series(tp_rate))
    #
    # false_outcome = risks_df[risks_df[outcome] == False][[model]]
    #
    # fp_rate = []
    # for threshold in thresholds:
    #     try:
    #         fp_rate.append(pd.Series(false_outcome[model] >= threshold).value_counts()[1] / len(
    #             false_outcome[model]) * (1 - prevalence_value))
    #     except KeyError:
    #         fp_rate.append(0 / len(false_outcome[model]) * (1 - prevalence_value))

def test_binary_famhistory_error():

    data = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    outcome = 'cancer'
    prevalence = None
    time = None
    time_to_outcome_col = None
    models_to_prob = None
    modelnames = ['famhistory']
    thresholds = np.arange(0, 1.0, 0.01)
    model = modelnames[0]
    harm = None

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # print('\n', risks_df.to_string())

    # endpoints_adjusted_risks_df = \
    #     _adjust_risks_df_endpoints(
    #         risks_df=risks_df,
    #
    #     )

    # 3. calculate prevalences

    prevalence_value = \
        _calc_prevalence(
            risks_df=risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # 4. Create initial dataframe for binary/survival cases

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    # print('\n', initial_df.to_string())

    # 5. Calculate model-specific consequences
    #
    for model in initial_df['model'].value_counts().index:
        test_pos_rate = _calc_test_pos_rate(risks_df=risks_df,
                                            thresholds=thresholds,
                                            model=model)
    #
    #     print('\n', model)
    #
    #     true_outcome = risks_df[risks_df[outcome] == True][[model]]
    #     print(outcome)
    #     print(risks_df[outcome])
    #     tp_rate = []
        # for threshold in thresholds:
        #
        #     print('\n', pd.Series(true_outcome[model] >= threshold))

            # try:
            #     tp_rate.append(
            #         pd.Series(true_outcome[model] >= threshold).value_counts()[1] / len(true_outcome[model]) * (
            #             prevalence_value))
            # except KeyError:
            #     tp_rate.append(0 / len(true_outcome[model]) * prevalence_value)


