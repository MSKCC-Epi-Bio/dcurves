
# Load Functions To Test/Needed For Testing
from dcurves.dca import _calc_tp_rate, _calc_fp_rate, _calc_test_pos_rate
from dcurves.dca import _calc_risk_rate_among_test_pos
from dcurves.risks import _create_risks_df
from dcurves.dca import _calc_prevalence, _create_initial_df, _calc_initial_stats
from dcurves.dca import _rectify_model_risk_boundaries

# Load Data for Testing
from dcurves.load_test_data import load_r_case2_results
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.load_test_data import load_tutorial_r_stdca_coxph_df
from dcurves.load_test_data import load_tutorial_r_stdca_coxph_pr_failure18_test_consequences
from dcurves.load_test_data import load_r_simple_surv_dca_result_df
from dcurves.load_test_data import load_r_simple_surv_tpfp_calc_df
# Load Tools
import pandas as pd
import numpy as np

# Load Statistics Libraries
import lifelines

# 230102 SP: Left off here, trying to figure out why survival dca won't match r survival dca results
# Doesn't match for pr_failure18 results, so trying on simple case with df surv first, compare to R results

def test_case2_surv_risk_rate_among_test_positive():
    # Note: To calc risk rate among test positive, divide tp_rate by test_pos_rate

    data = load_survival_df()
    thresholds = np.arange(0.00, 1.0, 0.01)
    outcome = 'cancer'
    modelnames = ['cancerpredmarker']
    models_to_prob = None
    time = 1
    time_to_outcome_col = 'ttcancer'

    r_benchmark_results = load_r_case2_results()

    all_rratp = (r_benchmark_results[r_benchmark_results['variable'] == 'all']['tp_rate'] / \
                r_benchmark_results[r_benchmark_results['variable'] == 'all']['test_pos_rate']).reset_index(drop=1)

    none_rratp = pd.Series(np.repeat(a=0.0, repeats=100))

    cpm_rratp = (r_benchmark_results[r_benchmark_results['variable'] == 'cancerpredmarker']['tp_rate'] / \
                r_benchmark_results[r_benchmark_results['variable'] == 'cancerpredmarker'][
                    'test_pos_rate']).reset_index(drop=1)

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    p_rratp_results = {}
    for model in ['all', 'none']:

        p_rratp_results[model] = \
            _calc_risk_rate_among_test_pos(
                risks_df=rectified_risks_df,
                outcome='cancer',
                model=model,
                thresholds=thresholds,
                time_to_outcome_col='ttcancer',
                time=time
            )

    # model = 'cancerpredmarker'
    #
    # p_rratp_results[model] = \
    #     _calc_risk_rate_among_test_pos(
    #         risks_df=rectified_risks_df,
    #         outcome='cancer',
    #         model=model,
    #         thresholds=thresholds,
    #         time_to_outcome_col='ttcancer',
    #         time=time
    #     )
    #
    # p_rratp_results_df = pd.DataFrame(p_rratp_results).reset_index(drop=True)

    # print('\n',
    #       pd.concat(
    #           [p_rratp_results_df,
    #            all_rratp,
    #            none_rratp,
    #            cpm_rratp],
    #           axis=1
    #       ).to_string()
    # )

    # round_dec_num = 6
    # assert p_rratp_results_df['all'].round(decimals=round_dec_num).equals(all_rratp.round(decimals=round_dec_num))
    # assert p_rratp_results_df['none'][54] == 0



    # print(sorted(risks_df[time_to_outcome_col].values))

    # print('\n', risks_df.to_string())
    #
    # sorted_ttcancer_vals = pd.Series(sorted(risks_df[time_to_outcome_col].values))
    #
    # print(
    #     '\n',
    #     pd.concat(
    #         [
    #             sorted_ttcancer_vals,
    #             p_rratp_results['cancerpredmarker']
    #         ],
    #         axis=1
    #     ).to_string()
    # )

    # print('\n', p_rratp_results_df['none'])
    # print('\n', none_rratp)

    risk_rate_among_test_pos = []

    for threshold in thresholds:
        risk_above_thresh_time = risks_df[risks_df[model] >= threshold][time_to_outcome_col]
        risk_above_thresh_outcome = risks_df[risks_df[model] >= threshold][outcome]

        kmf = lifelines.KaplanMeierFitter()

        if np.max(risks_df['ttcancer']) < time:
            risk_rate_among_test_pos.append(None)
        elif len(risk_above_thresh_time) == 0 and len(risk_above_thresh_outcome) == 0:
            risk_rate_among_test_pos.append(0)
        else:
            kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
            risk_rate_among_test_pos.append(1 - float(kmf.survival_function_at_times(time)))



# def test_risk_rate_among_test_pos():
#     pass

# def test_surv_tp_rate():
#     df_test_simple_surv_tpfp = load_r_simple_surv_tpfp_calc_df()
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     prevalence = None
#     time = 1.5
#     time_to_outcome_col = 'ttcancer'
#     models_to_prob = None
#     modelnames = ['famhistory']
#     thresholds = np.arange(0, 1.0, 0.01)
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
#     test_pos_rate = \
#         _calc_test_pos_rate(risks_df=risks_df,
#                             thresholds=thresholds,
#                             model=modelnames[0]
#                             )
#
#     risk_rate_among_test_pos = \
#         _calc_risk_rate_among_test_pos(
#             risks_df=risks_df,
#             outcome=outcome,
#             model=modelnames[0],
#             thresholds=thresholds,
#             time_to_outcome_col=time_to_outcome_col,
#             time=time
#         )
#
#     tp_rate = risk_rate_among_test_pos * test_pos_rate
#
#     round_dec_num = 6
#     assert df_test_simple_surv_tpfp['tp_rate'].round(decimals=round_dec_num).equals(tp_rate.round(decimals=round_dec_num))
#
# def test_surv_fp_rate():
#     df_test_simple_surv_tpfp = load_r_simple_surv_tpfp_calc_df()
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     prevalence = None
#     time = 1.5
#     time_to_outcome_col = 'ttcancer'
#     models_to_prob = None
#     modelnames = ['famhistory']
#     thresholds = np.arange(0, 1.0, 0.01)
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
#     test_pos_rate = \
#         _calc_test_pos_rate(risks_df=risks_df,
#                             thresholds=thresholds,
#                             model=modelnames[0]
#                             )
#
#     risk_rate_among_test_pos = \
#         _calc_risk_rate_among_test_pos(
#             risks_df=risks_df,
#             outcome=outcome,
#             model=modelnames[0],
#             thresholds=thresholds,
#             time_to_outcome_col=time_to_outcome_col,
#             time=time
#         )
#
#     fp_rate = (1 - risk_rate_among_test_pos) * test_pos_rate
#
#     # comp_df = \
#     #     pd.concat(
#     #         [
#     #             fp_rate,
#     #             df_test_simple_surv_tpfp['fp_rate']
#     #
#     #         ], axis=1
#     #     )
#
#     round_dec_num = 5
#     assert df_test_simple_surv_tpfp['fp_rate'].round(decimals=round_dec_num).equals(
#         fp_rate.round(decimals=round_dec_num))
#
# def test_initial_stats():
#
#     df_test_simple_surv_tpfp = load_r_simple_surv_tpfp_calc_df()
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     prevalence = None
#     time = 1.5
#     time_to_outcome_col = 'ttcancer'
#     models_to_prob = None
#     modelnames = ['famhistory']
#     thresholds = np.arange(0, 1.0, 0.01)
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
#     for model in initial_df['model'].value_counts().index:
#         test_pos_rate = _calc_test_pos_rate(risks_df=risks_df,
#                                             thresholds=thresholds,
#                                             model=model)
#         tp_rate = \
#             _calc_tp_rate(
#                 risks_df=risks_df,
#                 thresholds=thresholds,
#                 model=model,
#                 outcome=outcome,
#                 time=time,
#                 time_to_outcome_col=time_to_outcome_col,
#                 test_pos_rate=test_pos_rate,
#                 prevalence_value=prevalence_value
#             )
#
#         fp_rate = \
#             _calc_fp_rate(
#                 risks_df=risks_df,
#                 thresholds=thresholds,
#                 model=model,
#                 outcome=outcome,
#                 time=time,
#                 time_to_outcome_col=time_to_outcome_col,
#                 test_pos_rate=test_pos_rate,
#                 prevalence_value=prevalence_value
#             )
#
#         initial_df.loc[initial_df['model'] == model, 'test_pos_rate'] = test_pos_rate.tolist()
#         initial_df.loc[initial_df['model'] == model, 'tp_rate'] = tp_rate.tolist()
#         initial_df.loc[initial_df['model'] == model, 'fp_rate'] = fp_rate.tolist()
#
#     assert not initial_df.isnull().values.any()
#
#     round_dec_num = 6
#     for i in ['test_pos_rate', 'tp_rate', 'fp_rate']:
#         assert initial_df[initial_df['model'] == 'famhistory'][i].round(
#             decimals=\
#             round_dec_num).equals(df_test_simple_surv_tpfp[i].round(decimals=round_dec_num))
#
# def test_tut_pr_failure18_tp_rate():
#
#     df_time_to_cancer_dx = \
#             pd.read_csv(
#                     "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#                 )
#     cph = lifelines.CoxPHFitter()
#     cph.fit(df=df_time_to_cancer_dx,
#             duration_col='ttcancer',
#             event_col='cancer',
#             formula='age + famhistory + marker')
#     cph_pred_vals = \
#         cph.predict_survival_function(
#             df_time_to_cancer_dx[['age',
#                                   'famhistory',
#                                   'marker']],
#             times=[1.5]
#         )
#     df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]
#
#     df_time_to_cancer_dx = \
#             pd.read_csv(
#                     "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#                 )
#     cph = lifelines.CoxPHFitter()
#     cph.fit(df=df_time_to_cancer_dx,
#             duration_col='ttcancer',
#             event_col='cancer',
#             formula='age + famhistory + marker')
#     cph_pred_vals = \
#         cph.predict_survival_function(
#             df_time_to_cancer_dx[['age',
#                                   'famhistory',
#                                   'marker']],
#             times=[1.5]
#         )
#
#     df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]
#
#     outcome = 'cancer'
#     prevalence = None
#     time = 1.5
#     time_to_outcome_col = 'ttcancer'
#     thresholds = np.arange(0, 0.51, 0.01)
#     modelnames = ['pr_failure18']
#     harm = None
#
#     risks_df = \
#         _create_risks_df(
#             data=df_time_to_cancer_dx,
#             outcome='cancer',
#             models_to_prob=None,
#             time=time,
#             time_to_outcome_col='ttcancer'
#         )
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
#     initial_df = \
#         _create_initial_df(
#             thresholds=thresholds,
#             modelnames=modelnames,
#             input_df_rownum=len(risks_df.index),
#             prevalence_value=prevalence_value,
#             harm=harm
#         )
#
#     initial_stats_df = \
#         _calc_initial_stats(
#             initial_df=initial_df,
#             risks_df=risks_df,
#             thresholds=thresholds,
#             outcome=outcome,
#             prevalence_value=prevalence_value,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         ).reset_index().sort_values(by=['model',
#                                         'threshold'],
#                                     ascending=[True,
#                                                True]).reset_index(drop=True)
#
#     r_stats_df = load_tutorial_r_stdca_coxph_df().reset_index().sort_values(by=['variable',
#                                         'threshold'],
#                                     ascending=[True,
#                                                True]).reset_index(drop=True)
#
#     pr_failure18_conseq = load_tutorial_r_stdca_coxph_pr_failure18_test_consequences()
#
#     comp_df = \
#         pd.concat(
#             [
#                 initial_stats_df,
#                 r_stats_df
#             ],
#             axis=1
#         ).drop(['harm', 'index', 'label', 'n', 'net_benefit'], axis=1)
#
#     # Remove all, none models since those are the same
#     # Differences pop up in pr_failure18
#
#     comp_df = comp_df[comp_df['model']=='pr_failure18'].reset_index(drop=True)
#     comp_df = pd.concat([comp_df, pr_failure18_conseq[['r_test_pos_rate']]], axis=1)
#
#     # From test below we know test_pos_rate is same between Python and R
#     round_dec_num = 7
#     assert comp_df['test_pos_rate'].round(
#         decimals=round_dec_num
#     ).equals(comp_df['r_test_pos_rate'].round(
#         decimals=round_dec_num
#     ))





