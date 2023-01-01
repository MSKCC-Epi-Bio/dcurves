# # Load dcurves functions
# from dcurves.dca import dca
# from dcurves.dca import net_intervention_avoided
# from dcurves.plot_graphs import plot_graphs
# from dcurves.dca import _calc_prevalence, _create_initial_df
# from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats
# from dcurves.risks import _create_risks_df, _calc_binary_risks, _calc_surv_risks
#
# # Load data functions
# from dcurves.load_test_data import load_binary_df, load_survival_df, load_case_control_df
# from dcurves.load_test_data import load_tutorial_interventions, load_tutorial_risk_df
# from dcurves.load_test_data import load_tutorial_marker_risk_scores
#
# # Load outside functions
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# def test_python_dca():
#     # Test scenario from dca-tutorial r-dca_intervention code chunk
#
#     df_cancer_dx = \
#         pd.read_csv(
#             "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
#         )
#
#     p_dca_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['marker'],
#             models_to_prob=['marker'],
#             thresholds=np.arange(0.05, 0.36, 0.01)
#         )
#
#     unwanted_p_cols = ['index', 'harm', 'n', 'prevalence', 'test_pos_rate', 'neg_rate',
#                        'test_neg_rate', 'ppv', 'npv', 'sens', 'spec', 'lr_pos',
#                        'lr_neg', 'tn_rate', 'fn_rate', 'net_benefit_all']
#
#     p_net_int_df = \
#         net_intervention_avoided(
#             after_dca_df=p_dca_df
#         ).reset_index().sort_values(by=['model',
#                                         'threshold'],
#                                     ascending=[True,
#                                                True]).reset_index(drop=True).drop(unwanted_p_cols, axis=1)
#
#     r_net_int_df = \
#         load_tutorial_interventions().sort_values(by=['variable',
#                                                       'threshold'],
#                                                   ascending=[True,
#                                                              True]).reset_index(drop=True).drop(['label',
#                                                                                                  'pos_rate',
#                                                                                                  'harm',
#                                                                                                  'n'], axis=1)
#
#     assert r_net_int_df['threshold'].round(decimals=10).equals(other=p_net_int_df['threshold'].round(decimals=10))
#
#     print('\n', r_net_int_df.to_string())
#     print('\n', p_net_int_df.to_string())

# def test_dca_risk_conversion():
#     pass
    # r_marker_risks = load_tutorial_marker_risk_scores()
    #
    # df_cancer_dx = \
    #     pd.read_csv(
    #         "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    #     )
    #
    # p_marker_risks = \
    #     _calc_binary_risks(
    #         data=df_cancer_dx,
    #         outcome='cancer',
    #         model='marker'
    #     )
    #
    # print(type(r_marker_risks))
    #
    # print(type(p_marker_risks))

    # r_risk_df = load_tutorial_risk_df()

    # print(r_risk_df.to_string())

    # df_cancer_dx = \
    #     pd.read_csv(
    #         "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    #     )
    #
    #
    #
    # p_risk_df = \
    #     _create_risks_df(
    #         data=df_cancer_dx,
    #         outcome='cancer',
    #         models_to_prob=['marker']
    #     )













# def test_binary_dca():
#     data = load_binary_df()
#     outcome = 'cancer'
#     modelnames = ['marker', 'cancerpredmarker']
#     models_to_prob = ['marker']
#     harm = {
#         'cancerpredmarker': 0.04
#     }
#     thresholds = np.linspace(0, 1, 100)
#
    # dca_df = \
    #     dca(
    #         data=data,
    #         outcome=outcome,
    #         modelnames=modelnames,
    #         models_to_prob=models_to_prob,
    #         harm=harm
    #     )
    #
    #
    # plot_net_benefit(
    #     data=dca_df,
    #     model_name_colname='model'
    # )

    # risks_df = \
    #     _create_risks_df(
    #         data=data,
    #         outcome=outcome,
    #         models_to_prob=models_to_prob,
    #         time=None,
    #         time_to_outcome_col=None
    #     )
    #
    # # 3. calculate prevalences
    #
    # prevalence_value = \
    #     _calc_prevalence(
    #         risks_df=risks_df,
    #         outcome=outcome,
    #         prevalence=None,
    #         time=None,
    #         time_to_outcome_col=None
    #     )
    #
    # # 4. Create initial dataframe for binary/survival cases
    #
    # initial_df = \
    #     _create_initial_df(
    #         thresholds=thresholds,
    #         modelnames=modelnames,
    #         input_df_rownum=len(risks_df.index),
    #         prevalence_value=prevalence_value,
    #         harm=harm
    #     )
    #
    # # 5. Calculate model-specific consequences
    #
    # initial_stats_df = \
    #     _calc_modelspecific_stats(
    #         initial_df=initial_df,
    #         risks_df=risks_df,
    #         thresholds=thresholds,
    #         outcome=outcome,
    #         prevalence_value=prevalence_value,
    #         time=None,
    #         time_to_outcome_col=None
    #     )
    #
    # # 6. Generate DCA-ready df with full list of calculated statistics
    # final_dca_df = \
    #     _calc_nonspecific_stats(
    #         initial_stats_df=initial_stats_df
    #     )
    #
    # print(final_dca_df)


# def test_binary_dca2():
#
#     data = load_binary_df()
#     outcome = 'cancer'
#     modelnames = ['marker', 'cancerpredmarker']
#     models_to_prob = ['marker']
#     harm = {
#         'cancerpredmarker': 0.04
#     }
#     thresholds = np.linspace(0, 1, 100)
#
#     dca_df = \
#         dca(
#             data=data,
#             outcome=outcome,
#             modelnames=modelnames,
#             models_to_prob=models_to_prob,
#             harm=harm
#         )
#
#     plot_net_benefit(
#         data=dca_df,
#         model_name_colname='model'
#     )


# from dcurves.dca import _create_risks_df, _calc_prevalence, _create_initial_df
# from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats

# def test_new_dca_func_2():
#     data = load_survival_df()
#     outcome = 'cancer'
#     modelnames = ['marker', 'cancerpredmarker']
#     models_to_prob = ['marker']
#     time = 1
#     time_to_outcome_col = 'ttcancer'
#     thresholds = np.linspace(0.00, 1.00, 100)
#     prevalence = None
#     harm = {
#         'famhistory': 0,
#         'cancerpredmarker': 0
#     }
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
#     final_dca_df = \
#         _calc_nonspecific_stats(
#             initial_stats_df=initial_stats_df
#         )
#
#     plot_net_benefit(
#         data=final_dca_df,
#         model_name_colname='model'
#     )

# from dcurves.plot_graphs import plot_graphs
# from dcurves.dca import net_intervention_avoided

# from dcurves import *
#
# def test_new_graph_func():
#
#     bin_df = load_binary_df()
#
#     dca_df = \
#         dca(
#             data=bin_df,
#             outcome='cancer',
#             modelnames=['marker', 'cancerpredmarker'],
#             models_to_prob=['marker']
#         )
#
#     net_int_df = \
#         net_intervention_avoided(
#             after_dca_df=dca_df
#         )
#
#     plot_graphs(
#         plot_df=net_int_df,
#         graph_type='net_intervention_avoided',
#         y_limits=[-0.05, 0.3]
#     )











