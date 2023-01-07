# Load Basic Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Data
from dcurves.load_test_data import load_r_dca_famhistory

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
from dcurves.load_test_data import load_tutorial_coxph_pr_failure18_vals
from dcurves.load_test_data import load_tutorial_r_stdca_coxph_df

# Load Stats Libraries
import statsmodels.api as sm
import lifelines

# Load dcurves functions
from dcurves.dca import _calc_prevalence, _create_initial_df
from dcurves.risks import _create_risks_df, _calc_binary_risks, _calc_surv_risks
from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.load_test_data import load_tutorial_bin_interventions_df
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs
from dcurves.dca import net_intervention_avoided
import dcurves


# def test_python_model():
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     mod = sm.GLM.from_formula('cancer ~ famhistory', data=df_cancer_dx, family=sm.families.Binomial())
#     mod_results = mod.fit()
#
#     print(mod_results.summary())


def test_python_famhistory1():

    df_r_dca_famhistory = \
        load_r_dca_famhistory().sort_values(by=['model',
                                                'threshold'],
                                            ascending=[True,
                                                       True]).reset_index(drop=True)

    df_cancer_dx = \
        pd.read_csv("https://raw.githubusercontent.com/\n"
                    "ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv").sort_values(
            by=['variable',
                'threshold'],
            ascending=[True,
                       True]).reset_index(drop=True)



    dca_result_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory']
        ).sort_values(by=['model',
                          'threshold'],
                      ascending=[True,
                                 True]).reset_index(drop=True)

    print(' ')
    print(dca_result_df.columns)
    print(df_r_dca_famhistory.columns)

    # plot_graphs(
    #     plot_df=dca_result_df,
    #     graph_type='net_benefit'
    # )

# def test_python_famhistory2():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_result_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['famhistory'],
#             thresholds=np.arange(0, 0.36, 0.01),
#         )
#
#     plot_graphs(
#         plot_df=dca_result_df,
#         graph_type='net_benefit'
#     )


# def test_python_model_multi():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     mod = sm.GLM.from_formula('cancer ~ marker + age + famhistory', data=df_cancer_dx, family=sm.families.Binomial())
#     mod_results = mod.fit()
#
#     print(mod_results.summary())

# def test_python_dca_multi():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_result_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['famhistory', 'cancerpredmarker'],
#             thresholds=np.arange(0,0.36,0.01)
#         )
#
#     plot_graphs(
#         plot_df=dca_result_df,
#         y_limits=[-0.05, 0.2],
#         graph_type='net_benefit'
#     )

# def test_python_pub_model():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     df_cancer_dx['logodds_brown'] = 0.75 * df_cancer_dx['famhistory'] + 0.26*df_cancer_dx['age'] - 17.5
#     df_cancer_dx['phat_brown'] = np.exp(df_cancer_dx['logodds_brown']) / (1 + np.exp(df_cancer_dx['logodds_brown']))
#
#     dca_result_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['phat_brown'],
#             thresholds=np.arange(0,0.36,0.01),
#         )
#
#     plot_graphs(
#         plot_df=dca_result_df,
#         y_limits=[-0.05, 0.2],
#         graph_type='net_benefit'
#     )

# def test_python_joint():
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)
#
#     df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high') |
#                                      (df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)
#
#     df_cancer_dx['conditional'] = np.where((df_cancer_dx['risk_group'] == "high") |
#                                            ((df_cancer_dx['risk_group'] == "intermediate") &
#                                             (df_cancer_dx['cancerpredmarker'] > 0.15)), 1, 0)

# def test_python_dca_joint():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)
#
#     df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high') |
#                                      (df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)
#
#     df_cancer_dx['conditional'] = np.where((df_cancer_dx['risk_group'] == "high") |
#                                            ((df_cancer_dx['risk_group'] == "intermediate") &
#                                             (df_cancer_dx['cancerpredmarker'] > 0.15)), 1, 0)
#
#
#     dca_joint_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['high_risk', 'joint', 'conditional'],
#             thresholds=np.arange(0,0.36,0.01)
#         )
#
#     plot_graphs(
#         plot_df=dca_joint_df,
#         graph_type='net_benefit'
#     )


# def test_python_dca_harm_simple():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_simple_harm_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['marker'],
#             thresholds=np.arange(0,0.36,0.01),
#             harm={'marker': 0.0333},
#             models_to_prob=['marker']
#         )
#
#     plot_graphs(
#         plot_df=dca_simple_harm_df,
#         graph_type='net_benefit'
#     )

# def test_python_dca_harm():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     harm_marker = 0.0333
#     harm_conditional = (df_cancer_dx['risk_group'] == "intermediate").mean() * harm_marker
#
#     dca_harm_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['risk_group'],
#             models_to_prob=['risk_group'],
#             thresholds=np.arange(0, 0.36, 0.01),
#             harm={'risk_group': harm_conditional}
#         )
#
#     plot_graphs(
#         plot_df=dca_harm_df
#     )

# def test_python_dca_table():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_table_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['marker'],
#             models_to_prob=['marker'],
#             thresholds=np.arange(0.05, 0.36, 0.15)
#         )
#
#     print('\n', dca_table_df[['model', 'threshold', 'net_benefit']])

# def test_python_dca_intervention():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_result_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['marker'],
#             thresholds=np.arange(0.05, 0.36, 0.01),
#             models_to_prob=['marker']
#         )
#
#     dca_interventions_df = \
#         net_intervention_avoided(
#             after_dca_df=dca_result_df
#         )
#
#     plot_graphs(
#         plot_df=dca_interventions_df,
#         graph_type='net_intervention_avoided',
#         y_limits=[-10, 70]
#     )

# def test_python_import_ttcancer():
#
#     df_time_to_cancer_dx = \
#         pd.read_csv(
#             "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#         )
#
#     print(df_time_to_cancer_dx)

# def test_python_coxph():
#
#     df_time_to_cancer_dx = \
#         pd.read_csv(
#                 "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#             )
#
#     cph = lifelines.CoxPHFitter()
#     cph.fit(df=df_time_to_cancer_dx,
#             duration_col='ttcancer',
#             event_col='cancer',
#             formula='age + famhistory + marker')
#
#     cph_pred_vals = \
#         cph.predict_survival_function(
#             df_time_to_cancer_dx[['age',
#                                   'famhistory',
#                                   'marker']],
#             times=[1.5])
#
#     df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]


# def test_python_stdca_coxph():
#
#     r_stdca_coxph_df = load_tutorial_r_stdca_coxph_df()
#
#     r_stdca_coxph_df = \
#         r_stdca_coxph_df.sort_values(by=['variable',
#                                          'threshold'],
#                                      ascending=[True,
#                                                 True]).reset_index(drop=True).drop(['label',
#                                                                                     'pos_rate',
#                                                                                     'harm',
#                                                                                     'n'], axis=1)
#
#     df_time_to_cancer_dx = \
#         pd.read_csv(
#                 "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#             )
#
#     cph = lifelines.CoxPHFitter()
#     cph.fit(df=df_time_to_cancer_dx,
#             duration_col='ttcancer',
#             event_col='cancer',
#             formula='age + famhistory + marker')
#
#     cph_pred_vals = \
#         cph.predict_survival_function(
#             df_time_to_cancer_dx[['age',
#                                   'famhistory',
#                                   'marker']],
#             times=[1.5])
#
#     df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]
#
#     surv_dca_results = \
#         dca(
#             data=df_time_to_cancer_dx,
#             outcome='cancer',
#             modelnames=['pr_failure18'],
#             thresholds=np.arange(0, 0.51, 0.01),
#             time=1.5,
#             time_to_outcome_col='ttcancer'
#         )
#     surv_dca_results = \
#         surv_dca_results.reset_index().sort_values(by=['model',
#                                                        'threshold'],
#                                                    ascending=[True,
#                                                               True]).reset_index(drop=True).drop(['index',
#                                                                                                   'n',
#                                                                                                   'prevalence',
#                                                                                                   'harm',
#                                                                                                   'test_pos_rate'],
#                                                                                                  axis=1)
#
#
#     round_decimal_num = 0
#
#     # assert r_stdca_coxph_df[
#     #     'threshold'
#     #     ].round(decimals=round_decimal_num).equals(other=surv_dca_results['threshold'].round(decimals=round_decimal_num))
#     # assert r_stdca_coxph_df[
#     #     'tp_rate'].round(decimals=round_decimal_num).equals(other=surv_dca_results['tp_rate'].round(decimals=round_decimal_num))
#     # assert r_stdca_coxph_df[
#     #     'fp_rate'].round(decimals=round_decimal_num).equals(other=surv_dca_results['fp_rate'].round(decimals=round_decimal_num))
#     # assert r_stdca_coxph_df[
#     #     'net_benefit'].round(decimals=round_decimal_num).equals(other=surv_dca_results['net_benefit'].round(decimals=round_decimal_num))
#
#     comp_df = \
#         pd.DataFrame(
#             {
#                 'r_models': r_stdca_coxph_df['variable'].tolist(),
#                 'p_models': surv_dca_results['model'].tolist(),
#                 'r_thresholds': r_stdca_coxph_df['threshold'].tolist(),
#                 'p_thresh': surv_dca_results['threshold'].tolist(),
#                 'r_tp': r_stdca_coxph_df['tp_rate'].tolist(),
#                 'p_tp': surv_dca_results['tp_rate'].tolist()
#             }
#         )
#
#     print('\n', comp_df.to_string())






