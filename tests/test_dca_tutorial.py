import numpy as np
import pandas as pd

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm

# from dcurves.plot_graphs import plot_net_benefit

import statsmodels.api as sm

import matplotlib.pyplot as plt

from dcurves.dca import _create_risks_df, _calc_prevalence, _create_initial_df
from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.load_test_data import load_tutorial_interventions
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs

from dcurves.dca import net_intervention_avoided









import dcurves
import pandas as pd
import statsmodels.api as sm


# def test_python_model():
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     mod = sm.GLM.from_formula('cancer ~ famhistory', data=df_cancer_dx, family=sm.families.Binomial())
#     mod_results = mod.fit()
#
#     print(mod_results.summary())


# def test_python_famhistory1():
#
#     df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
#
#     dca_result_df = \
#         dca(
#             data=df_cancer_dx,
#             outcome='cancer',
#             modelnames=['famhistory']
#         )
#
#     plot_graphs(
#         plot_df=dca_result_df,
#         graph_type='net_benefit'
#     )

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

def test_python_dca_intervention():

    r_interventions_df = load_tutorial_interventions()
    r_interventions_df = r_interventions_df[['variable', 'threshold', 'net_benefit', 'net_intervention_avoided']]

    # print(r_interventions_df.size)

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_result_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            thresholds=np.arange(0.05, 0.36, 0.01),
            models_to_prob=['marker']
        )

    dca_interventions_df = \
        net_intervention_avoided(
            after_dca_df=dca_result_df
        )

    dca_interventions_df = dca_interventions_df[['model', 'threshold', 'net_benefit', 'net_intervention_avoided']]

    r_interventions_df2 = r_interventions_df.sort_values(by=['variable', 'threshold'], ascending=True).reset_index()
    dca_interventions_df2 = dca_interventions_df.sort_values(by=['model', 'threshold'], ascending=True).reset_index()

    # print(r_interventions_df2)

    # print(' ')
    # print('\n', pd.concat([r_interventions_df2, dca_interventions_df2], axis=1).to_string())
    comp_df = pd.concat([r_interventions_df2, dca_interventions_df2], axis=1)

    print('\n', comp_df.to_string())

    print()
    # print(comp_df[['model', 'threshold']])

    # for dan_nb_val, shaun_nb_val in zip(r_interventions_df['net_benefit'], dca_interventions_df['net_benefit']):
    #     print('\n', 'dan_val: ' + str(dan_nb_val), ' and ', 'Shaun_val: ' + str(shaun_nb_val))

    # print('\n', r_interventions_df['net_benefit'].sort_values())
    # print('\n', dca_interventions_df['net_benefit'].sort_values().equals(r_interventions_df['net_benefit'].sort_values()))




    # print('\n', r_interventions_df['net_benefit'].sort_values() == dca_interventions_df['net_benefit'].sort_values())

    # print('\n', len(dca_interventions_df))

    # shaun_nia_comp_df = \
    #     pd.DataFrame({
    #         'shaun_model': dca_interventions_df['model'],
    #         'shaun_thresh': dca_interventions_df['threshold'],
    #         'shaun_nb': dca_interventions_df['net_benefit'],
    #         'shaun_nia': dca_interventions_df['net_intervention_avoided']
    #     })
    #
    # dan_nia_comp_df = \
    #     pd.DataFrame({
    #         'dan_model': r_interventions_df['variable'],
    #         'dan_thresh': r_interventions_df['threshold'],
    #         'dan_nb': r_interventions_df['net_benefit'],
    #         'dan_nia': r_interventions_df['net_intervention_avoided']
    #     })
    #
    # shaun_nia_comp_df2 = shaun_nia_comp_df.sort_values(by=['shaun_model', 'shaun_thresh'], ascending=[True, True])
    # dan_nia_comp_df2 = dan_nia_comp_df.sort_values(by=['dan_model', 'dan_thresh'], ascending=[True, True])
    #
    #
    # comp_df = pd.concat([shaun_nia_comp_df2, dan_nia_comp_df2], axis=1)
    # print('\n', comp_df.to_string())

    #
    # print('\n', comp_df.to_string())



    # print(nia_comp_df.to_string())


    # print(dca_result_df.size)

    # print('\n', r_interventions_df.to_string())

    # print('\n', dca_result_df.to_string())

