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
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs


def test_binary_dca():
    data = load_binary_df()
    outcome = 'cancer'
    modelnames = ['marker', 'cancerpredmarker']
    models_to_prob = ['marker']
    harm = {
        'cancerpredmarker': 0.04
    }
    thresholds = np.linspace(0, 1, 100)

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


from dcurves.dca import _create_risks_df, _calc_prevalence, _create_initial_df
from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats

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

import dcurves
import pandas as pd
import statsmodels.api as sm


def test_python_model():
    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    mod = sm.GLM.from_formula('cancer ~ famhistory', data=df_cancer_dx, family=sm.families.Binomial())
    mod_results = mod.fit()

    # print(mod_results.summary())


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

def test_python_joint():
    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)


    # print(df_cancer_dx['cancerpredmarker'] > 0.15)
    print(np.where((df_cancer_dx['risk_group'] == 'high') |
                   (df_cancer_dx['cancerpredmarker'] > 0.15, 1, 0)))

    # df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high' |
    #                                   df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)



    # [1 if risk_level == 'high' | (risk_level == "intermediate" &
    #                                cancerpredmarker > 0.15) for (risk_level, cancerpredmarker) in zip(df_cancer_dx['risk_group'], df_cancer_dx['cancerpredmarker'])
    #                          else 0]
    # df_cancer_dx['conditional'] = [1 if df_cancer_dx['risk_group'] == 'high' |
    #                                     (df_cancer_dx["risk_group"] == "intermediate" &
    #                                      df_cancer_dx["cancerpredmarker"] > 0.15)
    #                                else 0]

    # print(df_cancer_dx)