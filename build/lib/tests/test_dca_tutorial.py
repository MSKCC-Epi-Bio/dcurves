# Load Basic Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Data
from dcurves.load_test_data import load_r_dca_famhistory
from dcurves.load_test_data import load_r_df_cancer_dx, load_r_df_cancer_dx2

# from dcurves.load_test_data import load_survival_df
# from dcurves.load_test_data import load_case_control_df
# from dcurves.load_test_data import load_tutorial_coxph_pr_failure18_vals
# from dcurves.load_test_data import load_tutorial_r_stdca_coxph_df

# Load Stats Libraries
import statsmodels.api as sm
import lifelines

# Load dcurves functions
from dcurves.dca import _calc_prevalence, _create_initial_df
from dcurves.risks import _create_risks_df, _calc_binary_risks, _calc_surv_risks
from dcurves.dca import _calc_initial_stats, _calc_more_stats
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.load_test_data import load_tutorial_bin_interventions_df
from dcurves.dca import dca
from dcurves.dca import _rectify_model_risk_boundaries
from dcurves.plot_graphs import plot_graphs
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
        load_r_dca_famhistory().sort_values(by=['variable',
                                                'threshold'], ascending=[True,
                                                                         True]).reset_index(drop=True)

    df_cancer_dx = \
        pd.read_csv("https://raw.githubusercontent.com/"
                    "ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_result_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory']
        ).sort_values(by=['model',
                          'threshold'],
                      ascending=[True,
                                 True]).reset_index(drop=True)

    for model in ['all', 'none', 'famhistory']:
        r_nb = df_r_dca_famhistory[df_r_dca_famhistory.variable == model][
            'net_benefit'].round(decimals=6).reset_index(drop=True)
        p_nb = dca_result_df[dca_result_df.model == model][
            'net_benefit'].round(decimals=6).reset_index(drop=True)

        assert r_nb.equals(p_nb)

    # plot_graphs(
    #     plot_df=dca_result_df
    # )

def test_python_famhistory2():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_famhistory2_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=np.arange(0, 0.36, 0.01),
        )

    # plot_graphs(
    #     plot_df=dca_result_df,
    #     graph_type='net_benefit'
    # )


def test_python_model_multi():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    mod = sm.GLM.from_formula('cancer ~ marker + age + famhistory', data=df_cancer_dx, family=sm.families.Binomial())
    mod_results = mod.fit()
#     print(mod_results.summary())

def test_python_dca_multi():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_multi_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory', 'cancerpredmarker'],
            thresholds=np.arange(0, 0.36, 0.01)
        )

    # plot_graphs(
    #     plot_df=dca_multi_df,
    #     y_limits=[-0.05, 0.2],
    #     graph_type='net_benefit'
    # )

def test_python_pub_model():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['logodds_brown'] = 0.75 * df_cancer_dx['famhistory'] + 0.26*df_cancer_dx['age'] - 17.5
    df_cancer_dx['phat_brown'] = np.exp(df_cancer_dx['logodds_brown']) / (1 + np.exp(df_cancer_dx['logodds_brown']))

    dca_pub_model_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['phat_brown'],
            thresholds=np.arange(0,0.36,0.01),
        )

    # plot_graphs(
    #     plot_df=dca_pub_model_df,
    #     y_limits=[-0.05, 0.2],
    #     graph_type='net_benefit'
    # )

def test_python_joint():
    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)

    df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high') |
                                     (df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)

    df_cancer_dx['conditional'] = np.where((df_cancer_dx['risk_group'] == "high") |
                                           ((df_cancer_dx['risk_group'] == "intermediate") &
                                            (df_cancer_dx['cancerpredmarker'] > 0.15)), 1, 0)

def test_python_dca_joint():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)

    df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high') |
                                     (df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)

    df_cancer_dx['conditional'] = np.where((df_cancer_dx['risk_group'] == "high") |
                                           ((df_cancer_dx['risk_group'] == "intermediate") &
                                            (df_cancer_dx['cancerpredmarker'] > 0.15)), 1, 0)

    dca_joint_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['high_risk', 'joint', 'conditional'],
            thresholds=np.arange(0, 0.36, 0.01)
        )

    # plot_graphs(
    #     plot_df=dca_joint_df,
    #     graph_type='net_benefit'
    # )


def test_python_dca_harm_simple():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_harm_simple_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            thresholds=np.arange(0, 0.36, 0.01),
            harm={'marker': 0.0333},
            models_to_prob=['marker']
        )

    # plot_graphs(
    #     plot_df=dca_harm_simple_df,
    #     graph_type='net_benefit'
    # )

def test_python_dca_harm():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    harm_marker = 0.0333
    harm_conditional = (df_cancer_dx['risk_group'] == "intermediate").mean() * harm_marker

    dca_harm_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['risk_group'],
            models_to_prob=['risk_group'],
            thresholds=np.arange(0, 0.36, 0.01),
            harm={'risk_group': harm_conditional}
        )

    # plot_graphs(
    #     plot_df=dca_harm_df,
    #     graph_type='net_benefit'
    # )

def test_python_dca_table():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_table_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            models_to_prob=['marker'],
            thresholds=np.arange(0.05, 0.36, 0.15)
        )

    # print('\n', dca_table_df[['model', 'threshold', 'net_benefit']])

def test_python_dca_intervention():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_intervention_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            thresholds=np.arange(0.05, 0.36, 0.01),
            models_to_prob=['marker']
        )

    # plot_graphs(
    #     plot_df=dca_intervention_df,
    #     graph_type='net_intervention_avoided'
    # )

# def test_python_import_ttcancer():
#
#     df_time_to_cancer_dx = \
#         pd.read_csv(
#             "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
#         )
#
#     print(df_time_to_cancer_dx)

def test_python_coxph():

    df_time_to_cancer_dx = \
        pd.read_csv(
                "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
            )

    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')

    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5])

    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]


def test_python_stdca_coxph():
    df_time_to_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
        )

    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')

    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5])

    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]

    stdca_coxph_results = \
        dca(
            data=df_time_to_cancer_dx,
            outcome='cancer',
            modelnames=['pr_failure18'],
            thresholds=np.arange(0, 0.51, 0.01),
            time=1.5,
            time_to_outcome_col='ttcancer'
        )

    # plot_graphs(
    #     plot_df=stdca_coxph_results,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )

def test_python_dca_case_control():

    df_case_control = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx_case_control.csv"
        )
    # Summarize Data
    medians = df_case_control.drop(columns='patientid').groupby(['casecontrol']).median()
    # print('\n', medians.to_string())


def test_python_dca_case_control():

    df_case_control = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx_case_control.csv"
        )

    dca_case_control_df = \
        dca(
            data=df_case_control,
            outcome='casecontrol',
            modelnames=['cancerpredmarker'],
            prevalence=0.20,
            thresholds=np.arange(0, 0.51, 0.01)
        )

    # plot_graphs(
    #     plot_df=dca_case_control_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )

def test_python_cross_validation():

    # Code below from converting R cross validation via ChatGPT, needs to be tweaked

    # import random
    # import numpy as np
    # import pandas as pd
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import net_benefit
    # from sklearn.metrics import plot_roc_curve
    # from sklearn.metrics import roc_curve
    # from sklearn.metrics import auc
    # from sklearn.metrics import classification_report
    # from sklearn.metrics import confusion_matrix
    # from sklearn.model_selection import RepeatedKFold
    # from sklearn.model_selection import cross_val_score
    #
    # df_cancer_dx = \
    #     pd.read_csv(
    #         "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx_case_control.csv"
    #     )
    #
    # # set seed
    # random.seed(112358)
    #
    # # Create a 10-fold cross validation set
    # cv = RepeatedKFold(n_splits=10, n_repeats=25, random_state=112358)
    #
    # for train_index, test_index in cv.split(df_cancer_dx):
    #     # for each cut of the data, build logistic regression on the 90% (analysis set),
    #     # and perform DCA on the 10% (assessment set)
    #     X_train, X_test = df_cancer_dx.iloc[train_index], df_cancer_dx.iloc[test_index]
    #     y_train, y_test = df_cancer_dx.iloc[train_index], df_cancer_dx.iloc[test_index]
    #
    #     # build regression model on analysis set
    #     logistic = LogisticRegression()
    #     logistic.fit(X_train, y_train)
    #     logistic_pred = logistic.predict(X_test)
    #     logistic_pred_prob = logistic.predict_proba(X_test)
    #
    #     # get predictions for assessment set
    #     df_assessment = pd.DataFrame(logistic_pred_prob)
    #     df_assessment.columns = logistic.classes_
    #     df_assessment["predict"] = logistic_pred
    #     df_assessment["real"] = y_test
    #     df_assessment["fitted"] = logistic.predict_proba(X_test)[:, 1]
    #
    #     # calculate net benefit on assessment set
    #     thresholds = np.linspace(0, 0.35, 36)
    #     dca_assessment = []
    #     for threshold in thresholds:
    #         predict = df_assessment["fitted"].apply(lambda x: 1 if x > threshold else 0)
    #         net_benefit_val = net_benefit(df_assessment["real"], predict)
    #         dca_assessment.append(net_benefit_val)
    #     dca_assessment = pd.DataFrame(dca_assessment)
    #     dca_assessment.columns = ["net_benefit"]
    #
    #     # pool results from the 10-fold cross validation
    #     dca_assessment = dca_assessment.mean()
    #     print(dca_assessment)
    pass
