# Load Basic Tools
# import numpy as np
import pandas as pd

# Load Data
from .load_test_data import load_r_dca_famhistory

# Load Stats Libraries
import statsmodels.api as sm
import lifelines

# Load dcurves functions
from dcurves import dca, plot_graphs

def test_python_model():
    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    mod = sm.GLM.from_formula('cancer ~ famhistory', data=df_cancer_dx, family=sm.families.Binomial())
    mod_results = mod.fit()

    # print(mod_results.summary())


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

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/"
                               "ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_famhistory2_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=[i/100 for i in range(0, 36)],
        )

    # plot_graphs(
    #     plot_df=dca_famhistory2_df,
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
            thresholds=[i/100 for i in range(0, 36)]
        )

    # plot_graphs(
    #     plot_df=dca_multi_df,
    #     y_limits=[-0.05, 0.2],
    #     graph_type='net_benefit',
    #     color_names=['cyan',
    #                  'purple',
    #                  'red',
    #                  'blue']
    # )

def test_python_pub_model():

    # df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
    #
    # df_cancer_dx['logodds_brown'] = 0.75 * df_cancer_dx['famhistory'] + 0.26*df_cancer_dx['age'] - 17.5
    # df_cancer_dx['phat_brown'] = np.exp(df_cancer_dx['logodds_brown']) / (1 + np.exp(df_cancer_dx['logodds_brown']))

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['logodds_brown'] = 0.75 * df_cancer_dx['famhistory'] + 0.26 * df_cancer_dx['age'] - 17.5
    df_cancer_dx['phat_brown'] = 1 / (1 + (1 / (2.718281828 ** df_cancer_dx['logodds_brown'])))

    dca_pub_model_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['phat_brown'],
            thresholds=[i/100 for i in range(0, 36)],
        )

    # plot_graphs(
    #     plot_df=dca_pub_model_df,
    #     y_limits=[-0.05, 0.2],
    #     graph_type='net_benefit'
    # )

def test_python_joint():
    import pandas as pd

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['high_risk'] = [1 if risk_group == "high" else 0 for risk_group in df_cancer_dx['risk_group']]

    df_cancer_dx['joint'] = [1 if (risk_group == 'high' or cancerpredmarker > 0.15) else 0 for
                             risk_group, cancerpredmarker in
                             zip(df_cancer_dx['risk_group'], df_cancer_dx['cancerpredmarker'])]

    df_cancer_dx['conditional'] = [
        1 if (risk_group == "high" or (risk_group == "intermediate" and cancerpredmarker > 0.15)) else 0 for
        risk_group, cancerpredmarker in zip(df_cancer_dx['risk_group'], df_cancer_dx['cancerpredmarker'])]

def test_python_dca_joint():

    # df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")
    #
    # df_cancer_dx['high_risk'] = np.where(df_cancer_dx['risk_group'] == "high", 1, 0)
    #
    # df_cancer_dx['joint'] = np.where((df_cancer_dx['risk_group'] == 'high') |
    #                                  (df_cancer_dx['cancerpredmarker'] > 0.15), 1, 0)
    #
    # df_cancer_dx['conditional'] = np.where((df_cancer_dx['risk_group'] == "high") |
    #                                        ((df_cancer_dx['risk_group'] == "intermediate") &
    #                                         (df_cancer_dx['cancerpredmarker'] > 0.15)), 1, 0)

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    df_cancer_dx['high_risk'] = df_cancer_dx['risk_group'].apply(lambda x: 1 if x == 'high' else 0)

    df_cancer_dx['joint'] = df_cancer_dx.apply(
        lambda x: 1 if x['risk_group'] == 'high' or x['cancerpredmarker'] > 0.15 else 0, axis=1)

    df_cancer_dx['conditional'] = df_cancer_dx.apply(lambda x: 1 if x['risk_group'] == 'high' or (
                x['risk_group'] == 'intermediate' and x['cancerpredmarker'] > 0.15) else 0, axis=1)

    dca_joint_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['high_risk', 'joint', 'conditional'],
            thresholds=[i/100 for i in range(0, 36)]
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
            thresholds=[i/100 for i in range(0, 36)],
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
            thresholds=[i/100 for i in range(0, 36)],
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
            thresholds=[i/100 for i in range(0, 36)]
        )

    # print('\n', dca_table_df[['model', 'threshold', 'net_benefit']])

def test_python_dca_intervention():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    dca_intervention_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            thresholds=[i/100 for i in range(0, 36)],
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
            thresholds=[i/100 for i in range(0, 51)],
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
            thresholds=[i/100 for i in range(0, 51)]
        )

    # plot_graphs(
    #     plot_df=dca_case_control_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )

def test_python_cross_validation():

    import random
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import log_loss
    import statsmodels.api as sm
    import dcurves

    random.seed(112358)

    df_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
        )

    # Create a 10-fold cross validation set
    cv = RepeatedKFold(n_splits=10, n_repeats=25, random_state=112358)

    # Define the formula (make sure the column names in your DataFrame match these)
    formula = 'cancer ~ marker + age + famhistory'

    # Create cross-validation object
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=112358)

    # Placeholder for predictions
    cv_predictions = []

    # Perform cross-validation
    for train_index, test_index in rkf.split(df_cancer_dx):
        # Split data into training and test sets
        train, test = df_cancer_dx.iloc[train_index], df_cancer_dx.iloc[test_index]

        # Fit the model
        model = sm.Logit.from_formula(formula, data=train).fit(disp=0)

        # Make predictions on the test set
        test['cv_prediction'] = model.predict(test)

        # Store predictions
        cv_predictions.append(test[['patientid', 'cv_prediction']])

    # Concatenate predictions from all folds
    df_predictions = pd.concat(cv_predictions)

    # Calculate mean prediction per patient
    df_mean_predictions = df_predictions.groupby('patientid')['cv_prediction'].mean().reset_index()

    # Join with original data
    df_cv_pred = pd.merge(df_cancer_dx, df_mean_predictions, on='patientid', how='left')

    # Decision curve analysis
    # Generate net benefit score for each threshold value
    df_dca_cv = dcurves.dca(
            data=df_cv_pred, modelnames=['cv_prediction'], outcome='cancer'
        )
    
    assert isinstance(df_dca_cv, pd.DataFrame), "df_dca_cv is not a pandas DataFrame"
