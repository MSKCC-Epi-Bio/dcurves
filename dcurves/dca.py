import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
import matplotlib.pyplot as plt

from test import resources



def convert_to_risk(model_frame: pd.DataFrame,
                    outcome: str,
                    predictor: str,
                    prevalence: float,
                    time: float,
                    time_to_outcome_col: str) -> pd.DataFrame:

    """Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    Parameters
    ----------
    model_frame : dataframe
    outcome : string
    predictor : string
    prevalence : float
    time : float
    time_to_outcome_col : string

    Returns
    -------
    dataframe
    """

    if not time_to_outcome_col:
        predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                        data=model_frame).fit().predict()
        model_frame[predictor] = [(1 - val) for val in predicted_vals]
        return model_frame
    elif time_to_outcome_col:
        #### From lifelines dataframe
        cph = lifelines.CoxPHFitter()
        cph_df = model_frame[['ttcancer' ,'cancer' ,'cancerpredmarker']]
        cph.fit(cph_df, 'ttcancer', 'cancer')

        new_cph_df = cph_df
        new_cph_df['ttcancer'] = [time for i in range(0 ,len(cph_df))]
        predicted_vals = cph.predict_survival_function(new_cph_df, times = time).values[0] #### all values in list of single list, so just dig em out with [0]
        new_model_frame = model_frame
        new_model_frame[predictor] = predicted_vals
        return new_model_frame



#### Things to input into CTC
#### inputs:
def calculate_test_consequences(model_frame: pd.DataFrame,
                                outcome: str,
                                predictor: str,
                                thresholds: list,
                                harm: float,
                                prevalence: float,
                                time: float,
                                time_to_outcome_col: str) -> pd.DataFrame:
    """Computes tpr and fpr from outcome values and a predictor with provided thresholds

    Parameters
    ----------
    model_frame : dataframe
    outcome : string
    predictor : string
    thresholds : string
    prevalence : float
    time : float
    time_to_outcome_col : string

    Returns
    -------
    dataframe
    """

    #### Handle prevalence values

    #### If case-control prevalence:
    if prevalence != None:
        prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)

    #### If binary
    elif not time_to_outcome_col:
        try:
            outcome_values = model_frame[outcome].values.flatten().tolist()
            prevalence_values = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
                thresholds)  #### need list to be as long as len(thresholds)
        except:
            return 'error: binary prevalence'

    #### If survival
    elif time_to_outcome_col:
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(model_frame[time_to_outcome_col], model_frame[outcome] * 1)  # *1 to convert from boolean to int

        prevalence = 1 - kmf.survival_function_at_times(time)
        prevalence = prevalence[1]
        #### Convert survival to risk by doing 1 - x (Figure out why)

        prevalence_values = [prevalence] * len(thresholds)

    #### paths:
    #### 1. survival or binary
    #### 2. if survival and
    n = len(model_frame.index)
    df = pd.DataFrame({'predictor': predictor,
                       'threshold': thresholds,
                       'n': [n] * len(thresholds),
                       'prevalence': prevalence_values})

    count = 0

    # If no time_to_outcome_col, it means binary
    if not time_to_outcome_col:

        true_outcome = model_frame[model_frame[outcome] == True][[predictor]]
        false_outcome = model_frame[model_frame[outcome] == False][[predictor]]
        test_pos_rate = []
        tp_rate = []
        fp_rate = []

        for (threshold, prevalence) in zip(thresholds, prevalence_values):

            count += 1

            #### Debugging try/except

            # test_pos_rate.append(pd.Series(model_frame[predictor] >= threshold).value_counts()[1]/len(model_frame.index))
            # test_pos_rate.append(pd.Series(df_binary[predictor] >= threshold).value_counts()[1]/len(df_binary.index))

            #### Indexing [1] doesn't work w/ value_counts when only index is 0, so [1] gives error, have to try/except so that when [1] doesn't work can input 0

            try:
                test_pos_rate.append(
                    pd.Series(model_frame[predictor] >= threshold).value_counts()[1] / len(model_frame.index))
                # test_pos_rate.append(pd.Series(df_binary[predictor] >= threshold).value_counts()[1]/len(df_binary.index))
            except:
                test_pos_rate.append(0 / len(model_frame.index))

            #             #### Indexing [1] doesn't work w/ value_counts since only 1 index ([0]), so [1] returns an error
            #             #### Have to try/except this so that when indexing doesn't work, can input 0

            try:
                tp_rate.append(
                    pd.Series(true_outcome[predictor] >= threshold).value_counts()[1] / len(true_outcome[predictor]) * (
                        prevalence))
            except KeyError:
                tp_rate.append(0 / len(true_outcome[predictor]) * (prevalence))
            try:
                fp_rate.append(pd.Series(false_outcome[predictor] >= threshold).value_counts()[1] / len(
                    false_outcome[predictor]) * (1 - prevalence))
            except KeyError:
                fp_rate.append(0 / len(false_outcome[predictor]) * (1 - prevalence))

        df['tpr'] = tp_rate
        df['fpr'] = fp_rate
    #### If time_to_outcome_col, then survival
    elif time_to_outcome_col:

        #         true_outcome = model_frame[model_frame[outcome] == True][[predictor]]
        #         false_outcome = model_frame[model_frame[outcome] == False][[predictor]]
        test_pos_rate = []
        risk_rate_among_test_pos = []
        tp_rate = []
        fp_rate = []

        #### For each threshold, get outcomes where risk value is greater than threshold, insert as formula
        for threshold in thresholds:
            #             test_pos_rate.append(pd.Series(model_frame[predictor] >= threshold).value_counts()[1]/len(model_frame.index))

            try:
                test_pos_rate.append(
                    pd.Series(model_frame[predictor] >= threshold).value_counts()[1] / len(model_frame.index))
                # test_pos_rate.append(pd.Series(df_binary[predictor] >= threshold).value_counts()[1]/len(df_binary.index))
            except:
                test_pos_rate.append(0 / len(model_frame.index))

            #### Indexing [1] doesn't work w/ value_counts since only 1 index ([0]), so [1] returns an error
            #### Have to try/except this so that when indexing doesn't work, can input 0

            #### Get risk value, which is kaplan meier output at specified time, or at timepoint right before specified time given there are points after timepoint as well
            #### Input for KM:

            risk_above_thresh_time = model_frame[model_frame[predictor] >= threshold][time_to_outcome_col]
            risk_above_thresh_outcome = model_frame[model_frame[predictor] >= threshold][outcome]

            kmf = lifelines.KaplanMeierFitter()
            try:
                kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
                risk_rate_among_test_pos.append(1 - pd.Series(kmf.survival_function_at_times(time))[1])
            except:

                risk_rate_among_test_pos.append(1)

        df['test_pos_rate'] = test_pos_rate
        df['risk_rate_among_test_pos'] = risk_rate_among_test_pos

        df['tpr'] = df['risk_rate_among_test_pos'] * test_pos_rate
        df['fpr'] = (1 - df['risk_rate_among_test_pos']) * test_pos_rate

    return df


def dca(data: pd.DataFrame,
        outcome: str,
        predictors: list,
        thresh_lo: float,
        thresh_hi: float,
        thresh_step: float,
        harm: dict,
        probabilities: list,
        time: float,
        prevalence: float,
        time_to_outcome_col: str) -> pd.DataFrame:

    '''
    Sequence of events
    1. convert to risk (convert to probabilities)
    2. calculate net benefit
        a. calculate_test_consequences for each predictor
        b. merge all dfs (one for each predictor)
        c. calculate net benefit based on other columns
            i. nb = tpr - thresh / (1 - thresh) * fpr - harm
    '''

    model_frame = data[np.append(outcome, predictors)]

    model_frame['all'] = [1 for i in range(0, len(model_frame.index))]
    model_frame['none'] = [0 for i in range(0, len(model_frame.index))]
    #### If survival, then time_to_outcome_col contains name of col
    #### Otherwise, time_to_outcome_col will not be set, so Binary
    if time_to_outcome_col:
        model_frame[time_to_outcome_col] = data[time_to_outcome_col]

    #### Convert to risk
    #### Convert selected columns to risk scores
    for i in range(0, len(predictors)):
        if probabilities[i]:
            model_frame = convert_to_risk(model_frame, outcome, predictors[i], prevalence, time, time_to_outcome_col)

    #### Prep data, add placeholder for 0 (10e-10), because can't use 0  for DCA, will output incorrect (incorrect?) value
    thresholds = np.arange(thresh_lo, thresh_hi + thresh_step, thresh_step)  # array of values
    thresholds = np.insert(thresholds, 0, 0.1 ** 9)

    covariate_names = [i for i in model_frame.columns if
                       i not in outcome]  # Get names of covariates (if survival, then will still have time_to_outcome_col
    #### If survival, get covariate names that are not time_to_outcome_col
    if time_to_outcome_col:
        covariate_names = [i for i in covariate_names if i not in time_to_outcome_col]

    testcons_list = []
    for covariate in covariate_names:
        temp_testcons_df = calculate_test_consequences(
            model_frame=model_frame,
            outcome=outcome,
            predictor=covariate,
            thresholds=thresholds,
            harm=harm,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

        temp_testcons_df['variable'] = [covariate] * len(temp_testcons_df.index)

        temp_testcons_df['harm'] = [harm[covariate] if harm != None else 0] * len(temp_testcons_df.index)
        testcons_list.append(temp_testcons_df)

    all_covariates_df = pd.concat(testcons_list)

    all_covariates_df['net_benefit'] = all_covariates_df['tpr'] - all_covariates_df['threshold'] / (
                1 - all_covariates_df['threshold']) * all_covariates_df['fpr'] - all_covariates_df['harm']

    return all_covariates_df


def plot_net_benefit_graphs(output_df: pd.DataFrame) -> (list, list):
    predictor_names = output_df['predictor'].value_counts().index
    color_names = ['blue', 'purple', 'red', 'green']
    x_val_list = []
    y_val_list = []

    for predictor_name, color_name in zip(predictor_names, color_names):
        single_pred_df = output_df[output_df['predictor'] == predictor_name]
        x_vals = single_pred_df['threshold']
        y_vals = single_pred_df['net_benefit']

        plt.plot(x_vals, y_vals, color=color_name)

        plt.ylim([-0.05, 0.2])
        plt.legend(predictor_names)
        plt.grid(b=True, which='both', axis='both')

    return