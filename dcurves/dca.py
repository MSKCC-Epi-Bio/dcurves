import pandas as pd
import numpy as np
import statsmodels.api as sm
from dcurves import _validate
from beartype import beartype
from typing import Optional, Union
import lifelines
import matplotlib.pyplot as plt

@beartype
def _create_risks_df(
        data: pd.DataFrame,
        outcome: str,
        models_to_prob: Optional[list] = None,
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None
) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    if models_to_prob is None:
        # print('NO PREDICTORS CONVERTED TO PROBABILITIES (BETW. 0 AND 1)')
        pass
    elif time_to_outcome_col is None:
        for predictor in models_to_prob:
            # print(predictor + ' CONVERTED TO PROBABILITY (0 to 1)')
            predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                            data=data).fit().predict()
            data[predictor] = [(1 - val) for val in predicted_vals]
    elif time_to_outcome_col is not None:
        for predictor in models_to_prob:
            print(predictor + ' CONVERTED TO PROBABILITY (0 to 1)')

            cph_df = data[[time_to_outcome_col, outcome, predictor]]
            # print(cph_df)
            cph = lifelines.CoxPHFitter()
            cph.fit(cph_df, time_to_outcome_col, outcome)
            cph_df[time_to_outcome_col] = [time for i in range(0, len(cph_df))]
            predicted_vals = cph.predict_survival_function(cph_df, times=time).values[0]
            data[predictor] = predicted_vals

    machine_epsilon = np.finfo(float).eps

    data['all'] = [1 - machine_epsilon for i in range(0, len(data.index))]
    data['none'] = [0 + machine_epsilon for i in range(0, len(data.index))]

    return data

def _calc_prevalence(
        risks_df: pd.DataFrame,
        outcome: str,
        prevalence: Optional[Union[int, float]] = None,
        time: Optional[Union[int, float]] = None,
        time_to_outcome_col: Optional[str] = None
):
    # Binary
    if time_to_outcome_col is None:
        if prevalence is not None:
            pass
        elif prevalence is None:
            outcome_values = risks_df[outcome].values.flatten().tolist()
            prevalence = pd.Series(outcome_values).value_counts()[1] / len(outcome_values)
    # Survival
    elif time_to_outcome_col is not None:
        if prevalence is not None:
            pass
        elif prevalence is None:
            kmf = lifelines.KaplanMeierFitter()
            kmf.fit(risks_df[time_to_outcome_col], risks_df[outcome] * 1)  # *1 to convert from boolean to int
            prevalence = 1 - kmf.survival_function_at_times(time)
            prevalence = prevalence[1]
    return prevalence

@beartype
def _create_initial_df(
        thresholds: np.ndarray,
        modelnames: list,
        input_df_rownum: int,
        prevalence_value: Union[float, int],
        harm: Optional[dict] = None
) -> pd.DataFrame:

    machine_epsilon = np.finfo(float).eps
    thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
    thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)

    modelnames = np.append(modelnames, ['all', 'none'])
    initial_df = pd.DataFrame(
        {'model':
             pd.Series([x for y in modelnames for x in [y] * len(thresholds)]),
         'threshold': thresholds.tolist() * len(modelnames),
         'n': [input_df_rownum] * len(thresholds) * len(modelnames),
         'prevalence': [prevalence_value] * len(thresholds) * len(modelnames),
         'harm': 0
         }
    )
    if harm is not None:
        for model in harm.keys():
            initial_df.loc[initial_df['model'] == model, 'harm'] = harm[model]
    elif harm is None:
        pass
    else:
        ValueError('Harm should be either None or dict')

    return initial_df

def _calc_test_pos_rate(
        risks_df: pd.DataFrame,
        thresholds: np.ndarray,
        model: str
):
    test_pos_rate = []

    for threshold in thresholds:
        try:
            test_pos_rate.append(
                pd.Series(risks_df[model] >= threshold).value_counts()[1] / len(risks_df.index))
        except KeyError:
            test_pos_rate.append(0 / len(risks_df.index))

    return test_pos_rate

def _calc_tp_rate(
        risks_df: pd.DataFrame,
        thresholds: np.ndarray,
        model: str,
        outcome: str,
        time: Union[float, int],
        time_to_outcome_col: str,
        test_pos_rate: list,
        prevalence_value: Union[float, int]
):
    # Survival
    if 'time' in risks_df.columns:
        risk_rate_among_test_pos = []
        for threshold in thresholds:
            risk_above_thresh_time = risks_df[risks_df[model] >= threshold][time_to_outcome_col]
            risk_above_thresh_outcome = risks_df[risks_df[model] >= threshold][outcome]

            kmf = lifelines.KaplanMeierFitter()

            try:
                kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
                risk_rate_among_test_pos.append(1 - pd.Series(kmf.survival_function_at_times(time))[1])
            except:
                risk_rate_among_test_pos.append(1)
        tp_rate = risk_rate_among_test_pos * test_pos_rate

    # Binary
    elif 'time' not in risks_df.columns:
        true_outcome = risks_df[risks_df[outcome] == True][[model]]

        tp_rate = []

        for threshold in thresholds:
            try:
                tp_rate.append(
                    pd.Series(true_outcome[model] >= threshold).value_counts()[1] / len(true_outcome[model]) * (
                        prevalence_value))
            except KeyError:
                tp_rate.append(0 / len(true_outcome[model]) * prevalence_value)
    return tp_rate

def _calc_fp_rate(
        risks_df: pd.DataFrame,
        thresholds: np.ndarray,
        model: str,
        outcome: str,
        time: Union[float, int],
        time_to_outcome_col: str,
        test_pos_rate: list,
        prevalence_value: Union[float, int]
):
    # Survival
    if 'time' in risks_df.columns:
        risk_rate_among_test_pos = []
        for threshold in thresholds:
            risk_above_thresh_time = risks_df[risks_df[model] >= threshold][time_to_outcome_col]
            risk_above_thresh_outcome = risks_df[risks_df[model] >= threshold][outcome]

            kmf = lifelines.KaplanMeierFitter()

            try:
                kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
                risk_rate_among_test_pos.append(1 - pd.Series(kmf.survival_function_at_times(time))[1])
            except:
                risk_rate_among_test_pos.append(1)
        fp_rate = (1 - risk_rate_among_test_pos) * test_pos_rate
    # Binary
    elif 'time' not in risks_df.columns:
        false_outcome = risks_df[risks_df[outcome] == False][[model]]

        fp_rate = []

        for threshold in thresholds:
            try:
                fp_rate.append(pd.Series(false_outcome[model] >= threshold).value_counts()[1] / len(
                    false_outcome[model]) * (1 - prevalence_value))
            except KeyError:
                fp_rate.append(0 / len(false_outcome[model]) * (1 - prevalence_value))
    return fp_rate

@beartype
def _calc_modelspecific_stats(
        initial_df: pd.DataFrame,
        risks_df: pd.DataFrame,
        thresholds: np.ndarray,
        outcome: str,
        prevalence_value: Union[float, int],
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None
):
    for model in initial_df['model'].value_counts().index:
        test_pos_rate = _calc_test_pos_rate(risks_df=risks_df,
                                            thresholds=thresholds,
                                            model=model)
        tp_rate = \
            _calc_tp_rate(
                risks_df=risks_df,
                thresholds=thresholds,
                model=model,
                outcome=outcome,
                time=time,
                time_to_outcome_col=time_to_outcome_col,
                test_pos_rate=test_pos_rate,
                prevalence_value=prevalence_value
            )

        fp_rate = \
            _calc_fp_rate(
                risks_df=risks_df,
                thresholds=thresholds,
                model=model,
                outcome=outcome,
                time=time,
                time_to_outcome_col=time_to_outcome_col,
                test_pos_rate=test_pos_rate,
                prevalence_value=prevalence_value
            )

        initial_df.loc[initial_df['model'] == model, 'test_pos_rate'] = test_pos_rate
        initial_df.loc[initial_df['model'] == model, 'tp_rate'] = tp_rate
        initial_df.loc[initial_df['model'] == model, 'fp_rate'] = fp_rate
    return initial_df

def _calc_nonspecific_stats(
        initial_stats_df: pd.DataFrame
):

    initial_stats_df['neg_rate'] = 1 - initial_stats_df['prevalence']
    initial_stats_df['fn_rate'] = initial_stats_df['prevalence'] - initial_stats_df['tp_rate']
    initial_stats_df['tn_rate'] = initial_stats_df['neg_rate'] - initial_stats_df['fp_rate']

    initial_stats_df['net_benefit'] = initial_stats_df['tp_rate'] - initial_stats_df['threshold'] / (
            1 - initial_stats_df['threshold']) * initial_stats_df['fp_rate'] - initial_stats_df['harm']
    initial_stats_df['test_neg_rate'] = initial_stats_df['fn_rate'] + initial_stats_df['tn_rate']
    initial_stats_df['ppv'] = initial_stats_df['tp_rate'] /\
                                  (initial_stats_df['tp_rate'] + initial_stats_df['fp_rate'])
    initial_stats_df['npv'] = initial_stats_df['tn_rate'] /\
                                  (initial_stats_df['tn_rate'] + initial_stats_df['fn_rate'])
    initial_stats_df['sens'] = initial_stats_df['tp_rate'] /\
                                   (initial_stats_df['tp_rate'] + initial_stats_df['fn_rate'])
    initial_stats_df['spec'] = initial_stats_df['tn_rate'] /\
                                   (initial_stats_df['tn_rate'] + initial_stats_df['fp_rate'])
    initial_stats_df['lr_pos'] = initial_stats_df['sens'] /\
                                     (1 - initial_stats_df['spec'])
    initial_stats_df['lr_neg'] = (1 - initial_stats_df['sens']) /\
                                     initial_stats_df['spec']

    final_dca_df = initial_stats_df

    return final_dca_df

def dca(data: pd.DataFrame,
        outcome: str,
        modelnames: list,
        thresholds: np.ndarray = np.linspace(0.00, 1.00, 101),
        harm: Optional[dict] = None,
        models_to_prob: Optional[list] = None,
        prevalence: Optional[Union[float, int]] = None,
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None) -> object:

    # 1. Perform checks on inputs to see if user is high or not
    # check_inputs(...)

    # 2. Convert requested columns to risk scores 0 to 1

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

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

    # 5. Calculate model-specific consequences

    initial_stats_df = \
        _calc_modelspecific_stats(
            initial_df=initial_df,
            risks_df=risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # 6. Generate DCA-ready df with full list of calculated statistics
    final_dca_df = \
        _calc_nonspecific_stats(
            initial_stats_df=initial_stats_df
        )

    return final_dca_df

def net_intervention_avoided(
        after_dca_df: pd.DataFrame,
        nper: int = 100
):
    """

    Calculate net interventions avoided after using the dca function

    Parameters
    ----------
    after_dca_df : pd.DataFrame
        dataframe outputted by dca function in the dcurves library
    nper : int
        number to report net interventions per （Defaults to 100)

    Examples
    ________

    >>> import
    >>> df_binary = dcurves.load_test_data.load_binary_df()

    >>> after_dca_df = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )

    >>> after_net_intervention_avoided_df = dcurves.net_intervention_avoided(
    ... after_dca_df = after_dca_df,
    ... nper = 100
    ...)

    |

    Parameters
    __________


    :returns

    merged_after_dca_df: pd.DataFrame
        dataframe with calculated net_intervention_avoided field joined to the inputted after_dca_df

    """

    all_records = after_dca_df[after_dca_df['model'] == 'all']
    all_records = all_records[['threshold', 'net_benefit']]
    all_records = all_records.rename(columns={'net_benefit': 'net_benefit_all'})

    merged_after_dca_df = after_dca_df.merge(all_records, on='threshold')

    merged_after_dca_df['net_intervention_avoided'] = \
        (merged_after_dca_df['net_benefit']
         - merged_after_dca_df['net_benefit_all']) \
        / (merged_after_dca_df['threshold']
            / (1 - merged_after_dca_df['threshold'])) * nper

    return merged_after_dca_df



