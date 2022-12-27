import pandas as pd
import numpy as np
import statsmodels.api as sm
from dcurves import _validate
from beartype import beartype
from typing import Optional, Union
import lifelines


@beartype
def _convert_to_risk(
        data: pd.DataFrame,
        outcome: str,
        predictors_to_prob: Optional[list] = None,
        time: Optional[Union[float, int]] = None,
        time_to_outcome_col: Optional[str] = None
) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    if predictors_to_prob is None:
        # print('NO PREDICTORS CONVERTED TO PROBABILITIES (BETW. 0 AND 1)')
        pass
    elif time_to_outcome_col is None:
        for predictor in predictors_to_prob:
            # print(predictor + ' CONVERTED TO PROBABILITY (0 to 1)')
            predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                            data=data).fit().predict()
            data[predictor] = [(1 - val) for val in predicted_vals]
    elif time_to_outcome_col is not None:
        for predictor in predictors_to_prob:
            print(predictor + ' CONVERTED TO PROBABILITY (0 to 1)')

            cph_df = data[[time_to_outcome_col, outcome, predictor]]
            # print(cph_df)
            cph = lifelines.CoxPHFitter()
            cph.fit(cph_df, time_to_outcome_col, outcome)
            cph_df[time_to_outcome_col] = [time for i in range(0, len(cph_df))]
            predicted_vals = cph.predict_survival_function(cph_df, times=time).values[0]
            data[predictor] = predicted_vals
    return data

def _calc_prevalences(
        risks_df: pd.DataFrame,
        outcome: str,
        thresholds: np.ndarray,
        prevalence: Optional[Union[int, float]] = None,
        time: Optional[Union[int, float]] = None,
        time_to_outcome_col: Optional[str] = None
):
    # Binary
    if time_to_outcome_col is None:
        if prevalence is not None:
            prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)
        elif prevalence is None:
            outcome_values = risks_df[outcome].values.flatten().tolist()
            prevalence_values = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
                thresholds)
    # Survival
    elif time_to_outcome_col is not None:
        if prevalence is not None:
            prevalence_values = [prevalence] * len(thresholds)
        elif prevalence is None:
            kmf = lifelines.KaplanMeierFitter()
            kmf.fit(risks_df[time_to_outcome_col], risks_df[outcome] * 1)  # *1 to convert from boolean to int
            prevalence = 1 - kmf.survival_function_at_times(time)
            prevalence = prevalence[1]
            prevalence_values = [prevalence] * len(thresholds)
    return prevalence_values

@beartype
def _calc_surv_consequences(
        risks_df: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: np.ndarray,
        prevalence_values: list
):

    test_consequences_df = pd.DataFrame({'predictor': predictor,
                                         'threshold': thresholds,
                                         'n': [len(risks_df.index)] * len(thresholds),
                                         'prevalence': prevalence_values})

    true_outcome = risks_df[risks_df[outcome] == True][[predictor]]
    false_outcome = risks_df[risks_df[outcome] == False][[predictor]]

    test_pos_rate = []
    tp_rate = []
    fp_rate = []

    for (threshold, prevalence) in zip(thresholds, prevalence_values):
        #### Indexing [1] doesn't work w/ value_counts when only index is 0, so [1] gives error, have to try/except
        # so that when [1] doesn't work can input 0
        try:
            test_pos_rate.append(
                pd.Series(risks_df[predictor] >= threshold).value_counts()[1] / len(risks_df.index))
        except KeyError:
            test_pos_rate.append(0 / len(risks_df.index))

        try:
            tp_rate.append(
                pd.Series(true_outcome[predictor] >= threshold).value_counts()[1] / len(true_outcome[predictor]) * (
                    prevalence))
        except KeyError:
            tp_rate.append(0 / len(true_outcome[predictor]) * prevalence)

        try:
            fp_rate.append(pd.Series(false_outcome[predictor] >= threshold).value_counts()[1] / len(
                false_outcome[predictor]) * (1 - prevalence))
        except KeyError:
            fp_rate.append(0 / len(false_outcome[predictor]) * (1 - prevalence))

    test_consequences_df['test_pos_rate'] = test_pos_rate
    test_consequences_df['tpr'] = tp_rate
    test_consequences_df['fpr'] = fp_rate
    test_consequences_df['variable'] = [predictor] * len(test_consequences_df.index)
    test_consequences_df['harm'] = [0 if harm is None
                                    else harm[predictor] if predictor in harm else 0] * len(test_consequences_df.index)
    return test_consequences_df

def _calculate_test_consequences(
        risks_df: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: np.ndarray,
        time_to_outcome_col: str = None,
        time: Union[float, int] = None,
        prevalence: Optional[Union[float, int]] = None,
        harm: Optional[dict] = None) -> pd.DataFrame:

    prevalence_values =\
        _calc_prevalences(
            risks_df=risks_df,
            outcome=outcome,
            thresholds=thresholds,
            prevalence=prevalence,
            time=time,
            time_to_outcome=time_to_outcome_col
        )

    test_consequences_df = pd.DataFrame({'predictor': predictor,
                                         'threshold': thresholds,
                                         'n': [len(risks_df.index)] * len(thresholds),
                                         'prevalence': prevalence_values})

    if time_to_outcome_col is None:


        _calc_surv_consequences(


        )

    return df


def dca(data: object,
        outcome: object,
        predictors: object,
        thresh_vals: object = [0.01, 0.99, 0.01],
        harm: object = None,
        probabilities: object = [False],
        time: object = None,
        prevalence: object = None,
        time_to_outcome_col: object = None) -> object:

    model_frame = data[np.append(outcome, predictors)]

    if time_to_outcome_col:
        model_frame[time_to_outcome_col] = data[time_to_outcome_col]

    for i in range(0, len(predictors)):
        if probabilities[i]:
            model_frame = _convert_to_risk(model_frame,
                                           outcome,
                                           predictors[i],
                                           prevalence,
                                           time,
                                           time_to_outcome_col)

    model_frame['all'] = [1 for i in range(0, len(model_frame.index))]
    model_frame['none'] = [0 for i in range(0, len(model_frame.index))]

    thresholds = np.arange(thresh_vals[0], thresh_vals[1] + thresh_vals[2], thresh_vals[2])  # array of values
    thresholds = np.insert(thresholds, 0, 0.1 ** 9).tolist()

    covariate_names = [i for i in model_frame.columns if
                       i not in outcome]
    if time_to_outcome_col:
        covariate_names = [i for i in covariate_names if i not in time_to_outcome_col]

    testcons_list = []
    for covariate in covariate_names:
        temp_testcons_df = _calculate_test_consequences(
            model_frame=model_frame,
            outcome=outcome,
            predictor=covariate,
            thresholds=thresholds,
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


def net_intervention_avoided(
        after_dca_df: pd.DataFrame,
        nper: int = 100
):
    """

    |

    Calculate net interventions avoided after performing decision curve analysis

    |

    Examples
    ________

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
    after_dca_df : pd.DataFrame
        dataframe outputted by dca function in the dcurves library
    nper : int
        number to report net interventions per （Defaults to 100)

    Return
    ______
    merged_after_dca_df: pd.DataFrame
        dataframe with calculated net_intervention_avoided field joined to the inputted after_dca_df

    """


    all_records = after_dca_df[after_dca_df['variable'] == 'all']
    all_records = all_records[['threshold', 'net_benefit']]
    all_records = all_records.rename(columns={'net_benefit': 'net_benefit_all'})

    merged_after_dca_df = after_dca_df.merge(all_records, on='threshold')

    merged_after_dca_df['net_intervention_avoided'] = (merged_after_dca_df['net_benefit']
                                             - merged_after_dca_df['net_benefit_all']) \
                                            / (merged_after_dca_df['threshold']
                                               / (1 - merged_after_dca_df['threshold'])) * nper

    return merged_after_dca_df


def plot_graphs(after_dca_df: pd.DataFrame,
                graph_type: str = 'net_benefit',
                y_limits: list = [-0.05, 0.2],
                color_names: list = ['blue', 'purple', 'red',
                                     'green', 'hotpink', 'orange',
                                     'saddlebrown', 'lime', 'magenta']
                ) -> None:
    """

    |

    Plot the outputted dataframe from dca() of this library.

    |

    Specifically, this function
    will plot the calculated net benefit values for each threshold value from those
    indicated in the dca() function.

    Examples
    ________

    |

    Simple plot binary DCA example. Load binary outcome dataframe, run DCA, plot calculated predictor net benefit values

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
    >>> binary_dca_results = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )
    >>> plot_graphs(after_dca_df = binary_dca_results)

    |

    Simple binary DCA example with plotting net interventions avoided. Load binary outcome dataframe, run DCA, run
    net_intervention_avoided(), plot outputted dataframe

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
    >>> after_dca_df = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )
    >>> after_net_interventions_avoided_df = net_intervention_avoided(after_dca_df=after_dca_df)
    >>> plot_graphs(after_dca_df = after_net_interventions_avoided_df,
    ...     graph_type='net_intervention_avoided',
    ...     y_limits=[-10,100],
    ...     color_names=['red','teal']
    ... )

    Parameters
    __________
    after_dca_df : pandas.DataFrame
        dataframe outputted by dca function in the dcurves library
    graph_type : str
        type of graph outputted, either 'net_benefit' or 'net_intervention_avoided' (defaults to 'net_benefit')
    y_limits : list[float]
        list of float that control graph lower and upper y-axis limits
        (defaults to [-0.05, 0.2] for graph_type == net_benefit. Change values for
         graph_type == net_intervention_avoided)
    color_names : list[str]
        list of colors specified by user (defaults to ['blue', 'purple', 'red',
        'green', 'hotpink', 'orange', 'saddlebrown', 'lime', 'magenta']

    Returns
    _______
    None

    """

    _validate._plot_graphs_input_checks(after_dca_df=after_dca_df,
                                        y_limits=y_limits,
                                        color_names=color_names,
                                        graph_type=graph_type)

    _validate._plot_net_intervention_input_checks(after_dca_df=after_dca_df,
                                                  graph_type=graph_type)

    if graph_type == 'net_benefit':

        predictor_names = after_dca_df['predictor'].value_counts().index
        # color_names = ['blue', 'purple','red',
        #                'green', 'hotpink', 'orange',
        #                'saddlebrown', 'lime', 'magenta']

        for predictor_name, color_name in zip(predictor_names, color_names):
            single_pred_df = after_dca_df[after_dca_df['predictor'] == predictor_name]
            x_vals = single_pred_df['threshold']
            y_vals = single_pred_df['net_benefit']
            plt.plot(x_vals, y_vals, color=color_name)

            plt.ylim(y_limits)
            plt.legend(predictor_names)
            plt.grid(b=True, which='both', axis='both')
            plt.xlabel('Threshold Values')
            plt.ylabel('Calculated Net Benefit')

    elif graph_type == 'net_intervention_avoided':

        # Don't want to plot 'all'/'none' for net_intervention_avoided

        cleaned_after_dca_df = after_dca_df[~(after_dca_df["predictor"].isin(['all', 'none']))]

        predictor_names = cleaned_after_dca_df['predictor'].value_counts().index

        for predictor_name, color_name in zip(predictor_names, color_names):
            single_pred_df = cleaned_after_dca_df[cleaned_after_dca_df['predictor'] == predictor_name]
            x_vals = single_pred_df['threshold']
            y_vals = single_pred_df['net_intervention_avoided']
            plt.plot(x_vals, y_vals, color=color_name)

            plt.ylim(y_limits)
            plt.legend(predictor_names)
            plt.grid(b=True, which='both', axis='both')
            plt.xlabel('Threshold Values')
            plt.ylabel('Calculated Net Interventions Avoided')

    return
