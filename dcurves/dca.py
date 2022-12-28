import pandas as pd
import numpy as np
import statsmodels.api as sm
from dcurves import _validate
from beartype import beartype
from typing import Optional, Union
import lifelines


@beartype
def _create_risks_df(
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

    machine_epsilon = np.finfo(float).eps

    data['all'] = [1 - machine_epsilon for i in range(0, len(data.index))]
    data['none'] = [0 + machine_epsilon for i in range(0, len(data.index))]

    return data

def _calc_prevalence(
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
            pass
        elif prevalence is None:
            outcome_values = risks_df[outcome].values.flatten().tolist()
            prevalence = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
                thresholds)
    # Survival
    elif time_to_outcome_col is not None:
        if prevalence is not None:
            pass
        elif prevalence is None:
            kmf = lifelines.KaplanMeierFitter()
            kmf.fit(risks_df[time_to_outcome_col], risks_df[outcome] * 1)  # *1 to convert from boolean to int
            prevalence = 1 - kmf.survival_function_at_times(time)
            prevalence = prevalence[1]
            prevalence = [prevalence] * len(thresholds)
    return prevalence

@beartype
def _calc_predictor_consequences(
        risks_df: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: np.ndarray,
        prevalence_values: list,
        harm: Optional[dict] = None
) -> pd.DataFrame:

    machine_epsilon = np.finfo(float).eps
    thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
    thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)

    test_consequences_df = pd.DataFrame({'predictor': predictor,
                                         'threshold': thresholds,
                                         'n': [len(risks_df.index)] * len(thresholds),
                                         'prevalence': prevalence_values,
                                         'harm': harm[predictor]})

    true_outcome = risks_df[risks_df[outcome] == True][[predictor]]
    false_outcome = risks_df[risks_df[outcome] == False][[predictor]]
    test_pos_rate = []
    tp_rate = []
    fp_rate = []



    for (threshold, prevalence) in zip(thresholds, prevalence_values):
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
                                         'prevalence': prevalence_values,
                                         'harm': harm[predictor]})





    return


def dca(data: pd.DataFrame,
        outcome: str,
        predictors: list,
        time_to_outcome_col: str,
        thresholds: np.ndarray = np.linspace(0.00, 1.00, 101),
        harm: Optional[dict] = None,
        predictors_to_prob: Optional[list] = None,
        prevalence: Optional[Union[float, int]] = None,
        time: Optional[Union[float, int]] = None) -> object:

    # 1. Perform checks on inputs to see if user is high or not
    # check_inputs(...)


    # 2. Convert requested columns to risk scores 0 to 1

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            predictors_to_prob=predictors_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    prevalence_values = \
        _calc_prevalence(
            risks_df=risks_df,
            outcome=outcome,
            thresholds=thresholds,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    test_consequences_df = pd.DataFrame()

    test_consequences_df['predictor'] = pd.Series([[s]*len(thresholds) ])

    test_consequences_df = \
        pd.DataFrame(
            {
                'predictor': pd.Series()
            }
        )

    test_consequences_df = pd.DataFrame({'predictor': predictor,
                                         'threshold': thresholds,
                                         'n': [len(risks_df.index)] * len(thresholds),
                                         'prevalence': prevalence_values,
                                         'harm': harm[predictor]})


    covariate_names = np.append(predictors, ['all', 'none'])

    # 3. Calculate model-specific consequences

    test_consequences_df = \
        pd.concat([_calc_predictor_consequences(
            risks_df=vars_to_risk_df,
            outcome=outcome,
            predictor=predictor,
            thresholds=thresholds,
            prevalence_values=prevalence_values,
            time=time,
            time_to_outcome_col=time_to_outcome_col,
            harm=harm) for predictor in covariate_names])

    # 4. Calculate non-variable-specific consequences
    # calc

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


    all_records = after_dca_df[after_dca_df['predictor'] == 'all']
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
