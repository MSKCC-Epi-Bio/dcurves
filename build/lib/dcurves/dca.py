import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
import matplotlib.pyplot as plt
from dcurves import _validate


def _convert_to_risk(model_frame: pd.DataFrame,
                     outcome: str,
                     predictor: str,
                     prevalence: float = None,
                     time: float = None,
                     time_to_outcome_col: str = None) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    _validate._convert_to_risk_input_checks(model_frame=model_frame,
                                            outcome=outcome,
                                            predictor=predictor,
                                            prevalence=prevalence,
                                            time=time,
                                            time_to_outcome_col=time_to_outcome_col)

    # Binary DCA
    if not time_to_outcome_col:
        predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                        data=model_frame).fit().predict()
        model_frame[predictor] = [(1 - val) for val in predicted_vals]
        # model_frame.loc[model_frame['predictor']]
        return model_frame

    # Survival DCA
    elif time_to_outcome_col:
        #### From lifelines dataframe
        cph = lifelines.CoxPHFitter()
        cph_df = model_frame[['ttcancer', 'cancer', 'cancerpredmarker']]
        cph.fit(cph_df, 'ttcancer', 'cancer')

        new_cph_df = cph_df
        new_cph_df['ttcancer'] = [time for i in range(0, len(cph_df))]
        predicted_vals = cph.predict_survival_function(new_cph_df, times=time).values[
            0]  #### all values in list of single list, so just dig em out with [0]
        new_model_frame = model_frame
        new_model_frame[predictor] = predicted_vals
        return new_model_frame


def _calculate_test_consequences(model_frame: pd.DataFrame,
                                 outcome: str,
                                 predictor: str,
                                 thresholds: list,
                                 prevalence: float = None,
                                 time: float = None,
                                 time_to_outcome_col: str = None) -> pd.DataFrame:
    # This function calculates the following and outputs them in a pandas DataFrame
    # For binary evaluation:
    # will calculate [tpr, fpr]
    # For survival evaluation
    # will calculate [tpr, fpr, 'test_pos_rate', risk_rate_among_test_pos]

    _validate._calculate_test_consequences_input_checks(
        model_frame=model_frame,
        outcome=outcome,
        predictor=predictor,
        thresholds=thresholds,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col
    )

    #### Handle prevalence values

    # If provided - case-control
    # If not provided
    # if not time_to_outcome_col - binary
    # if time_to_outcome_col - survival

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

            #### Indexing [1] doesn't work w/ value_counts since only 1 index ([0]), so [1] returns an error
            #### Have to try/except this so that when indexing doesn't work, can input 0

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

        # For each threshold, get outcomes where risk value is greater than threshold, insert as formula
        for threshold in thresholds:
            # test_pos_rate.append(pd.Series(model_frame[predictor] >= threshold).value_counts()[1]/len(model_frame.index))

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


def dca(data: object,
        outcome: object,
        predictors: object,
        thresh_vals: object = [0.01, 0.99, 0.01],
        harm: object = None,
        probabilities: object = [False],
        time: object = None,
        prevalence: object = None,
        time_to_outcome_col: object = None) -> object:
    """

    Perform Decision Curve Analysis

    |

    Diagnostic and prognostic models are typically evaluated with measures of
    accuracy that do not address clinical consequences.

    |

    Decision-analytic techniques allow assessment of clinical outcomes but often
    require collection of additional information may be cumbersome to apply to
    models that yield a continuous result. Decision curve analysis is a method
    for evaluating and comparing prediction models that incorporates clinical
    consequences, requires only the data set on which the models are tested,
    and can be applied to models that have either continuous or dichotomous
    results.
    The dca function performs decision curve analysis for binary outcomes.

    |

    Review the
    [DCA Vignette](http://www.danieldsjoberg.com/dcurves/articles/dca.html)
    for a detailed walk-through of various applications.

    |

    Also, see [www.decisioncurveanalysis.org]
    (https://www.mskcc.org/departments/epidemiology-biostatistics/biostatistics/decision-curve-analysis) for more information.

    |

    Examples
    ________

    |

    Load simulation binary data dataframe, print contents.

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
    >>> print(df_binary)
         patientid  cancer  ...    marker cancerpredmarker
    0            1   False  ...  0.776309         0.037201
    1            2   False  ...  0.267086         0.578907
    2            3   False  ...  0.169621         0.021551
    3            4   False  ...  0.023996         0.003910
    4            5   False  ...  0.070910         0.018790
    ..         ...     ...  ...       ...              ...
    745        746   False  ...  0.654782         0.057813
    746        747    True  ...  1.030259         0.160424
    747        748   False  ...  0.151616         0.108838
    748        749   False  ...  0.624602         0.015285
    749        750   False  ...  0.270679         0.011938

    [750 rows x 8 columns]

    |

    Run DCA on simulation binary data. Print the results.

    |

    >>> print(
    ...   dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ...    )
    ... )
          predictor     threshold    n  prevalence    tpr       fpr    variable  harm  net_benefit
    0    famhistory  1.000000e-09  750        0.14  0.032  0.121333  famhistory     0     0.032000
    1    famhistory  1.000000e-02  750        0.14  0.032  0.121333  famhistory     0     0.030774
    2    famhistory  2.000000e-02  750        0.14  0.032  0.121333  famhistory     0     0.029524
    3    famhistory  3.000000e-02  750        0.14  0.032  0.121333  famhistory     0     0.028247
    4    famhistory  4.000000e-02  750        0.14  0.032  0.121333  famhistory     0     0.026944
    ..          ...           ...  ...         ...    ...       ...         ...   ...          ...
    96         none  9.600000e-01  750        0.14  0.000  0.000000        none     0     0.000000
    97         none  9.700000e-01  750        0.14  0.000  0.000000        none     0     0.000000
    98         none  9.800000e-01  750        0.14  0.000  0.000000        none     0     0.000000
    99         none  9.900000e-01  750        0.14  0.000  0.000000        none     0     0.000000
    100        none  1.000000e+00  750        0.14  0.000  0.000000        none     0          NaN

    |

    Load simulation survival data and run DCA on it. Print the results.

    |

    >>> df_surv = dcurves.load_test_data.load_survival_df()
    >>> print(
    ...   dcurves.dca(
    ...     data = dcurves.load_test_data.load_survival_df(),
    ...     outcome = 'cancer',
    ...     predictors = ['cancerpredmarker'],
    ...     thresh_vals = [0.01, 1.0, 0.01],
    ...     probabilities = [False],
    ...     time = 1,
    ...     time_to_outcome_col = 'ttcancer'
    ...   )
    ... )
                predictor     threshold    n  prevalence  ...       fpr          variable  harm  net_benefit
    0    cancerpredmarker  1.000000e-09  750    0.147287  ...  0.852713  cancerpredmarker     0     0.147287
    1    cancerpredmarker  1.000000e-02  750    0.147287  ...  0.742181  cancerpredmarker     0     0.139656
    2    cancerpredmarker  2.000000e-02  750    0.147287  ...  0.613444  cancerpredmarker     0     0.132703
    3    cancerpredmarker  3.000000e-02  750    0.147287  ...  0.523820  cancerpredmarker     0     0.123979
    4    cancerpredmarker  4.000000e-02  750    0.147287  ...  0.474956  cancerpredmarker     0     0.115921
    ..                ...           ...  ...         ...  ...       ...               ...   ...          ...
    96               none  9.600000e-01  750    0.147287  ...  0.000000              none     0     0.000000
    97               none  9.700000e-01  750    0.147287  ...  0.000000              none     0     0.000000
    98               none  9.800000e-01  750    0.147287  ...  0.000000              none     0     0.000000
    99               none  9.900000e-01  750    0.147287  ...  0.000000              none     0     0.000000
    100              none  1.000000e+00  750    0.147287  ...  0.000000              none     0          NaN


    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column name of the data frame to use as the outcome
    predictors : str OR list(str)
        the column(s) that will be used to predict the outcome
    thresh_vals : list(float OR int)
        3 values in list - threshold probability lower bound, upper bound,
        then step size, respectively (defaults to [0.01, 1, 0.01]). The lower
        bound must be >0.
    harm : float or list(float)
        the harm associated with each predictor
        harm must have the same length as the predictors list
    probabilities : bool or list(bool)
        whether the outcome is coded as a probability
        probability must have the same length as the predictors list
    time : float (defaults to None)
        survival endpoint time for risk calculation
    prevalence : float (defaults to None)
        population prevalence value
    time_to_outcome_col : str (defaults to None)
        name of input dataframe column that contains time-to-outcome data


    Return
    -------
    all_covariates_df : pd.DataFrame
        A dataframe containing calculated net benefit values and threshold values for plotting

    """

    _validate._dca_input_checks(
        model_frame=data,
        outcome=outcome,
        predictors=predictors,
        thresh_vals=thresh_vals,
        harm=harm,
        probabilities=probabilities,
        time=time,
        prevalence=prevalence,
        time_to_outcome_col=time_to_outcome_col
    )

    # make model_frame df of outcome and predictor cols from data

    model_frame = data[np.append(outcome, predictors)]

    #### If survival, then time_to_outcome_col contains name of col
    #### Otherwise, time_to_outcome_col will not be set (will = None), which means we're doing Binary DCA

    if time_to_outcome_col:
        model_frame[time_to_outcome_col] = data[time_to_outcome_col]

    #### Convert to risk
    #### Convert selected columns to risk scores

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

    # thresh_vals input from user contains 3 values: lower threshold bound, higher threshold
    # bound, step increment in positions [0,1,2]

    # nr.arange takes 3 vals: start, stop + one step increment, and step increment
    thresholds = np.arange(thresh_vals[0], thresh_vals[1] + thresh_vals[2], thresh_vals[2])  # array of values
    #### Prep data, add placeholder for 0 (10e-10), because can't use 0  for DCA, will output incorrect (incorrect?) value
    thresholds = np.insert(thresholds, 0, 0.1 ** 9).tolist()

    covariate_names = [i for i in model_frame.columns if
                       i not in outcome]  # Get names of covariates (if survival, then will still have time_to_outcome_col
    #### If survival, get covariate names that are not time_to_outcome_col
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
