import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
import matplotlib.pyplot as plt
from dcurves import _validate
from beartype import beartype

# 221112 SP: Go back and fix prevalence type hinting
# Issue is that while beartype input validation decorator should be a float,
# we also need to allow for input of None in case user wants the default calculated prevalence value
# In short, issue is that prevalence needs to have 2 type hints and beartype has to accept either as valid input

# For now, set prev. type hint to object to bypass issue
@beartype
def _binary_convert_to_risk(
        model_frame: pd.DataFrame,
        outcome: str,
        predictor: str,
        prevalence: object = None
) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    _validate._binary_convert_to_risk_input_checks(
        prevalence=prevalence)

    predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                    data=model_frame).fit().predict()
    model_frame[predictor] = [(1 - val) for val in predicted_vals]
    # model_frame.loc[model_frame['predictor']]
    return model_frame


@beartype
def _binary_prevalence_calc(
        model_frame: pd.DataFrame,
        outcome: str,
        thresholds: list,
        prevalence: object = None
):

    if prevalence != None:
        prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)
    else:
        outcome_values = model_frame[outcome].values.flatten().tolist()
        prevalence_values = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
            thresholds)  #### need list to be as long as len(thresholds)

    return prevalence_values


# 221113 SP: Go back and fix prevalence type hinting
# Issue is that while beartype input validation decorator should be a float,
# we also need to allow for input of None in case user wants the default calculated prevalence value
# In short, issue is that prevalence needs to have 2 type hints and beartype has to accept either as valid input

# For now, set prev. type hint to object to bypass issue

@beartype
def _binary_calculate_test_consequences(
        model_frame: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: list,
        prevalence: object = None) -> pd.DataFrame:
    # This function calculates the following and outputs them in a pandas DataFrame
    # For binary evaluation:
    # will calculate [tpr, fpr]

    _validate._binary_calculate_test_consequences_input_checks(
        thresholds=thresholds,
        prevalence=prevalence
    )

    #### Handle prevalence values

    # If provided - case-control
    # If not provided
    # if not time_to_outcome_col - binary
    # if time_to_outcome_col - survival

    #### If case-control prevalence:
    if prevalence != None:
        prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)
    else:
        outcome_values = model_frame[outcome].values.flatten().tolist()
        prevalence_values = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
            thresholds)  #### need list to be as long as len(thresholds)


    n = len(model_frame.index)
    df = pd.DataFrame({'predictor': predictor,
                       'threshold': thresholds,
                       'n': [n] * len(thresholds),
                       'prevalence': prevalence_values})

    count = 0

    # If no time_to_outcome_col, it means binary

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
            tp_rate.append(0 / len(true_outcome[predictor]) * prevalence)

        try:
            fp_rate.append(pd.Series(false_outcome[predictor] >= threshold).value_counts()[1] / len(
                false_outcome[predictor]) * (1 - prevalence))
        except KeyError:
            fp_rate.append(0 / len(false_outcome[predictor]) * (1 - prevalence))

    df['test_pos_rate'] = test_pos_rate
    df['tpr'] = tp_rate
    df['fpr'] = fp_rate

    return df

@beartype
def binary_dca(
        data: object,
        outcome: object,
        predictors: object,
        thresh_vals: object = [0.01, 0.99, 0.01],
        harm: object = None,
        probabilities: object = [False],
        prevalence: object = None) -> object:

    _validate._binary_dca_input_checks(
        predictors=predictors,
        thresh_vals=thresh_vals,
        harm=harm,
        probabilities=probabilities,
        prevalence=prevalence
    )

    # make model_frame df of outcome and predictor cols from data

    model_frame = data[np.append(outcome, predictors)]

    #### Convert to risk
    #### Convert selected columns to risk scores

    for i in range(0, len(predictors)):
        if probabilities[i]:
            model_frame = \
                _binary_convert_to_risk(
                    model_frame,
                    outcome,
                    predictors[i],
                    prevalence
                )

    model_frame['all'] = [1 for i in range(0, len(model_frame.index))]
    model_frame['none'] = [0 for i in range(0, len(model_frame.index))]

    # thresh_vals input from user contains 3 values: lower threshold bound, higher threshold
    # bound, step increment in positions [0,1,2]

    # nr.arange takes 3 vals: start, stop + one step increment, and step increment
    thresholds = np.arange(thresh_vals[0], thresh_vals[1] + thresh_vals[2], thresh_vals[2])  # array of values
    #### Prep data, add placeholder for 0 (10e-10), because can't use 0  for DCA, will output incorrect (incorrect?) value
    thresholds = np.insert(thresholds, 0, 0.1 ** 9).tolist()

    # Get names of covariates (if survival, then will still have time_to_outcome_col
    covariate_names = [i for i in model_frame.columns if i not in outcome]

    testcons_list = []
    for covariate in covariate_names:
        temp_testcons_df = _binary_calculate_test_consequences(
            model_frame=model_frame,
            outcome=outcome,
            predictor=covariate,
            thresholds=thresholds,
            prevalence=prevalence
        )

        temp_testcons_df['variable'] = [covariate] * len(temp_testcons_df.index)

        temp_testcons_df['harm'] = [harm[covariate] if harm != None else 0] * len(temp_testcons_df.index)
        testcons_list.append(temp_testcons_df)

    all_covariates_df = pd.concat(testcons_list)

    all_covariates_df['net_benefit'] = all_covariates_df['tpr'] - all_covariates_df['threshold'] / (
            1 - all_covariates_df['threshold']) * all_covariates_df['fpr'] - all_covariates_df['harm']

    return all_covariates_df




binary_dca.__doc__ = """

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
        whether the predictor is coded as a probability
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



