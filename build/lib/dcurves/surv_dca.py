import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
from dcurves import _validate
import beartype


@beartype
def _surv_convert_to_risk(
        model_frame: pd.DataFrame,
        outcome: str,
        predictor: str,
        time: float,
        time_to_outcome_col: str,
        prevalence: float = None
) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1
    # pass
    # _validate._surv_convert_to_risk_input_checks(
    #     prevalence=prevalence,
    #     time=time,
    #     time_to_outcome_col=time_to_outcome_col)

    # From lifelines dataframe
    cph = lifelines.CoxPHFitter()
    cph_df = model_frame[[time_to_outcome_col, outcome, predictor]]
    cph.fit(cph_df, time_to_outcome_col, outcome)

    new_cph_df = cph_df
    new_cph_df[time_to_outcome_col] = [time for i in range(0, len(cph_df))]
    predicted_vals = cph.predict_survival_function(new_cph_df, times=time).values[0]
    #### all values in list of single list, so just dig em out with [0]
    new_model_frame = model_frame
    new_model_frame[predictor] = predicted_vals
    return new_model_frame

@beartype
def _surv_prevalence_calc(
        model_frame: pd.DataFrame,
        outcome: str,
        thresholds: list,
        time: float,
        time_to_outcome_col: str,
        prevalence: object = None
):
    if prevalence != None:
        prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)
    else:
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(model_frame[time_to_outcome_col], model_frame[outcome] * 1)  # *1 to convert from boolean to int
        prevalence = 1 - kmf.survival_function_at_times(time)
        prevalence = prevalence[1]
        #### Convert survival to risk by doing 1 - x (Figure out why)

    return prevalence


@beartype
def _surv_calculate_test_consequences(
        model_frame: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: list,
        time: float,
        time_to_outcome_col: str,
        prevalence: float = None
) -> pd.DataFrame:
    # This function calculates the following and outputs them in a pandas DataFrame
    # For binary evaluation:
    # will calculate [tpr, fpr]
    # For survival evaluation
    # will calculate [tpr, fpr, 'test_pos_rate', risk_rate_among_test_pos]

    _validate._surv_calculate_test_consequences_input_checks(
        thresholds=thresholds,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col
    )


    prevalence_values = \
    _surv_prevalence_calc(
        model_frame=model_frame,
        outcome=outcome,
        thresholds=thresholds,
        time=time
    )

    #### Handle prevalence values

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

    test_pos_rate = []
    risk_rate_among_test_pos = []
    # tp_rate = []
    # fp_rate = []

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


def surv_dca(
        data: object,
        outcome: object,
        predictors: object,
        thresh_vals: object = [0.01, 0.99, 0.01],
        harm: object = None,
        probabilities: object = [False],
        time: object = None,
        prevalence: object = None,
        time_to_outcome_col: object = None
) -> object:

    _validate._surv_dca_input_checks(
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
            model_frame = \
                _surv_convert_to_risk(
                    model_frame=model_frame,
                    outcome=outcome,
                    predictor=predictors[i],
                    prevalence=prevalence,
                    time=time,
                    time_to_outcome_col=time_to_outcome_col
                )

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
        temp_testcons_df = _surv_calculate_test_consequences(
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





surv_dca.__doc__ = """

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


