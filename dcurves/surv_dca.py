import pandas as pd
import numpy as np
import lifelines
from dcurves import _validate
from beartype import beartype
from typing import Optional, Union

@beartype
def _surv_convert_to_risk(
        data: pd.DataFrame,
        outcome: str,
        time: Union[float, int],
        time_to_outcome_col: str,
        predictors_to_prob: Optional[list] = None) -> pd.DataFrame:

    if predictors_to_prob is None:
        pass
        # print('NO PREDICTORS CONVERTED TO PROBABILITIES (BETW. 0 AND 1)')
    else:
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


@beartype
def _surv_calculate_test_consequences(
        risks_df: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: np.ndarray,
        time_to_outcome_col: str,
        time: Union[float, int] = 1,
        prevalence: Optional[Union[float, int]] = None,
        harm: Optional[dict] = None) -> pd.DataFrame:

    if prevalence is not None:
        prevalence_values = [prevalence] * len(thresholds)
    elif prevalence is None:
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(risks_df[time_to_outcome_col], risks_df[outcome] * 1)  # *1 to convert from boolean to int
        prevalence = 1 - kmf.survival_function_at_times(time)
        prevalence = prevalence[1]
        prevalence_values = [prevalence] * len(thresholds)

    machine_epsilon = np.finfo(float).eps

    thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
    thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)

    test_consequences_df = pd.DataFrame({'predictor': predictor,
                                         'threshold': thresholds,
                                         'n': [len(risks_df.index)] * len(thresholds),
                                         'prevalence': prevalence_values})

    test_pos_rate = []
    risk_rate_among_test_pos = []

    for threshold in thresholds:
        try:
            test_pos_rate.append(
                pd.Series(risks_df[predictor] >= threshold).value_counts()[1] / len(risks_df.index))
        except:
            test_pos_rate.append(0 / len(risks_df.index))

        risk_above_thresh_time = risks_df[risks_df[predictor] >= threshold][time_to_outcome_col]
        risk_above_thresh_outcome = risks_df[risks_df[predictor] >= threshold][outcome]

        kmf = lifelines.KaplanMeierFitter()
        try:
            kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
            risk_rate_among_test_pos.append(1 - pd.Series(kmf.survival_function_at_times(time))[1])
        except:
            risk_rate_among_test_pos.append(1)

    test_consequences_df['test_pos_rate'] = test_pos_rate

    test_consequences_df['risk_rate_among_test_pos'] = risk_rate_among_test_pos
    test_consequences_df['tpr'] = test_consequences_df['risk_rate_among_test_pos'] * test_pos_rate
    test_consequences_df['fpr'] = (1 - test_consequences_df['risk_rate_among_test_pos']) * test_pos_rate

    test_consequences_df['harm'] = [0 if harm is None
                                    else harm[predictor] if predictor in harm else 0] * len(test_consequences_df.index)

    return test_consequences_df

@beartype
def surv_dca(
        data: pd.DataFrame,
        outcome: str,
        predictors: list,
        time_to_outcome_col: str,
        thresholds: np.ndarray = np.linspace(0.00, 1.00, 101),
        harm: Optional[dict] = None,
        predictors_to_prob: Optional[list] = None,
        prevalence: Optional[Union[float, int]] = None,
        time: Optional[Union[float, int]] = 1.0
) -> pd.DataFrame:

    vars_to_risk_df = \
        _surv_convert_to_risk(
            data=data,
            outcome=outcome,
            predictors_to_prob=predictors_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    covariate_names = np.append(predictors, ['all', 'none'])

    test_consequences_df = \
        pd.concat([_surv_calculate_test_consequences(
            risks_df=vars_to_risk_df,
            outcome=outcome,
            predictor=predictor,
            thresholds=thresholds,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col,
            harm=harm) for predictor in covariate_names])

    test_consequences_df['neg_rate'] = 1 - test_consequences_df['prevalence']
    test_consequences_df['fnr'] = test_consequences_df['prevalence'] - test_consequences_df['tpr']
    test_consequences_df['tnr'] = test_consequences_df['neg_rate'] - test_consequences_df['fpr']

    test_consequences_df['net_benefit'] = test_consequences_df['tpr'] - test_consequences_df['threshold'] / (
            1 - test_consequences_df['threshold']) * test_consequences_df['fpr'] - test_consequences_df['harm']
    test_consequences_df['test_neg_rate'] = test_consequences_df['fnr'] + test_consequences_df['tnr']
    test_consequences_df['ppv'] = test_consequences_df['tpr'] / \
                                  (test_consequences_df['tpr'] + test_consequences_df['fpr'])
    test_consequences_df['npv'] = test_consequences_df['tnr'] / \
                                  (test_consequences_df['tnr'] + test_consequences_df['fnr'])
    test_consequences_df['sens'] = test_consequences_df['tpr'] / \
                                   (test_consequences_df['tpr'] + test_consequences_df['fnr'])
    test_consequences_df['spec'] = test_consequences_df['tnr'] / \
                                   (test_consequences_df['tnr'] + test_consequences_df['fpr'])
    test_consequences_df['lr_pos'] = test_consequences_df['sens'] / \
                                     (1 - test_consequences_df['spec'])
    test_consequences_df['lr_neg'] = (1 - test_consequences_df['sens']) / \
                                     test_consequences_df['spec']

    return test_consequences_df

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


