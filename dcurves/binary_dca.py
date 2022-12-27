import pandas as pd
import numpy as np
import statsmodels.api as sm
from dcurves import _validate
from beartype import beartype
from typing import Optional, Union

# 221112 SP: Go back and fix prevalence type hinting
# Issue is that while beartype input validation decorator should be a float,
# we also need to allow for input of None in case user wants the default calculated prevalence value
# In short, issue is that prevalence needs to have 2 type hints and beartype has to accept either as valid input

# For now, set prev. type hint to object to bypass issue
@beartype
def _binary_convert_to_risk(
        data: pd.DataFrame,
        outcome: str,
        predictors_to_prob: Optional[list] = None
) -> pd.DataFrame:
    # Converts indicated predictor columns in dataframe into probabilities from 0 to 1

    if predictors_to_prob is None:
        # print('NO PREDICTORS CONVERTED TO PROBABILITIES (BETW. 0 AND 1)')
        pass
    else:
        for predictor in predictors_to_prob:
            # print(predictor + ' CONVERTED TO PROBABILITY (0 to 1)')
            predicted_vals = sm.formula.glm(outcome + '~' + predictor, family=sm.families.Binomial(),
                                            data=data).fit().predict()
            data[predictor] = [(1 - val) for val in predicted_vals]
    return data




# 221113 SP: Go back and fix prevalence type hinting
# Issue is that while beartype input validation decorator should be a float,
# we also need to allow for input of None in case user wants the default calculated prevalence value
# In short, issue is that prevalence needs to have 2 type hints and beartype has to accept either as valid input

# For now, set prev. type hint to object to bypass issue

@beartype
def _binary_calculate_test_consequences(
        risks_df: pd.DataFrame,
        outcome: str,
        predictor: str,
        thresholds: np.ndarray,
        prevalence: Optional[Union[float, int]] = None,
        harm: Optional[dict] = None) -> pd.DataFrame:

    if prevalence is not None:
        prevalence_values = [prevalence] * len(thresholds)  #### need list to be as long as len(thresholds)
    elif prevalence is None:
        outcome_values = risks_df[outcome].values.flatten().tolist()
        prevalence_values = [pd.Series(outcome_values).value_counts()[1] / len(outcome_values)] * len(
            thresholds)

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

        #### Indexing [1] doesn't work w/ value_counts when only index is 0, so [1] gives error, have to try/except so that when [1] doesn't work can input 0

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

@beartype
def binary_dca(
        data: pd.DataFrame,
        outcome: str,
        predictors: list,
        thresholds: np.ndarray = np.linspace(0.00, 1.00, 101),
        predictors_to_prob: Optional[list] = None,
        harm: Optional[dict] = None,
        prevalence: Optional[Union[float, int]] = None) -> pd.DataFrame:

    vars_to_risk_df = \
        _binary_convert_to_risk(
            data=data,
            outcome=outcome,
            predictors_to_prob=predictors_to_prob
        )

    machine_epsilon = np.finfo(float).eps

    vars_to_risk_df['all'] = [1 - machine_epsilon for i in range(0, len(vars_to_risk_df.index))]
    vars_to_risk_df['none'] = [0 + machine_epsilon for i in range(0, len(vars_to_risk_df.index))]

    thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
    thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)

    covariate_names = np.append(predictors, ['all', 'none'])

    test_consequences_df = \
        pd.concat([_binary_calculate_test_consequences(
                        risks_df=vars_to_risk_df,
                        outcome=outcome,
                        predictor=predictor,
                        thresholds=thresholds,
                        prevalence=prevalence,
                        harm=harm) for predictor in covariate_names])


    test_consequences_df['neg_rate'] = 1 - test_consequences_df['prevalence']
    test_consequences_df['fnr'] = test_consequences_df['prevalence'] - test_consequences_df['tpr']
    test_consequences_df['tnr'] = test_consequences_df['neg_rate'] - test_consequences_df['fpr']

    test_consequences_df['net_benefit'] = test_consequences_df['tpr'] - test_consequences_df['threshold'] / (
            1 - test_consequences_df['threshold']) * test_consequences_df['fpr'] - test_consequences_df['harm']

    test_consequences_df['test_neg_rate'] = test_consequences_df['fnr'] + test_consequences_df['tnr']

    test_consequences_df['ppv'] = test_consequences_df['tpr'] /\
                                  (test_consequences_df['tpr'] + test_consequences_df['fpr'])

    test_consequences_df['npv'] = test_consequences_df['tnr'] /\
                                  (test_consequences_df['tnr'] + test_consequences_df['fnr'])

    test_consequences_df['sens'] = test_consequences_df['tpr'] /\
                                   (test_consequences_df['tpr'] + test_consequences_df['fnr'])

    test_consequences_df['spec'] = test_consequences_df['tnr'] /\
                                   (test_consequences_df['tnr'] + test_consequences_df['fpr'])

    test_consequences_df['lr_pos'] = test_consequences_df['sens'] /\
                                     (1 - test_consequences_df['spec'])

    test_consequences_df['lr_neg'] = (1 - test_consequences_df['sens']) /\
                                     test_consequences_df['spec']

    return test_consequences_df

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
    
    from dcurves.binary_dca import binary_dca, surv_dca
    

    |

    Load simulation binary data dataframe, print contents.

    |

    >>> df_binary = dcurves.load_test_data.load_binary_df()
 
    |

    Run DCA on simulation binary data. Print the results.

    |

    >>> bin_dca_result_df = \
    ...   dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ...    )

    |

    Load simulation survival data and run DCA on it. Print the results.

    |
    
    >>> df_surv = load


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



