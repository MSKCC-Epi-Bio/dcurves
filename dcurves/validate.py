import operator as opr
import pandas as pd
import statsmodels.api as sm


def data_validate(data):
    """Validates the input data by dropping any incomplete cases

    Parameters
    ----------
    data : pd.DataFrame
        the data set under analysis

    Returns
    -------
    pd.DataFrame
        the passed in data where any rows with a NaN value are dropped

    Raises
    ------
    TypeError
        if `data` is not a pandas DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    return data.dropna(axis=0)


def outcome_validate(data, outcome):
    """Validates that specified outcome is coded 0/1 and does not have
    any values out of that range

    Parameters
    ----------
    data : pd.DataFrame
        the data set under analysis
    outcome : str
        the column of the data set to use as the outcome

    Returns
    -------
    str
        the outcome string passed in

    Raises
    ------
    ValueError
        if a value, 'x', in the outcome column is not in range 0 <= x <= 1
    DCAError
        if a the specified `outcome` is not in `data`
    """
    try:
        if (max(data[outcome]) > 1) or (min(data[outcome]) < 0):
            raise ValueError("all outcome values must be in range 0-1")
    except KeyError:
        raise DCAError("outcome must be a column in the dataframe")

    return outcome


def predictors_validate(predictors, data=None):
    """Validates the predictors and ensures that they are type list(str)

    Optionally checks that the predictors are columns in the data set. Only
    performs this check if the data parameter is not None

    Parameters
    ----------
    predictors: list(str) or str
        the predictor(s) to validate
    data : pd.DataFrame or None, optional
        the data set to validate the predictors are in

    Returns
    -------
    list(str)
        validated predictors

    Raises
    ------
    ValueError
        if a predictor is named 'all' or 'none'
        if a predictor is not a column in the data set

    Examples
    --------
    >>> predictors_validate('famhistory')
    ['famhistory']
    >>> predictors_validate(['famhistory', 'marker'])
    ['famhistory', 'marker']
    >>> predictors_validate('all')
    Traceback (most recent call last):
      ...
    ValueError: predictor cannot be named 'all' or 'none'
    """
    if isinstance(predictors, str):  # single predictor
        predictors = [predictors]  # convert to list

    # cant't use 'all' or 'none' columns as predictors
    for predictor in predictors:
        if predictor in ['all', 'none']:
            raise ValueError("predictor cannot be named 'all' or 'none'")

    # check that predictors are columns in the data
    if data is not None:
        for predictor in predictors:
            if predictor not in data.columns:
                raise ValueError("predictor must be a column in the dataframe")
    else:
        pass  # skip check

    return predictors


def threshold_validate(bound, value, curr_bounds):
    """Validates the value for the given boundary against set min/max values
    and against the current boundaries, if applicable

    Parameters
    ----------
    bound : str
        the bound to validate, valid values are `lower`, `upper`, `step`
    value : float
        the value of the bound being set
    curr_bounds : list(float)
        the current boundaries, `[lower, upper, step]`

    Returns
    -------
    float
        the value passed in, if it is valid

    Raises
    ------
    ValueError
        if the specified value is not valid for the specified bound
    DCAError
        if the specified `bound` is not a valid bound

    Examples
    --------
    >>> threshold_validate('upper', 0.5, [0.01, 0.99, 0.01])
    0.5
    >>> threshold_validate('lower', 0.5, [0.4, 0.44, 0.01])
    Traceback (most recent call last):
      ...
    ValueError
    >>> threshold_validate('step', -0.01, [0.01, 0.99, 0.01])
    Traceback (most recent call last):
      ...
    ValueError
    >>> threshold_validate('not_a_bound', 0.01, [0.01, 0.99, 0.01])
    Traceback (most recent call last):
      ...
    DCAError: did not specify a valid bound, valid values are 'lower', 'upper', 'step'
    """
    bound = bound.lower()
    mapping = {'upper': [(opr.lt, 1), (opr.gt, curr_bounds[0])],
               'lower': [(opr.gt, 0), (opr.lt, curr_bounds[1])],
               'step': [(opr.gt, 0), (opr.lt, curr_bounds[1])]
               }
    try:
        if mapping[bound][0][0](value, mapping[bound][0][1]):
            if mapping[bound][1][0](value, mapping[bound][1][1]):
                return value
            else:
                # TODO: this error msg
                raise ValueError
        else:
            # TODO: this error msg
            raise ValueError
    except KeyError:
        raise DCAError("did not specify a valid bound, valid values are 'lower', 'upper', 'step'")


def probabilities_validate(probabilities, predictors):
    """Validates that the probability list is valid for the current predictors

    Parameters
    ----------
    probabilities : list(bool) or None
        the probability list to be validated
    predictors : list(str)
        the current predictors for the analysis

    Returns
    -------
    list(bool)
         the probability list that was passed in, if it is valid

    Raises
    ------
    TypeError
        if not all of the values in the probability list are booleans
    DCAError
        if the length of the probability list doesn't match that of the predictors

    Examples
    --------
    >>> probability_validate([True], ['predictor'])
    [True]
    >>> probability_validate([False, True], ['predictor'])
    Traceback (most recent call last)
      ...
    DCAError: number of probabilities must match number of predictors
    >>> probability_validate(['no', True], ['predictor1', 'predictor2'])
    Traceback (most recent call last)
      ...
    TypeError: all values in the probability list must be booleans
    """
    if probabilities is None:
        return [True] * len(predictors)

    if len(probabilities) != len(predictors):
        raise DCAError("number of probabilities must match number of predictors")
    for prob in probabilities:
        if not isinstance(prob, bool):
            raise TypeError("all values in the probability list must be booleans")

    return probabilities


def harms_validate(harms, predictors):
    """Validates that the harm list is valid for the current predictors)

    Parameters
    ----------
    harms : list(float) or None
        the list of harms for the predictors
    predictors : list(str)
        the current predictors for the analysis

    Returns
    -------
    list(float)
        the list of harms, if they are valid

    Raises
    ------
    DCAError
        if the number of harms doesn't match number of predictors

    Examples
    --------
    >>> harm_validate([0.5], ['predictor'])
    [0.5]
    >>> harm_validate([0.4, 0.5], ['predictor'])
    Traceback (most recent call last)
      ...
    DCAError: number of harms must match number of predictors
    """
    if harms is None:
        return [0] * len(predictors)

    if len(harms) != len(predictors):
        raise DCAError("number of harms must match number of predictors")
    return harms


def lowess_frac_validate(value, lowess_frac=None):
    """Validates that a valid lowess fraction was specified

    Parameters
    ----------
    value : float
        the `frac` value to use for lowess smoothing

    Returns
    -------
    float
        the value passed in, if valid

    Raises
    ------
    ValueError
        if the value is not between 0 and 1

    Examples
    --------
    >>> lowess_frac_validate(0.5)
    0.5
    >>> lowess_frac_validate(1.1)
    Traceback (most recent call last)
      ...
    ValueError: lowess_frac must be between 0 and 1
    """

    if value > 1 or value < 0:
        raise ValueError("lowess_frac must be between 0 and 1")
    return lowess_frac


def dca_input_validation(data, outcome, predictors,
                         x_start, x_stop, x_by,
                         probability, harm, intervention_per,
                         lowess_frac):
    """Performs input validation for the dca function

    Checks all relevant parameters, raises a ValueError if input is not valid

    Returns
    -------
    pd.DataFrame, [str], [bool], [float]
        A tuple of length 4 (data, predictors, probability, harm) where each is the
        newly updated or initialized version of its original input
    """
    # if probability is specified, len must match # of predictors
    # if not specified, initialize the probability parameter
    if probability is not None:
        # check if the number of probabilities matches the number of predictors
        if len(predictors) != len(probability):
            raise ValueError("Number of probabilites must match number of predictors")
        # validate and possibly convert predictors based on probabilities
        # data = _validate_predictors_dca(data, outcome, predictors, probability)
    else:
        probability = [True] * len(predictors)

    # if harm is specified, len must match # of predictors
    # if not specified, initialize the harm parameter
    if harm is not None:
        if len(predictors) != len(harm):
            raise ValueError("Number of harms must match number of predictors")
    else:
        harm = [0] * len(predictors)

    # check that 0 <= lowess_frac <= 1
    if lowess_frac < 0 or lowess_frac > 1:
        raise ValueError("Smoothing fraction must be between 0 and 1")

    return data, predictors, probability, harm  # return any mutated objects


def validate_data_predictors(data, outcome, predictors, probabilities, survival_time=False):
    """Validates that for each predictor column, all values are within the range 0-1

    Notes
    -----
    If a predictor has probability `True`, checks that the column `data[predictor]` has all values in the appropriate range.
    If a predictor has probability `False`, converts all values in that column with logistic regression

    Parameters
    ----------
    data : pd.DataFrame
        the data set
    outcome : str
        the column to use as 'outcome'
    predictors : list(str)
        the list of predictors for the analysis
    probabilities: list(bool)
        list marking whether a predictor is a probability
    survival_time : bool
        if the analysis is a survival time analysis
    """
    for i in range(0, len(predictors)):
        if probabilities[i]:
            # validate that any predictors with probability TRUE are b/t 0 and 1
            if (max(data[predictors[i]]) > 1) or (min(data[predictors[i]]) < 0):
                raise ValueError("{val} must be between 0 and 1"
                                 .format(val=repr(predictors[i])))
        else:
            if survival_time:
                pass
                # from statsmodels.sandbox.cox import CoxPH
                # TODO
            else:

                from statsmodels.api import Logit
                # predictor is not a probability, convert with logistic regression

                # model = Logit(data[outcome], data[predictors[i]])
                data[predictors[i]] = data.marker / data.marker.max()

                # .y_pred
                # data[predictors[i]] = df_binary.marker = df_binary.marker / df_binary.marker.max()

                # import sklearn.linear_model as linmod
                # model = linmod.LinearRegression(fit_intercept=False)  # might need to add C = 1e9 to
                # # remove regularization which makes results same as statsmodels' linear regression
                # mdl = model.fit(data[predictors[i]].values.reshape(-1, 1),
                #                 data[outcome].values.reshape(-1, 1))
                # data[predictors[i]] = mdl.fit

    return data


def _validate_predictors_stdca(data, outcome, predictors, probability):
    """TODO
    """
    for i, prob in enumerate(probability):
        if prob:
            # validate that any predictors with probability TRUE are b/t 0 and 1
            if (max(data[predictors[i]]) > 1) or (min(data[predictors[i]]) < 0):
                raise ValueError("{val} column values must all be between 0 and 1"
                                 .format(val=repr(predictors[i])))
        else:
            # predictor is not a probability, convert with cox regression
            import statsmodels.formula.api as smf

            # TODO -- implement cox regression
            raise NotImplementedError()
    return data


def stdca_input_validation(data, outcome, predictors, thresh_lb, thresh_ub,
                           thresh_step, probability, harm, intervention_per,
                           lowess_frac):
    """Performs input validation for the stdca function

    Checks all relevant parameters, raises a ValueError if input is not valid

    Returns
    -------
    tuple(pd.DataFrame, [str], [bool], [float]

    """
    # same validation as dca, except we don't want to validate the probabilities
    data, predictors, skip_prob, harm = dca_input_validation(data, outcome,
                                                             predictors, thresh_lb,
                                                             thresh_ub, thresh_step,
                                                             probability, harm,
                                                             intervention_per, lowess_frac)
    # do special validation for probabilities
    # if probability is specified, must match length of predictors
    if probability is not None:
        # length check already done by dca_input_validation
        data = _validate_predictors_stdca(data, outcome, predictors, probability)
    else:
        # default
        probability = [True] * len(predictors)


class DCAError(Exception):
    """Exception raised by DCA classes/functions
    """
    pass
