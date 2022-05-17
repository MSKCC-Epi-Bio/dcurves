from .dca import *


def _validate_dataframe(data: pd.DataFrame):
    # make sure dataframe input is a dataframe

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be of type: pandas DataFrame")


def _validate_outcome(outcome: str):
    # make sure outcome input is a str

    if not isinstance(outcome, str):
        raise TypeError("outcome must be of type: string")


def _validate_predictor(predictor: str):
    # make sure predictor input is a str

    if not isinstance(predictor, str):
        raise TypeError("predictor must be of type: string")


def _validate_prevalence(prevalence: float):
    # make sure prevalence input is a float

    if prevalence is None:
        pass
    elif isinstance(prevalence, float) or isinstance(prevalence, int):
        pass
    else:
        raise TypeError("prevalence must be of type: float or int")


def _validate_time(time: float or int):
    # make sure time input is a float or int

    if time is None:
        pass
    elif isinstance(time, float) or isinstance(time, int):
        pass
    else:
        raise TypeError("time must be of type: float or int")


def _validate_time_to_outcome_col(time_to_outcome_col: str):
    # make sure time_to_outcome_col input is a string

    if time_to_outcome_col is None:
        pass
    elif isinstance(time_to_outcome_col, str):
        pass
    else:
        raise TypeError("time_to_outcome_col must be of type: str")


def _convert_to_risk_input_checks(
                    model_frame: pd.DataFrame,
                    outcome: str,
                    predictor: str,
                    prevalence: float,
                    time: float,
                    time_to_outcome_col: str):
    # make sure _convert_to_risk inputs are kosher

    _validate_dataframe(data=model_frame)
    _validate_outcome(outcome=outcome)
    _validate_predictor(predictor=predictor)
    _validate_prevalence(prevalence=prevalence)
    _validate_time(time=time)
    _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)


def _validate_thresholds(thresholds: list):
    # make sure thresholds input is a list of floats

    if not isinstance(thresholds, list):
        raise TypeError('probabilities must be of type list')
    else:
        for i in thresholds:
            if not isinstance(i, float):
                raise TypeError("time_to_outcome_col must be of type: float")
        if not (len(thresholds) > 0):
            raise ValueError("Thresholds must contain at least 1 value")


def _calculate_test_consequences_input_checks(
                                model_frame: pd.DataFrame,
                                outcome: str,
                                predictor: str,
                                thresholds: list,
                                prevalence: float,
                                time: float,
                                time_to_outcome_col: str):
    # make sure _calculate_test_consequences inputs are kosher

    _validate_dataframe(data=model_frame)
    _validate_outcome(outcome=outcome)
    _validate_predictor(predictor=predictor)
    _validate_thresholds(thresholds=thresholds)
    _validate_prevalence(prevalence=prevalence)
    _validate_time(time=time)
    _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)


def _validate_predictors(predictors: list):
    # Make sure input is a list, and all list elements are strings

    if predictors is None:
        pass
    elif not isinstance(predictors, list):
        raise TypeError('Predictors must be of type list')
    else:
        for i in predictors:
            if not isinstance(i, str):
                raise TypeError('All elements in predictors list must be of type str')


def _validate_thresh_vals(thresh_vals: list):
    # make sure thresh_vals input is a list, and all list elements are int/float

    if not isinstance(thresh_vals, list):
        raise TypeError('probabilities must be of type list')
    else:
        for i in thresh_vals:
            if not (isinstance(i,float) or isinstance(i,int)):
                raise TypeError("thresh_vals contents must be of type: float/int")

    if not (len(thresh_vals) == 3):
        raise ValueError("thresh_vals list must contain 3 values")


def _validate_harm(harm: dict):
    # make sure harm input is a dict

    if harm is None:
        pass
    elif not isinstance(harm, dict):
        raise TypeError('harm input not of type dict')


def _validate_probabilities(probabilities: list, predictors: list):
    # make sure probabilities input is a list, the length of
    # probabilities/predictors is the same, and all elements
    # in probabilities is of type bool

    _validate_predictors(predictors=predictors)

    if probabilities is None:
        pass
    elif not isinstance(probabilities, list):
        raise TypeError('probabilities must be of type list')
    else:
        if not (len(probabilities) == len(predictors)):
            raise ValueError('probabilities input and predictors input must be of same length')
        for i in probabilities:
            if not isinstance(i, bool):
                raise TypeError('All elements in probabilities must be of type bool')


def _dca_input_checks(
        model_frame: pd.DataFrame,
        outcome: str,
        predictors: list,
        thresh_vals: list,
        harm: dict,
        probabilities: list,  # list of TRUE/FALSE values indicating which predictors
        time: float,
        prevalence: float,
        time_to_outcome_col: str):
    # make sure dca inputs are kosher

    _validate_dataframe(data=model_frame)
    _validate_outcome(outcome=outcome)
    _validate_predictors(predictors=predictors)
    _validate_thresh_vals(thresh_vals=thresh_vals)
    _validate_harm(harm=harm)
    _validate_probabilities(probabilities=probabilities, predictors=predictors)
    _validate_time(time=time)
    _validate_prevalence(prevalence=prevalence)
    _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)

def _plot_net_benefit_graphs_input_checks(output_df: pd.DataFrame):
    # make sure output_df input is pandas DataFrame

    _validate_dataframe(data=output_df)






















