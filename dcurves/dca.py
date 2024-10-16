"""
This module houses most of the functions used in performing decision curve 
analysis, along with the main user-facing dca() functions used for binary
and survival outcomes.
"""

from typing import Optional, Union, Iterable
import pandas as pd
import lifelines

from .risks import _create_risks_df, _rectify_model_risk_boundaries
from .prevalence import _calc_prevalence


def _create_initial_df(
    thresholds: Iterable,
    modelnames: list,
    input_df_rownum: int,
    prevalence_value: Union[float, int],
    harm: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Create initial dataframe that will form the outputted table containing
    net benefit/interventions avoided values for plotting.

    Parameters
    ----------
    thresholds : Iterable
        Threshold values (x values) at which net benefit and net
        interventions avoided will be calculated
    modelnames : list[str]
        Column names from risks_df that contain model risk scores
    input_df_rownum : int
        Number of rows in original input dataframe
    prevalence_value : int or float
        Calculated prevalence value
    harm : dict[float]
        Models with their associated harm values

    Returns
    -------
    pd.DataFrame
        DataFrame set with initial parameters
    """

    modelnames = modelnames + ["all", "none"]
    rows = len(thresholds) * len(modelnames)
    model_column = pd.Series([x for y in modelnames for x in [y] * len(thresholds)])
    threshold_column = pd.Series(list(thresholds) * len(modelnames))
    n_column = pd.Series([input_df_rownum] * rows)
    prevalence_column = pd.Series([prevalence_value] * rows)
    harm_column = pd.Series([float(0)] * rows)

    if harm is not None:
        if isinstance(harm, dict):
            for model in harm.keys():
                harm_column.loc[model_column == model] = float(harm[model])
        elif not isinstance(harm, dict):
            raise ValueError("Harm should be either None or dict")
    else:
        pass

    initial_df = pd.DataFrame(
        {
            "model": model_column,
            "threshold": threshold_column,
            "n": n_column,
            "prevalence": prevalence_column,
            "harm": harm_column,
        }
    )

    return initial_df


def _calc_test_pos_rate(risks_df: pd.DataFrame, thresholds: Iterable, model: str) -> pd.Series:
    """
    Calculate each test positive rate per threshold value,
    which will be used to calculate true and false positive rates
    per threshold values in the survival DCA case.

    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores
        (scores ranging from 0 to 1 for columns of interest)
    thresholds : Iterable
        Threshold values (x values) at which net test positive
        rate will be calculated
    model : str
        Model column name in risks_df

    Returns
    -------
    pd.Series
        Calculated test positive rates for each threshold value for a model
    """

    test_pos_rate = []

    for threshold in thresholds:
        risk_above_thresh_tf_dict = dict(pd.Series(risks_df[model] >= threshold).value_counts())

        if True not in risk_above_thresh_tf_dict:
            test_pos_rate.append(0 / len(risks_df.index))
        elif True in risk_above_thresh_tf_dict:
            test_pos_rate.append(risk_above_thresh_tf_dict[True] / len(risks_df.index))

    return pd.Series(test_pos_rate)


def _calc_risk_rate_among_test_pos(
    risks_df: pd.DataFrame,
    outcome: str,
    model: str,
    thresholds: Iterable,
    time: Union[float, int],
    time_to_outcome_col: str,
) -> pd.Series:
    """
    Calculate the risk rate among test positive cases for each threshold value
    , which will be used to calculate true and false positive rates per
    threshold values in the survival DCA case.

    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging
        from 0 to 1 for columns of interest)
    outcome : str
        Column name of outcome of interest in risks_df
    model : str
        Model column name in risks_df
    thresholds : Iterable
        Threshold values (x values) at which risk rate among test positives
        will be calculated
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name containing time to outcome values, used in Survival DCA

    Returns
    -------
    pd.Series
        Calculated risk rate among test positive for each threshold value
    """

    risk_rate_among_test_pos = []
    kmf = lifelines.KaplanMeierFitter()

    max_time = max(risks_df[time_to_outcome_col])
    if max_time < time:
        return [None] * len(thresholds)

    for threshold in thresholds:
        mask = risks_df[model] >= threshold
        risk_above_thresh_time = risks_df.loc[mask, time_to_outcome_col].copy()
        risk_above_thresh_outcome = risks_df.loc[mask, outcome].copy()

        if len(risk_above_thresh_time) == 0 and len(risk_above_thresh_outcome) == 0:
            risk_rate_among_test_pos.append(0.0)
        else:
            kmf.fit(risk_above_thresh_time, risk_above_thresh_outcome * 1)
            timeline = max(kmf.timeline)
            if timeline < time:
                risk_rate_among_test_pos.append(None)
            elif timeline >= time:
                risk_rate_among_test_pos.append(
                    1 - float(kmf.survival_function_at_times(time).iloc[0])
                )

    return pd.Series(risk_rate_among_test_pos)


def _calc_tp_rate(
    risks_df: pd.DataFrame,
    thresholds: Iterable,
    model: str,
    outcome: str,
    test_pos_rate: Optional[pd.Series] = None,
    prevalence_value: Optional[Union[float, int]] = None,
    time: Optional[Union[float, int]] = None,
    time_to_outcome_col: Optional[str] = None,
) -> pd.Series:
    """
    Calculate true positive rates per threshold value in binary and survival DCA cases.

    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging from 0
        to 1 for columns of interest)
    thresholds : Iterable
        Threshold values (x values) at which true positive rate will be calculated
    model : str
        Model column name in risks_df
    outcome : str
        Column name of outcome of interest in risks_df
    test_pos_rate : pd.Series
        Calculated test positive rates for use in survival calculation of tp_rate
    prevalence_value : int or float
        Calculated prevalence value
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name in risks_df containing time to outcome values, used in Survival DCA

    Returns
    -------
    pd.Series
        Calculated true positive rate for each threshold value
    """

    if time_to_outcome_col is not None:
        risk_rate_among_test_pos = _calc_risk_rate_among_test_pos(
            risks_df=risks_df,
            outcome=outcome,
            model=model,
            thresholds=thresholds,
            time_to_outcome_col=time_to_outcome_col,
            time=time,
        )
        tp_rate = risk_rate_among_test_pos * test_pos_rate
    else:
        selected_rows = risks_df[risks_df[outcome].isin([True])].copy()

        true_outcome = selected_rows[[model]].copy()
        num_true_outcomes = len(true_outcome[model])

        tp_rate = []
        for threshold in thresholds:
            true_tf_above_thresh_count = (
                (true_outcome[model] >= threshold).value_counts().get(True, 0)
            )
            tp_rate.append((true_tf_above_thresh_count / num_true_outcomes) * prevalence_value)

    return pd.Series(tp_rate)


def _calc_fp_rate(
    risks_df: pd.DataFrame,
    thresholds: Iterable,
    model: str,
    outcome: str,
    test_pos_rate: Optional[pd.Series] = None,
    prevalence_value: Optional[Union[float, int]] = None,
    time: Optional[Union[float, int]] = None,
    time_to_outcome_col: Optional[str] = None,
) -> pd.Series:
    """
    Calculate false positive rates per threshold value in binary and survival DCA cases.

    Parameters
    ----------
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging from 0 to
        1 for columns of interest)
    thresholds : Iterable
        Threshold values (x values) at which false positive rate will be calculated
    model : str
        Model column name in risks_df
    outcome : str
        Column name of outcome of interest in risks_df
    test_pos_rate : pd.Series
        Calculated test positive rates for each threshold value for a model
    prevalence_value : int or float
        Calculated prevalence value
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col: str
        Column name of time of interest in risks_df
    Returns
    -------
    pd.Series
        Calculated false positive rate for each threshold value
    """
    # Survival
    if time_to_outcome_col is not None:
        risk_rate_among_test_pos = _calc_risk_rate_among_test_pos(
            risks_df=risks_df,
            outcome=outcome,
            model=model,
            thresholds=thresholds,
            time_to_outcome_col=time_to_outcome_col,
            time=time,
        )
        fp_rate = (1 - risk_rate_among_test_pos) * test_pos_rate
    # Binary
    elif time_to_outcome_col is None:
        false_outcome = risks_df[risks_df[outcome].isin([False])][[model]]

        fp_rate = []
        for threshold in thresholds:
            try:
                fp_counts = pd.Series(false_outcome[model] >= threshold).value_counts()
                fp_rate.append(fp_counts[True] / len(false_outcome[model]) * (1 - prevalence_value))
            except KeyError:
                fp_rate.append(0 / len(false_outcome[model]) * (1 - prevalence_value))

    return pd.Series(fp_rate)


def _calc_initial_stats(
    initial_df: pd.DataFrame,
    risks_df: pd.DataFrame,
    thresholds: Iterable,
    outcome: str,
    prevalence_value: Union[float, int],
    time: Optional[Union[float, int]] = None,
    time_to_outcome_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate the test positive, true positive, and false positive rate per each threshold value.

    Parameters
    ----------
    initial_df : pd.DataFrame
        DataFrame set with initial parameters
    risks_df : pd.DataFrame
        Data containing (converted if necessary) risk scores (scores ranging from 0 to 1
        for columns of interest)
    thresholds : Iterable
        Threshold values (x values) at which net benefit and net interventions avoided
        will be calculated
    outcome : str
        Column name of outcome of interest in risks_df
    prevalence_value : int or float
        Calculated prevalence value
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name in risks_df containing time to outcome values, used in Survival DCA

    Returns
    -------
    pd.DataFrame
        Initially set data with calculated test pos rate, true positive rate, false
        positive rate per threshold
    """

    for model in initial_df["model"].value_counts().index:
        test_pos_rate = _calc_test_pos_rate(risks_df=risks_df, thresholds=thresholds, model=model)
        tp_rate = _calc_tp_rate(
            risks_df=risks_df,
            thresholds=thresholds,
            model=model,
            outcome=outcome,
            time=time,
            time_to_outcome_col=time_to_outcome_col,
            test_pos_rate=test_pos_rate,
            prevalence_value=prevalence_value,
        )

        fp_rate = _calc_fp_rate(
            risks_df=risks_df,
            thresholds=thresholds,
            model=model,
            outcome=outcome,
            time=time,
            time_to_outcome_col=time_to_outcome_col,
            test_pos_rate=test_pos_rate,
            prevalence_value=prevalence_value,
        )

        # .copy() below added to prevent chained indexing
        initial_df.loc[
            initial_df["model"] == model, "test_pos_rate"
        ] = test_pos_rate.tolist().copy()
        initial_df.loc[initial_df["model"] == model, "tp_rate"] = tp_rate.tolist().copy()
        initial_df.loc[initial_df["model"] == model, "fp_rate"] = fp_rate.tolist().copy()

    return initial_df


def _calc_more_stats(initial_stats_df: pd.DataFrame, nper: int = 1) -> pd.DataFrame:
    """
    Calculate additional statistics (net benefit, net interventions avoided) and
    add them to initial_stats_df.

    Parameters
    ----------
    initial_stats_df : pd.DataFrame
        Initially set data with calculated test pos rate, true positive rate, false
        positive rate per threshold
    nper : int
        Total number of interventions, multiplies proportion of interventions
        avoided to get scaled plots
    Returns
    -------
    pd.DataFrame
        Data of full set of stats offered by the package per each threshold
        value (test_pos_rate, tp, fp, nb, nia)

    """
    initial_stats_df["net_benefit"] = (
        initial_stats_df["tp_rate"]
        - (initial_stats_df["threshold"] / (1 - initial_stats_df["threshold"]))
        * initial_stats_df["fp_rate"]
        - initial_stats_df["harm"]
    )

    mask = initial_stats_df["model"] == "all"
    temp = initial_stats_df[mask]["net_benefit"]
    initial_stats_df["net_benefit_all"] = temp.tolist() * len(
        initial_stats_df["model"].value_counts()
    )
    initial_stats_df["net_intervention_avoided"] = (
        (initial_stats_df.net_benefit - initial_stats_df.net_benefit_all)
        / (initial_stats_df.threshold / (1 - initial_stats_df.threshold))
        * nper
    )

    initial_stats_df = initial_stats_df.drop(columns="net_benefit_all").copy()

    # initial_stats_df['neg_rate'] = 1 - initial_stats_df['prevalence']
    # initial_stats_df['fn_rate'] = initial_stats_df['prevalence'] - initial_stats_df['tp_rate']
    # initial_stats_df['tn_rate'] = initial_stats_df['neg_rate'] - initial_stats_df['fp_rate']
    #
    #
    # initial_stats_df['test_neg_rate'] = initial_stats_df['fn_rate'] + initial_stats_df['tn_rate']
    # initial_stats_df['ppv'] = initial_stats_df['tp_rate'] /\
    #                               (initial_stats_df['tp_rate'] + initial_stats_df['fp_rate'])
    # initial_stats_df['npv'] = initial_stats_df['tn_rate'] /\
    #                               (initial_stats_df['tn_rate'] + initial_stats_df['fn_rate'])
    # initial_stats_df['sens'] = initial_stats_df['tp_rate'] /\
    #                                (initial_stats_df['tp_rate'] + initial_stats_df['fn_rate'])
    # initial_stats_df['spec'] = initial_stats_df['tn_rate'] /\
    #                                (initial_stats_df['tn_rate'] + initial_stats_df['fp_rate'])
    # initial_stats_df['lr_pos'] = initial_stats_df['sens'] /\
    #                                  (1 - initial_stats_df['spec'])
    # initial_stats_df['lr_neg'] = (1 - initial_stats_df['sens']) /\
    #                                  initial_stats_df['spec']

    final_dca_df = initial_stats_df

    return final_dca_df


def dca(
    data: pd.DataFrame,
    outcome: str,
    modelnames: list,
    thresholds: Iterable = [i / 100 for i in range(0, 100)],
    harm: Optional[dict] = None,
    models_to_prob: Optional[list] = None,
    prevalence: Optional[Union[float, int]] = None,
    time: Optional[Union[float, int]] = None,
    time_to_outcome_col: Optional[str] = None,
    nper: Optional[int] = 1,
) -> pd.DataFrame:
    """
    Decision curve analysis is a method for evaluating and comparing prediction
    models that incorporates clinical consequences, requiring only the data set
    on which the models are tested, and can be applied to models that have eit-
    her continuous or dichotomous results. The dca function performs decision
    curve analysis for binary and survival outcomes.

    Parameters
    ----------
    data: pd.DataFrame
        Initial raw data ideally containing risk scores (scores ranging from 0
        to 1), or else predictor/model values, and outcome of interest
    outcome: str
        Column name of outcome of interest in risks_df
    modelnames : list[str]
        Column names from data that contain model risk scores or values
    thresholds : Iterable
        Threshold values (x values) at which net benefit and net interventions
        avoided will be calculated
    harm : dict[float]
        Models with their associated harm values
    models_to_prob : list[str]
        Columns that need to be converted to risk scores from 0 to 1
    prevalence : int or float
        Value that indicates the prevalence among the population, only to be
        specified in case-control situations
    time : int or float
        Time of interest in years, used in Survival DCA
    time_to_outcome_col : str
        Column name in data containing time to outcome values, used in Survival DCA
    nper : int
        Total number of interventions, multiplies proportion of interventions
        avoided to get scaled plots

    Returns
    -------
    pd.DataFrame
        Data containing net benefit and interventions avoided scores to be plotted
        against threshold values

    Examples
    --------
    from dcurves import dca, plot_graphs, load_test_data

    import numpy as np

    |

    dca_results = \
        dca(
            data=load_test_data.load_binary_df(),
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=np.arange(0,0.45,0.01)
        )

    |

    plot_graphs(
        plot_df=dca_results,
        graph_type='net_benefit',
        y_limits=[-0.05, 0.15],
        color_names=['blue', 'red', 'green']
    )

    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas DataFrame")

    risks_df = _create_risks_df(
        data=data,
        outcome=outcome,
        models_to_prob=models_to_prob,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    rectified_risks_df = _rectify_model_risk_boundaries(risks_df=risks_df, modelnames=modelnames)

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=outcome,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
        harm=harm,
    )

    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        outcome=outcome,
        prevalence_value=prevalence_value,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    final_dca_df = _calc_more_stats(initial_stats_df=initial_stats_df, nper=nper)

    return final_dca_df
