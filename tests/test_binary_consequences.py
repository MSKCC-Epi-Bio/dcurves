
# Load Functions To Test/Needed For Testing
from dcurves.dca import _calc_tp_rate, _calc_fp_rate
from dcurves.risks import _create_risks_df
from dcurves.dca import _calc_prevalence, _create_initial_df, _calc_initial_stats
from dcurves.dca import _calc_more_stats
from dcurves.dca import _rectify_model_risk_boundaries

# Load Data for Testing
from dcurves.load_test_data import load_r_case1_results
from dcurves.load_test_data import load_binary_df

# Load Tools
import pandas as pd
import numpy as np


def test_case1_binary_test_pos_rate():

    # Set Variables
    data = load_binary_df()
    thresholds = np.arange(0, 1.0, 0.01)
    outcome = 'cancer'
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    modelnames = ['famhistory']
    prevalence = None
    harm = None

    # Load in R Benchmarking Results
    r_benchmark_results = load_r_case1_results()

    # Make DF of Thresholds, test_pos_rate For Each Model
    r_benchmark_test_pos_rates_df = \
        pd.DataFrame(
            {
                'threshold': thresholds,
                'famhistory': r_benchmark_results[r_benchmark_results[
                                                      'variable'] == 'famhistory'][
                    'test_pos_rate'].reset_index(drop=True),
                'all': r_benchmark_results[r_benchmark_results[
                                               'variable'] == 'all'][
                    'test_pos_rate'].reset_index(drop=True),
                'none': r_benchmark_results[r_benchmark_results[
                                                'variable'] == 'none'][
                    'test_pos_rate'].reset_index(drop=True)
            }
        )

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )


    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(rectified_risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    initial_stats_df = \
        _calc_initial_stats(
            initial_df=initial_df,
            risks_df=rectified_risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    p_test_pos_rate_df = \
        pd.DataFrame(
            {
                'threshold': thresholds,
                'famhistory': initial_stats_df[initial_stats_df[
                                                   'model'] == 'famhistory']['test_pos_rate'].reset_index(drop=True),
                'all': initial_stats_df[initial_stats_df[
                                                   'model'] == 'all']['test_pos_rate'].reset_index(drop=True),
                'none': initial_stats_df[initial_stats_df[
                                     'model'] == 'none']['test_pos_rate'].reset_index(drop=True),
            }
        )

    assert r_benchmark_test_pos_rates_df['famhistory'].round(\
        decimals=6).equals(p_test_pos_rate_df['famhistory'].round(decimals=6))

    assert r_benchmark_test_pos_rates_df['all'].round(\
        decimals=6).equals(p_test_pos_rate_df['all'].round(decimals=6))

    assert r_benchmark_test_pos_rates_df['none'].round(\
        decimals=6).equals(p_test_pos_rate_df['none'].round(decimals=6))


def test_case1_binary_tp_rate():

    r_benchmark_results_df = load_r_case1_results()

    data = load_binary_df()
    outcome = 'cancer'
    prevalence = None
    time = None
    time_to_outcome_col = None
    models_to_prob = None
    modelnames = ['famhistory']
    model = modelnames[0]
    thresholds = np.arange(0, 1.0, 0.01)
    harm = None
    # model = modelnames[0]

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    p_tp_rate = \
        _calc_tp_rate(
            risks_df=rectified_risks_df,
            thresholds=thresholds,
            model=model,
            outcome='cancer',
            test_pos_rate=None,
            prevalence_value=prevalence_value
        )

    bm_tp_rate = r_benchmark_results_df[r_benchmark_results_df.variable == 'famhistory'].tp_rate.reset_index(drop=True)

    assert bm_tp_rate.round(decimals=6).equals(p_tp_rate.round(decimals=6))


def test_case1_binary_fp_rate():

    r_benchmark_results_df = load_r_case1_results()

    data = load_binary_df()
    outcome = 'cancer'
    prevalence = None
    time = None
    time_to_outcome_col = None
    models_to_prob = None
    modelnames = ['famhistory']
    model = modelnames[0]
    thresholds = np.arange(0, 1.0, 0.01)
    harm = None
    # model = modelnames[0]

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    p_fp_rate = \
        _calc_fp_rate(
            risks_df=rectified_risks_df,
            thresholds=thresholds,
            model=model,
            outcome='cancer',
            test_pos_rate=None,
            prevalence_value=prevalence_value
        )

    bm_fp_rate = r_benchmark_results_df[r_benchmark_results_df.variable == 'famhistory'].fp_rate.reset_index(drop=True)

    assert bm_fp_rate.round(decimals=6).equals(p_fp_rate.round(decimals=6))

def test_case1_binary_calc_initial_stats():
    r_benchmark_results_df = load_r_case1_results()

    data = load_binary_df()
    outcome = 'cancer'
    prevalence = None
    time = None
    time_to_outcome_col = None
    models_to_prob = None
    modelnames = ['famhistory']
    model = modelnames[0]
    thresholds = np.arange(0, 1.0, 0.01)
    harm = None

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(rectified_risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    initial_stats_df = \
        _calc_initial_stats(
            initial_df=initial_df,
            risks_df=rectified_risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    for model in ['all', 'none', 'famhistory']:
        for stat in ['test_pos_rate', 'tp_rate', 'fp_rate']:

            assert initial_stats_df[initial_stats_df.model == model][
                stat].round(decimals=6).reset_index(drop=True).equals(r_benchmark_results_df[
                r_benchmark_results_df.variable == model][stat].round(decimals=6).reset_index(drop=True))


def test_case1_binary_calc_more_stats():
    r_benchmark_results_df = load_r_case1_results()

    data = load_binary_df()
    outcome = 'cancer'
    prevalence = None
    time = None
    time_to_outcome_col = None
    models_to_prob = None
    modelnames = ['famhistory']
    model = modelnames[0]
    thresholds = np.arange(0, 1.0, 0.01)
    harm = None

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(rectified_risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    initial_stats_df = \
        _calc_initial_stats(
            initial_df=initial_df,
            risks_df=rectified_risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    final_dca_df = \
        _calc_more_stats(
            initial_stats_df=initial_stats_df
        )

    for model in ['all', 'none', 'famhistory']:
        assert final_dca_df[final_dca_df.model == model][
                'net_benefit'].round(decimals=6).reset_index(drop=True).equals(r_benchmark_results_df[
                r_benchmark_results_df.variable == model]['net_benefit'].round(decimals=6).reset_index(drop=True))
