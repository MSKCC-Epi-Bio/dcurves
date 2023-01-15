
# Load Functions To Test/Needed For Testing
from dcurves.dca import _calc_tp_rate, _calc_fp_rate, _calc_test_pos_rate
from dcurves.dca import _calc_risk_rate_among_test_pos
from dcurves.risks import _create_risks_df
from dcurves.dca import _calc_prevalence, _create_initial_df, _calc_initial_stats, _calc_more_stats
from dcurves.dca import _rectify_model_risk_boundaries

# Load Data for Testing
from dcurves.load_test_data import load_r_case2_results
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.load_test_data import load_tutorial_r_stdca_coxph_df
from dcurves.load_test_data import load_tutorial_r_stdca_coxph_pr_failure18_test_consequences
from dcurves.load_test_data import load_r_simple_surv_dca_result_df
from dcurves.load_test_data import load_r_simple_surv_tpfp_calc_df
# Load Tools
import pandas as pd
import numpy as np

# Load Statistics Libraries
import lifelines

# 230102 SP: Left off here, trying to figure out why survival dca won't match r survival dca results
# Doesn't match for pr_failure18 results, so trying on simple case with df surv first, compare to R results

# Don't need to test test_pos_rate for survival case since it's the same

def test_case2_surv_risk_rate_among_test_positive():
    # Note: To calc risk rate among test positive, divide tp_rate by test_pos_rate

    data = load_survival_df()
    thresholds = np.arange(0.00, 1.0, 0.01)
    outcome = 'cancer'
    modelnames = ['cancerpredmarker']
    models_to_prob = None
    time = 1
    time_to_outcome_col = 'ttcancer'
    harm = None
    prevalence = None

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

    model_rratp_dict = {}
    for model in ['all', 'none', 'cancerpredmarker']:
        risk_rate_among_test_pos_series = \
            _calc_risk_rate_among_test_pos(
                risks_df=rectified_risks_df,
                outcome=outcome,
                model=model,
                thresholds=thresholds,
                time_to_outcome_col=time_to_outcome_col,
                time=time
            )
        model_rratp_dict[model] = risk_rate_among_test_pos_series.reset_index(drop=1)

    r_benchmark_results = load_r_case2_results()

    r_benchmark_results['risk_rate_among_test_pos'] = \
        r_benchmark_results.tp_rate / r_benchmark_results.test_pos_rate
    r_benchmark_results.loc[r_benchmark_results.variable == 'none', 'risk_rate_among_test_pos'] = float(0)

    for model in ['all', 'none', 'cancerpredmarker']:

        assert model_rratp_dict[
            model].reset_index(drop=True).round(decimals=5).equals(r_benchmark_results[
            r_benchmark_results.variable == model]['risk_rate_among_test_pos'].reset_index(drop=True).round(decimals=5))


def test_case2_all_stats():

    r_benchmark_results = load_r_case2_results()

    data = load_survival_df()
    outcome = 'cancer'
    prevalence = None
    time = 1
    time_to_outcome_col = 'ttcancer'
    models_to_prob = None
    modelnames = ['cancerpredmarker']
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

    for model in ['all', 'none', 'cancerpredmarker']:
        for stat in ['tp_rate', 'fp_rate', 'net_benefit']:

            round_dec_num = 6
            p_tp_rate_series = final_dca_df[
                final_dca_df.model == model][stat].reset_index(drop=True).round(decimals=round_dec_num)

            r_tp_rate_series = r_benchmark_results[
                r_benchmark_results.variable == model][stat].reset_index(drop=1).round(decimals=round_dec_num)

            assert p_tp_rate_series.equals(r_tp_rate_series)


def test_tut_pr_failure18_tp_rate():

    df_time_to_cancer_dx = \
            pd.read_csv(
                    "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
                )
    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')
    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5]
        )
    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]

    df_time_to_cancer_dx = \
            pd.read_csv(
                    "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
                )
    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')
    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5]
        )

    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]

    outcome = 'cancer'
    prevalence = None
    time = 1.5
    time_to_outcome_col = 'ttcancer'
    thresholds = np.arange(0, 0.51, 0.01)
    modelnames = ['pr_failure18']
    harm = None

    risks_df = \
        _create_risks_df(
            data=df_time_to_cancer_dx,
            outcome='cancer',
            models_to_prob=None,
            time=time,
            time_to_outcome_col='ttcancer'
        )

    prevalence_value = \
        _calc_prevalence(
            risks_df=risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    initial_stats_df = \
        _calc_initial_stats(
            initial_df=initial_df,
            risks_df=risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        ).reset_index().sort_values(by=['model',
                                        'threshold'],
                                    ascending=[True,
                                               True]).reset_index(drop=True)

    r_stats_df = load_tutorial_r_stdca_coxph_df().reset_index().sort_values(by=['variable',
                                        'threshold'],
                                    ascending=[True,
                                               True]).reset_index(drop=True)

    pr_failure18_conseq = load_tutorial_r_stdca_coxph_pr_failure18_test_consequences()

    comp_df = \
        pd.concat(
            [
                initial_stats_df,
                r_stats_df
            ],
            axis=1
        ).drop(['harm', 'index', 'label', 'n', 'net_benefit'], axis=1)

    # Remove all, none models since those are the same
    # Differences pop up in pr_failure18

    comp_df = comp_df[comp_df['model']=='pr_failure18'].reset_index(drop=True)
    comp_df = pd.concat([comp_df, pr_failure18_conseq[['r_test_pos_rate']]], axis=1)

    # From test below we know test_pos_rate is same between Python and R
    round_dec_num = 7
    assert comp_df['test_pos_rate'].round(
        decimals=round_dec_num
    ).equals(comp_df['r_test_pos_rate'].round(
        decimals=round_dec_num
    ))





