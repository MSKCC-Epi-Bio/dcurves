# Load dcurves functions
from dcurves.dca import dca
from dcurves.dca import net_intervention_avoided
from dcurves.plot_graphs import plot_graphs

from dcurves.dca import _calc_prevalence, _create_initial_df
from dcurves.dca import _calc_initial_stats, _calc_more_stats
from dcurves.risks import _create_risks_df, _calc_binary_risks, _calc_surv_risks


# Load data functions
from dcurves.load_test_data import load_binary_df, load_survival_df, load_case_control_df
from dcurves.load_test_data import load_tutorial_bin_interventions_df
from dcurves.load_test_data import load_tutorial_bin_marker_risks_list
from dcurves.load_test_data import load_tutorial_bin_interventions_df
from dcurves.load_test_data import load_r_simple_surv_dca_result_df

# Load outside functions
import numpy as np
import pandas as pd
import statsmodels.api as sm

def test_python_bin_dca_and_nia():
    # Test scenario from dca-tutorial r-dca_intervention code chunk

    df_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
        )

    p_dca_df = \
        dca(
            data=df_cancer_dx,
            outcome='cancer',
            modelnames=['marker'],
            models_to_prob=['marker'],
            thresholds=np.arange(0.05, 0.36, 0.01)
        )

    # unwanted_p_cols = ['index', 'harm', 'n', 'prevalence', 'test_pos_rate', 'neg_rate',
    #                    'test_neg_rate', 'ppv', 'npv', 'sens', 'spec', 'lr_pos',
    #                    'lr_neg', 'tn_rate', 'fn_rate', 'net_benefit_all']

    p_net_int_df = \
        net_intervention_avoided(
            after_dca_df=p_dca_df
        ).sort_values(by=['model',
                          'threshold'],
                      ascending=[True,
                                 True]).reset_index(drop=True)

    r_net_int_df = \
        load_tutorial_bin_interventions_df().sort_values(by=['variable',
                                                            'threshold'],
                                                         ascending=[True,
                                                                    True]).reset_index(drop=True).drop(['label',
                                                                                                        'pos_rate',
                                                                                                        'harm',
                                                                                                        'n'], axis=1)

    comp_df = \
        pd.concat([
            p_net_int_df,
            r_net_int_df
        ],
        axis=1)
    assert r_net_int_df['threshold'].round(decimals=10).equals(other=p_net_int_df['threshold'].round(decimals=10))
    assert r_net_int_df['tp_rate'].round(decimals=10).equals(other=p_net_int_df['tp_rate'].round(decimals=10))
    assert r_net_int_df['fp_rate'].round(decimals=10).equals(other=p_net_int_df['fp_rate'].round(decimals=10))
    assert r_net_int_df['net_benefit'].round(decimals=10).equals(other=p_net_int_df['net_benefit'].round(decimals=10))
    assert r_net_int_df['net_intervention_avoided'].round(decimals=10).equals(other=p_net_int_df['net_intervention_avoided'].round(decimals=10))

def test_surv_dca():
    '''
    Run through all internal dca functions for sanity check
    '''

    data = load_survival_df()
    outcome = 'cancer'
    modelnames = ['famhistory']
    models_to_prob = None
    time = 1.5
    time_to_outcome_col = 'ttcancer'
    prevalence = None
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

    # 3. calculate prevalences

    prevalence_value = \
        _calc_prevalence(
            risks_df=risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # 4. Create initial dataframe for binary/survival cases

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    # 5. Calculate initial consequences

    initial_stats_df = \
        _calc_initial_stats(
            initial_df=initial_df,
            risks_df=risks_df,
            thresholds=thresholds,
            outcome=outcome,
            prevalence_value=prevalence_value,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # 6. Generate DCA-ready df with full list of calculated statistics
    final_dca_df = \
        _calc_more_stats(
            initial_stats_df=initial_stats_df
        )

    # plot_graphs(
    #     plot_df=final_dca_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )

def test_simple_surv():

    r_benchmark_results = load_r_simple_surv_dca_result_df()

    r_benchmark_df = \
        r_benchmark_results[['variable',
                             'threshold',
                             'tp_rate']]

    r_benchmark_df = \
        r_benchmark_df.sort_values(by=['variable',
                                       'threshold'],
                                   ascending=[True,
                                              True]).reset_index(drop=True)


    df_surv = load_survival_df()
    surv_dca_results = \
        dca(
            data=df_surv,
            outcome='cancer',
            modelnames=['famhistory'],
            time=1.5,
            time_to_outcome_col='ttcancer',
            thresholds=np.arange(0, 1.00, 0.01)
        )

    surv_dca_df = \
        surv_dca_results[['model',
                          'threshold',
                          'tp_rate']]

    surv_dca_df = \
        surv_dca_df.sort_values(by=['model',
                                    'threshold'],
                                ascending=[True,
                                           True]).reset_index(drop=True)

    round_dec_num = 6
    r_all_df = r_benchmark_df[r_benchmark_df['variable']=='all']['tp_rate'].round(decimals=round_dec_num)
    p_all_df = surv_dca_df[surv_dca_df['model']=='all']['tp_rate'].round(decimals=round_dec_num)
    assert r_all_df.equals(p_all_df)

    r_none_df = r_benchmark_df[r_benchmark_df['variable']=='none']['tp_rate'].round(decimals=round_dec_num)
    p_none_df = surv_dca_df[surv_dca_df['model']=='none']['tp_rate'].round(decimals=round_dec_num)
    assert r_none_df.equals(p_none_df)

    r_fam_df = r_benchmark_df[r_benchmark_df['variable']=='famhistory']['tp_rate']
    p_fam_df = surv_dca_df[surv_dca_df['model']=='famhistory']['tp_rate']
    assert r_fam_df.round(decimals=round_dec_num).equals(p_fam_df.round(decimals=round_dec_num))




