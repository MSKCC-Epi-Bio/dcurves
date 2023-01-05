# Load dcurves functions
from dcurves.dca import dca
from dcurves.dca import net_intervention_avoided
from dcurves.plot_graphs import plot_graphs

from dcurves.dca import _calc_prevalence, _create_initial_df
from dcurves.dca import _calc_modelspecific_stats, _calc_nonspecific_stats
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
        ).reset_index().sort_values(by=['model',
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

