import pytest
import numpy as np
import pandas as pd
from dcurves.surv_dca import _surv_convert_to_risk, _surv_calculate_test_consequences, surv_dca
from dcurves.load_test_data import load_survival_df

from dcurves.load_test_data import load_survival_df
from dcurves.load_test_data import load_case_control_df
import statsmodels.api as sm

from dcurves.plot_graphs import plot_net_benefit

from dcurves.dca import _calculate_test_consequences

import statsmodels.api as sm

import matplotlib.pyplot as plt

# def test_reg_ctc():
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     predictor = 'famhistory'
#     thresholds = np.linspace(0.01, 0.99, 98)
#     prevalence = 0
#     time = 1.0
#     time_to_outcome_col = 'ttcancer'
#
#
#     # ctc_result = \
#     #     _calculate_test_consequences(
#     #         model_frame=data,
#     #         outcome=outcome,
#     #         predictor=predictor,
#     #         thresholds=thresholds,
#     #         prevalence=prevalence,
#     #         time=time,
#     #         time_to_outcome_col=time_to_outcome_col
#     #     )
#
#     sctc_result = \
#         _surv_calculate_test_consequences(
#             risks_df=data,
#             outcome=outcome,
#             predictor=predictor,
#             thresholds=thresholds,
#             prevalence=prevalence,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#
#     # print('ctc')
#     # print(ctc_result)
#     print('sctc')
#     print(sctc_result)






# def test_surv_ctr():
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     predictors_to_prob = ['marker']
#     time = 1
#     time_to_outcome_col = 'ttcancer'
#
#
#     vars_to_risk_df = \
#         _surv_convert_to_risk(
#             data=data,
#             outcome=outcome,
#             predictors_to_prob=predictors_to_prob,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#

# def test_surv_ctc():
#
#     data = load_survival_df()
#     outcome = 'cancer'
#     predictors_to_prob = ['marker']
#     time = 1
#     time_to_outcome_col = 'ttcancer'
#
#     vars_to_risk_df = \
#         _surv_convert_to_risk(
#             data=data,
#             outcome=outcome,
#             predictors_to_prob=predictors_to_prob,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col
#         )
#
#     machine_epsilon = np.finfo(float).eps
#
#     vars_to_risk_df['all'] = [1 - machine_epsilon for i in range(0, len(vars_to_risk_df.index))]
#     vars_to_risk_df['none'] = [0 + machine_epsilon for i in range(0, len(vars_to_risk_df.index))]
#
#     predictors = ['famhistory', 'cancerpredmarker']
#     covariate_names = np.append(predictors, ['all', 'none'])
#
#     prevalence = 0.5
#
#     thresholds = np.linspace(0, 1, 101)
#     thresholds = np.where(thresholds == 0.00, 0.00 + machine_epsilon, thresholds)
#     thresholds = np.where(thresholds == 1.00, 1.00 - machine_epsilon, thresholds)
#
#     harm = {'cancerpredmarker': 0.12}
#
#     test_consequences_df = \
#         pd.concat([_surv_calculate_test_consequences(
#             risks_df=vars_to_risk_df,
#             outcome=outcome,
#             predictor=predictor,
#             thresholds=thresholds,
#             prevalence=prevalence,
#             time=time,
#             time_to_outcome_col=time_to_outcome_col,
#             harm=harm) for predictor in covariate_names])
#
#     print(test_consequences_df)

# def test_scratch2():
#     surv_df = load_survival_df()
#     time = 1
#     tto_col = 'ttcancer'
#
#     surv_calcs_df = \
#         surv_dca(
#             data=surv_df,
#             outcome='cancer',
#             predictors=['marker', 'cancerpredmarker', 'famhistory'],
#             predictors_to_prob=['marker'],
#             prevalence=0.5,
#             # harm={'cancerpredmarker': 0.3},
#             time=1.0,
#             time_to_outcome_col='ttcancer'
#         )
#
#     # print(surv_calcs_df.columns)
#     # print(surv_calcs_df['variable'].value_counts())
#     plot_net_benefit(
#         data=surv_calcs_df
#     )



def test_scratch1():

    asdf = pd.read_csv('https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv')
    print(
        sm.formula.glm(
            formula='cancer ~ famhistory',
            data=asdf,
            family=sm.families.Binomial()
        ).fit().summary()
    )

    import

    binary_dca(

    )

