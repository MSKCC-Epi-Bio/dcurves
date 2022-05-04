# Load packages to be used:
import dcurves
from dcurves.dca import dca
import pandas as pd
from dcurves.load_test_data import load_binary_df, load_survival_df
from dcurves.dca import plot_net_benefit_graphs
import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------

pd.set_option('display.max_rows', 500)

# Use local functions from load_test_data.py module in dcurves package,
# load simulated binary and survival data

df_binary = load_binary_df()
df_surv = load_survival_df()

# Examples
# 1. Binary input for Decision Curve Analysis (DCA)

# Set binary input dataframe for DCA

# binary_inputs = {
#     'data': df_binary,
#     'outcome': 'cancer',
#     'predictors': ['cancerpredmarker', 'marker'],
#     'thresh_lo': 0.01,
#     'thresh_hi': 0.35,
#     'thresh_step': 0.01,
#     'harm': None,
#     'probabilities': [False, True],
#     'time': None,
#     'prevalence': None,
#     'time_to_outcome_col': None
# }

# Run DCA on binary data inputs
# Obtain a dataframe containing predictor and outcome data post\
# application of dca, use for plotting

# binary_output_df = dca(
#         data=binary_inputs['data'],
#         outcome=binary_inputs['outcome'],
#         predictors=binary_inputs['predictors'],
#         thresh_lo=binary_inputs['thresh_lo'],
#         thresh_hi=binary_inputs['thresh_hi'],
#         thresh_step=binary_inputs['thresh_step'],
#         harm=binary_inputs['harm'],
#         probabilities=binary_inputs['probabilities'],
#         time=binary_inputs['time'],
#         prevalence=binary_inputs['prevalence'],
#         time_to_outcome_col=binary_inputs['time_to_outcome_col'])

# Plot binary DCA output
# Uncomment line below to plot:

# plot_net_benefit_graphs(binary_output_df)

# 2. Survival input for Decision Curve Analysis (DCA)

# Set survival input dataframe for DCA

# survival_inputs = {
#     'data': df_surv,
#     'outcome': 'cancer',
#     'predictors': ['cancerpredmarker'],
#     'thresh_lo': 0.01,
#     'thresh_hi': 0.50,
#     'thresh_step': 0.01,
#     'harm': None,
#     'probabilities': [False],
#     'time': 1,
#     'prevalence': None,
#     'time_to_outcome_col': 'ttcancer'
# }

# Run DCA on survival data inputs
# Obtain a dataframe containing predictor and outcome data post\
# application of dca, use for plotting

# survival_output_df = dca(
#         data=survival_inputs['data'],
#         outcome=survival_inputs['outcome'],
#         predictors=survival_inputs['predictors'],
#         thresh_lo=survival_inputs['thresh_lo'],
#         thresh_hi=survival_inputs['thresh_hi'],
#         thresh_step=survival_inputs['thresh_step'],
#         harm=survival_inputs['harm'],
#         probabilities=survival_inputs['probabilities'],
#         time=survival_inputs['time'],
#         prevalence=survival_inputs['prevalence'],
#         time_to_outcome_col=survival_inputs['time_to_outcome_col'])

# Plot binary DCA output
# Uncomment line below to plot:

# plot_net_benefit_graphs(survival_output_df)

# ---------------------------------------------------------------------

# Scratch

# import os
# cwd = os.getcwd()
# print(cwd)
# files = os.listdir(cwd)
# print("Files in %r: %s" % (cwd, files))

df_dan_test = pd.read_csv('/Users/ShaunPorwal/Desktop/df_cancer_dx.csv')

dan_test_inputs = {
    'data': df_dan_test,
    'outcome': 'cancer',
    'predictors': ['famhistory'],
    'thresh_lo': 0.01,
    'thresh_hi': 1,
    'thresh_step': 0.01,
    'harm': None,
    'probabilities': [False],
    'time': None,
    'prevalence': None,
    'time_to_outcome_col': None
}

dan_test_output_df = dca(
        data=df_dan_test,
        outcome=dan_test_inputs['outcome'],
        predictors=dan_test_inputs['predictors'],
        thresh_lo=dan_test_inputs['thresh_lo'],
        thresh_hi=dan_test_inputs['thresh_hi'],
        thresh_step=dan_test_inputs['thresh_step'],
        harm=dan_test_inputs['harm'],
        probabilities=dan_test_inputs['probabilities'],
        time=dan_test_inputs['time'],
        prevalence=dan_test_inputs['prevalence'],
        time_to_outcome_col=dan_test_inputs['time_to_outcome_col'])



help(dca)

# print(dan_test_output_df)




