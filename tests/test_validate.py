# import numpy as np
# import unittest
# from dcurves.load_test_data import load_binary_df
#
#
# class TestCTCInputValidation(unittest.TestCase):
#
#     df_binary = load_binary_df()
#     binary_inputs = {
#         'data': df_binary,
#         'outcome': 'cancer',
#         'predictors': ['cancerpredmarker', 'marker'],
#         'thresh_vals': [0.01, 0.35, 0.01],
#         'probabilities': [False, True]
#     }
#
#     thresholds = np.arange(binary_inputs['thresh_vals'][0],
#                            binary_inputs['thresh_vals'][1] + binary_inputs['thresh_vals'][2],
#                            binary_inputs['thresh_step'])  # array of values
#
#     thresholds = np.insert(thresholds, 0, 0.1 ** 9)
#
#     # CTC = Calculate Test Consequences
#     def testCTCInputValidation(self, binary_inputs):
#
#         # thresholds = np.arange(binary_inputs['thresh_lo'], binary_inputs['thresh_hi'], binary_inputs['thresh_step'],
#         #                        binary_inputs['thresh_step'])  # array of values
#         # thresholds = np.insert(thresholds, 0, 0.1 ** 9)
#
#         print(binary_inputs['data'])
#
#         # validate._calculate_test_consequences_input_checks(
#         #                         model_frame=binary_inputs['data'],
#         #                         outcome=binary_inputs['outcome'],
#         #                         predictor=binary_inputs['predictors'][1],
#         #                         thresholds=thresholds,
#         #                         prevalence=binary_inputs['prevalence'],
#         #                         time=binary_inputs['time'],
#         #                         time_to_outcome_col=binary_inputs['time_to_outcome_col'])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
