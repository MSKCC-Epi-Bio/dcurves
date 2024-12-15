"""
This module is commented out, and will be populated in the future to add checks
and display helpful error messages.
"""
# from .dca import *
#
#
# def _validate_time(time: float or int):
#     # make sure time input is a float or int
#
#     if time is None:
#         pass
#     elif isinstance(time, float) or isinstance(time, int):
#         pass
#     else:
#         raise TypeError("time must be of type: float or int")
#
#
# def _validate_time_to_outcome_col(time_to_outcome_col: str):
#     # make sure time_to_outcome_col input is a string
#
#     if time_to_outcome_col is None:
#         pass
#     elif isinstance(time_to_outcome_col, str):
#         pass
#     else:
#         raise TypeError("time_to_outcome_col must be of type: str")
#
#
# def _surv_convert_to_risk_input_checks(
#         time: float,
#         time_to_outcome_col: str):
#     # make sure _convert_to_risk inputs are kosher
#
#     _validate_time(time=time)
#     _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)
#
#
# def _validate_thresholds(thresholds: list):
#     # make sure thresholds input is a list of floats
#
#     if not isinstance(thresholds, list):
#         raise TypeError('probabilities must be of type list')
#     else:
#         for i in thresholds:
#             if not isinstance(i, float):
#                 raise TypeError("time_to_outcome_col must be of type: float")
#         if not (len(thresholds) > 0):
#             raise ValueError("Thresholds must contain at least 1 value")
#
#
# def _binary_calculate_test_consequences_input_checks(
#         thresholds: list):
#     # make sure _calculate_test_consequences inputs are kosher
#
#     _validate_thresholds(thresholds=thresholds)
#
#
# def _surv_calculate_test_consequences_input_checks(
#         thresholds: list,
#         time: float,
#         time_to_outcome_col: str):
#     # make sure _calculate_test_consequences inputs are kosher
#
#     _validate_thresholds(thresholds=thresholds)
#     _validate_time(time=time)
#     _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)
#
# def _validate_str_list(input_list: list):
#     # Make sure input is a list, and all list elements are strings
#
#     if input_list is None:
#         pass
#     elif not isinstance(input_list, list):
#         raise TypeError('Predictors must be of type list')
#     else:
#         for i in input_list:
#             if not isinstance(i, str):
#                 raise TypeError('All elements in predictors list must be of type str')
#
#
# def _validate_thresh_list(input_list: list):
#     # make sure input_list input is a list, and all list elements are int/float
#
#     if not isinstance(input_list, list):
#         raise TypeError('input must be of type list')
#     else:
#         for i in input_list:
#             if not (isinstance(i, float) or isinstance(i, int)):
#                 raise TypeError("input list contents must be of type: float/int")
#
#     if not (len(input_list) == 3):
#         raise ValueError("input list must contain 3 values")
#
# def _validate_limits_list(input_list: list):
#     # make sure input_list input is a list, and all list elements are int/float
#
#     if not isinstance(input_list, list):
#         raise TypeError('input must be of type list')
#     else:
#         for i in input_list:
#             if not (isinstance(i, float) or isinstance(i, int)):
#                 raise TypeError("input list contents must be of type: float/int")
#
#     if not (len(input_list) == 2):
#         raise ValueError("input list must contain 2 values")
#
#
# def _validate_harm(harm: dict):
#     # make sure harm input is a dict
#
#     if harm is None:
#         pass
#     elif not isinstance(harm, dict):
#         raise TypeError('harm input not of type dict')
#
#
# def _validate_probabilities(probabilities: list, predictors: list):
#     # make sure probabilities input is a list, the length of
#     # probabilities/predictors is the same, and all elements
#     # in probabilities is of type bool
#
#     _validate_str_list(input_list=predictors)
#
#     if probabilities is None:
#         pass
#     elif not isinstance(probabilities, list):
#         raise TypeError('probabilities must be of type list')
#     else:
#         if not (len(probabilities) == len(predictors)):
#             raise ValueError('probabilities input and predictors input must be of same length')
#         for i in probabilities:
#             if not isinstance(i, bool):
#                 raise TypeError('All elements in probabilities must be of type bool')
#
#
# def _binary_dca_input_checks(
#         predictors: list,
#         thresh_vals: list,
#         harm: dict,
#         probabilities: list,  # list of TRUE/FALSE values indicating which predictors
#         ):
#     # make sure dca inputs are kosher
#
#     _validate_str_list(input_list=predictors)
#     _validate_thresh_list(input_list=thresh_vals)
#     _validate_harm(harm=harm)
#     _validate_probabilities(probabilities=probabilities, predictors=predictors)
#
# def _surv_dca_input_checks(
#         predictors: list,
#         thresh_vals: list,
#         harm: dict,
#         probabilities: list,  # list of TRUE/FALSE values indicating which predictors
#         time: float,
#         prevalence: float,
#         time_to_outcome_col: str):
#     # make sure dca inputs are kosher
#
#     _validate_str_list(input_list=predictors)
#     _validate_thresh_list(input_list=thresh_vals)
#     _validate_harm(harm=harm)
#     _validate_probabilities(probabilities=probabilities, predictors=predictors)
#     _validate_time(time=time)
#     _validate_time_to_outcome_col(time_to_outcome_col=time_to_outcome_col)
#
#
# def _validate_graph_type(graph_type: str):
#
#     if not isinstance(graph_type, str):
#         raise TypeError('input must be of type str')
#
#     if graph_type not in ['net_benefit', 'net_intervention_avoided']:
#         raise ValueError('graph_type must be either: net_benefit or net_intervention_avoided')
#
#
# def _plot_graphs_input_checks(
#         y_limits: list,
#         color_names: list,
#         graph_type: str):
#     # make sure output_df input is pandas DataFrame
#
#     _validate_limits_list(input_list=y_limits)
#     _validate_str_list(input_list=color_names)
#     _validate_graph_type(graph_type=graph_type)
#
# def _plot_net_intervention_input_checks(
#         after_dca_df: pd.DataFrame,
#         graph_type: str):
#
#     if graph_type == 'net_intervention_avoided':
#         if 'net_intervention_avoided' not in after_dca_df.columns:
#             raise ValueError("""net_intervention_avoided field not found in inputted dataframe
#             columns - must run net_intervention_avoided() function to calculate net
#             interventions avoided, and reinput returned dataframe into plotting function""")
