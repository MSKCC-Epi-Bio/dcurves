import unittest
from dcurves.dca import dca
from dcurves.load_test_data import load_binary_df, load_survival_df


class TestBinaryDCA(unittest.TestCase):

    def testBinaryOutput(self):

        df_binary = load_binary_df()
        binary_inputs = {
            'data': df_binary,
            'outcome': 'cancer',
            'predictors': ['cancerpredmarker', 'marker'],
            'thresh_vals': [0.01, 0.50, 0.01],
            'harm': None,
            'probabilities': [False, True],
            'time': None,
            'prevalence': None,
            'time_to_outcome_col': None
        }

        binary_output_df = dca(data=binary_inputs['data'],
                               outcome=binary_inputs['outcome'],
                               predictors=binary_inputs['predictors'],
                               thresh_vals=binary_inputs['thresh_vals'],
                               harm=binary_inputs['harm'],
                               probabilities=binary_inputs['probabilities'],
                               time=binary_inputs['time'],
                               prevalence=binary_inputs['prevalence'],
                               time_to_outcome_col=binary_inputs['time_to_outcome_col'])

        print(binary_output_df)

        # assert (isinstance(type(binary_output_df), pd.core.frame.DataFrame)), "should be TRUE"

class TestSurvivalDCA(unittest.TestCase):

    def testSurvivalOutput(self):

        df_surv = load_survival_df()
        survival_inputs = {
            'data': df_surv,
            'outcome': 'cancer',
            'predictors': ['cancerpredmarker'],
            'thresh_vals': [0.01, 0.50, 0.01],
            'harm': None,
            'probabilities': [False],
            'time': 1,
            'prevalence': None,
            'time_to_outcome_col': 'ttcancer'
        }

        survival_output_df = dca(data=survival_inputs['data'],
                                 outcome=survival_inputs['outcome'],
                                 predictors=survival_inputs['predictors'],
                                 thresh_vals=survival_inputs['thresh_vals'],
                                 harm=survival_inputs['harm'],
                                 probabilities=survival_inputs['probabilities'],
                                 time=survival_inputs['time'],
                                 prevalence=survival_inputs['prevalence'],
                                 time_to_outcome_col=survival_inputs['time_to_outcome_col'])

        # assert (isinstance(type(binary_output_df), pd.core.frame.DataFrame)), "should be TRUE"


# def testDefaultArgs():
#     print(DecisionCurveAnalysis.__init__())
#     return(DecisionCurveAnalysis.__init__())
#
# def testBinaryDCA():
#
#     df_binary = load_binary_data()
#
#     binary_inputs = {
#         'data': df_binary,
#         'outcome': 'cancer',
#         'predictors': ['cancerpredmarker', 'marker'],
#         'thresh_lo': 0.01,
#         'thresh_hi': 0.35,
#         'thresh_step': 0.01,
#         'harm': None,
#         'probabilities': [False, True],
#         'time': None,
#         'prevalence': None,
#         'time_to_outcome_col': None
#     }
#
#     binary_output_df = dca(data=binary_inputs['data'],
#                            outcome=binary_inputs['outcome'],
#                            predictors=binary_inputs['predictors'],
#                            thresh_lo=binary_inputs['thresh_lo'],
#                            thresh_hi=binary_inputs['thresh_hi'],
#                            thresh_step=binary_inputs['thresh_step'],
#                            harm=binary_inputs['harm'],
#                            probabilities=binary_inputs['probabilities'],
#                            time=binary_inputs['time'],
#                            prevalence=binary_inputs['prevalence'],
#                            time_to_outcome_col=binary_inputs['time_to_outcome_col'])
#
#     return binary_output_df

# def testSurvivalDCA():
#     df_binary = load_binary_data()
#
#     binary_inputs = {
#         'data': df_binary,
#         'outcome': 'cancer',
#         'predictors': ['cancerpredmarker', 'marker'],
#         'thresh_lo': 0.01,
#         'thresh_hi': 0.35,
#         'thresh_step': 0.01,
#         'harm': None,
#         'probabilities': [False, True],
#         'time': None,
#         'prevalence': None,
#         'time_to_outcome_col': None
#     }
#     print(binary_inputs)
#     pass

# class ConvertToRiskTest(unittest.TestCase):
#
#     def test_convert_to_risk(self):
#         print('test1')
#         pass
#
# class CalculateTestConsequencesTest(unittest.TestCase):
#
#     def test_calculate_test_consequences(self):
#         print('test2')
#         pass
#     pass

# class DCATest(unittest.TestCase):
#
#     data = load_binary_data()
#     outcome = 'cancer'
#     predictors = ['cancerpredmarker', 'marker']
#     thresh_lo = 0.01
#     thresh_hi = 0.99
#     thresh_step = 0.01
#     probabilities = [False, True]
#
#     # r_nb, r_ia = load_r_results('univ_canc_famhist')
#
#     # def test_dca(self):
#     #     p
#
# class PlotNetBenefitGraphsTest(unittest.TestCase):
#
#     def plot_net_benefit_graphs(self):
#         pass
#     pass

if __name__ == '__main__':
    unittest.main()
