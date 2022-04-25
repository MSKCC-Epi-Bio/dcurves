

import unittest
from dcurves.dca import dca

from dcurves.test import load_binary_data


# def testDCA():
#     # DecisionCurveAnalysis.

def testMath():
    assert 5 == 5

# def testDefaultArgs():
#     print(DecisionCurveAnalysis.__init__())
#     return(DecisionCurveAnalysis.__init__())

def testBinaryDCA():

    df_binary = load_binary_data()

    binary_inputs = {
        'data': df_binary,
        'outcome': 'cancer',
        'predictors': ['cancerpredmarker', 'marker'],
        'thresh_lo': 0.01,
        'thresh_hi': 0.35,
        'thresh_step': 0.01,
        'harm': None,
        'probabilities': [False, True],
        'time': None,
        'prevalence': None,
        'time_to_outcome_col': None
    }

    binary_output_df = dca(data=binary_inputs['data'],
                           outcome=binary_inputs['outcome'],
                           predictors=binary_inputs['predictors'],
                           thresh_lo=binary_inputs['thresh_lo'],
                           thresh_hi=binary_inputs['thresh_hi'],
                           thresh_step=binary_inputs['thresh_step'],
                           harm=binary_inputs['harm'],
                           probabilities=binary_inputs['probabilities'],
                           time=binary_inputs['time'],
                           prevalence=binary_inputs['prevalence'],
                           time_to_outcome_col=binary_inputs['time_to_outcome_col'])

    # print(binary_inputs['data'])

    return binary_output_df

def testSurvivalDCA():
    df_binary = load_binary_data()

    binary_inputs = {
        'data': df_binary,
        'outcome': 'cancer',
        'predictors': ['cancerpredmarker', 'marker'],
        'thresh_lo': 0.01,
        'thresh_hi': 0.35,
        'thresh_step': 0.01,
        'harm': None,
        'probabilities': [False, True],
        'time': None,
        'prevalence': None,
        'time_to_outcome_col': None
    }

    pass

class ConvertToRiskTest(unittest.TestCase):

    def test_convert_to_risk(self):
        pass

class CalculateTestConsequencesTest(unittest.TestCase):

    def test_calculate_test_consequences(self):
        pass
    pass

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

# if __name__ == '__main__':
#
#     print(DecisionCurveAnalysis.__init__().data)
