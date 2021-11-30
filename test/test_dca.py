

import unittest
from dcurves import DecisionCurveAnalysis
from dcurves.dca import dca
from test import load_r_results, load_default_data, load_binary_df, load_survival_df


class ConvertToRiskTest(unittest.TestCase):

    def test_convert_to_risk(self):
        pass


class CalculateTestConsequencesTest(unittest.TestCase):

    def test_calculate_test_consequences(self):
        pass
    pass

class DCATest(unittest.TestCase):

    data = load_binary_df()
    outcome = 'cancer'
    predictors = ['cancerpredmarker', 'marker']
    thresh_lo = 0.01
    thresh_hi = 0.99
    thresh_step = 0.01
    probabilities = [False, True]

    r_nb, r_ia = load_r_results('univ_canc_famhist')

    def test_dca(self):
        pass



class PlotNetBenefitGraphsTest(unittest.TestCase):

    def plot_net_benefit_graphs(self):
        pass
    pass


if __name__ == '__main__':
    unittest.main()
