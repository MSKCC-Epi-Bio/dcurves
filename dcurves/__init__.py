import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
import matplotlib.pyplot as plt
import dcurves.dca as dca
import dcurves.load_test_data as load_data
import dcurves.validate as val
from dcapy.validate import DCAError

__all__ = ['DecisionCurveAnalysis']  # only public member should be the class

class DecisionCurveAnalysis:

    # outcome: str,
    # predictors: list,
    # thresh_lo: float,
    # thresh_hi: float,
    # thresh_step: float,
    # harm: dict,
    # probabilities: list,
    # time: float,
    # prevalence: float,
    # time_to_outcome_col: str) -> pd.DataFrame:

    # universal parameters for dca
    _common_args = {'outcome': None,
                    'predictors': None,
                    'thresh_lo': 0.01,
                    'thresh_hi': 0.99,
                    'thresh_step': 0.01,
                    'harms': None,
                    'probabilities': None,
                    'data': None,

                    }

    # stdca-specific attributes
    _stdca_args = {'tt_outcome': None,
                   'time_point': None,
                   'cmp_risk': False}


    def __init__(self, algorithm='dca', **kwargs):
        """Initializes the DecisionCurveAnalysis object

        Arguments for the analysis may be passed in as keywords upon object initialization

        Parameters
        ----------
        algorithm : str
            the algorithm to use, valid options are 'dca' or 'stdca'
        **kwargs :
            keyword arguments to populate instance attributes that will be used in analysis

        Raises
        ------
        ValueError
            if user doesn't specify a valid algorithm; valid values are 'dca' or 'stdca'
            if the user specifies an invalid keyword
        """

        if algorithm not in ['dca', 'stdca']:
            raise ValueError("did not specify a valid algorithm, only 'dca' and 'stdca' are valid")
        self.algorithm = algorithm

        # set args based on keywords passed in
        # this naively assigns values passed in -- validation occurs afterwords
        for kw in kwargs:
            if kw in self._common_args:
                self._common_args[kw] = kwargs[kw]  # assign
                continue
            elif kw in self._stdca_args:
                self._stdca_args[kw] = kwargs[kw]
            else:
                raise ValueError("{kw} is not a valid DCA keyword"
                                 .format(kw=repr(kw)))

        print(self._common_args)

        # do validation on all args, make sure we still have a valid analysis
        self.data = val.data_validate(self.data)
        self.outcome = val.outcome_validate(self.data, self.outcome)
        self.predictors = val.predictors_validate(self.predictors, self.data)
        # validate bounds
        new_bounds = []
        curr_bounds = [self._common_args['thresh_lo'], self._common_args['thresh_hi'],
                       self._common_args['thresh_step']]
        for i, bound in enumerate(['lower', 'upper', 'step']):
            new_bounds.append(val.threshold_validate(bound, self.threshold_bound(bound),
                                                     curr_bounds))

        print('test1')
        #self.set_threshold_bounds(lower=new_bounds[0], upper=new_bounds[1], step=new_bounds[2])
        self.set_threshold_bounds(lower=curr_bounds[0], upper=curr_bounds[1], step=curr_bounds[2])

        # validate predictor-reliant probs/harms
        self.probabilities = val.probabilities_validate(self.probabilities,
                                                        self.predictors)
        self.harms = val.harms_validate(self.harms, self.predictors)
        # validate the data in each predictor column
        self.data = val.validate_data_predictors(self.data, self.outcome, self.predictors,
                                                 self.probabilities)

    def _args_dict(self):
        """Forms the arguments to pass to the analysis algorithm

        Returns
        -------
        dict(str, object)
            A dictionary that can be unpacked and passed to the algorithm for the
            analysis
        """

        if self.algorithm == 'dca':
            return self._common_args
        else:
            from collections import Counter
            return dict(Counter(self._common_args) + Counter(self._stdca_args))

    def _algo(self):
        """The algorithm to use for this analysis
        """
        return dca.dca #### Right now, dca function in dca.py file handles both dca and stdca
            #if self.algorithm == 'dca' else algo.stdca




    def run(self, return_results=False):
        """Performs the analysis

        Parameters
        ----------
        return_results : bool
            if `True`, sets the results to the instance attribute `results`
            if `False` (default), the function returns the results as a tuple

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            Returns net_benefit, interventions_avoided if `return_results=True`
        """
        all_covariates_df = self._algo()(**(self._args_dict()))
        if return_results:
            return all_covariates_df
        else:
            self.results = all_covariates_df


    def plot_net_benefit(self, all_covariates_df):


        dca.plot_net_benefit_graphs(all_covariates_df)














