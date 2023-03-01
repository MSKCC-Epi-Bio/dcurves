# Load Data
from .load_test_data import load_binary_df

# Load DCA function(s)
from dcurves.dca import dca
from dcurves.plot_graphs import _plot_net_benefit, _plot_net_intervention_avoided, plot_graphs

# Load Libraries Used in Testing
import pandas as pd
import numpy as np
import lifelines

def test_plot_smoothing():


    from typing import List
    import lifelines
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df_time_to_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
        )

    cph = lifelines.CoxPHFitter()
    cph.fit(df=df_time_to_cancer_dx,
            duration_col='ttcancer',
            event_col='cancer',
            formula='age + famhistory + marker')

    cph_pred_vals = \
        cph.predict_survival_function(
            df_time_to_cancer_dx[['age',
                                  'famhistory',
                                  'marker']],
            times=[1.5])

    df_time_to_cancer_dx['pr_failure18'] = [1 - val for val in cph_pred_vals.iloc[0, :]]

    stdca_coxph_results = \
        dca(
            data=df_time_to_cancer_dx,
            outcome='cancer',
            modelnames=['pr_failure18'],
            thresholds=[i/100 for i in range(0, 51)],
            time=1.5,
            time_to_outcome_col='ttcancer'
        )


    def polynomial_smooth(x: List[float], y: List[float], degree: int = 3, alpha: float = None) -> List[float]:
        if alpha is not None:
            y = lifelines.smoothers.spline_smoother(x, y, alpha=alpha)
        else:
            y = lowess(y, x, return_sorted=False, frac=1. / degree)
        return y



    stdca_coxph_results['smoothed_nb'] = polynomial_smooth(x=stdca_coxph_results['threshold'],
                                                           y=stdca_coxph_results['net_benefit'],
                                                           degree=4,
                                                           alpha=1)

    plot_df = stdca_coxph_results
    y_limits = [-0.05, 0.25]

    # plot_graphs(
    #     plot_df=plot_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )
