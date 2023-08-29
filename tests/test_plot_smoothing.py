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

    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt

    import pandas
    pandas.set_option('expand_frame_repr', False)

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

    # plot_graphs(
    #     plot_df=stdca_coxph_results,
    #     graph_type='net_benefit'
    # )

    # print('\n')
    # print(stdca_coxph_results)

    # lowess = sm.nonparametric.lowess(stdca_coxph_results['net_benefit'].values, stdca_coxph_results['threshold'].values, frac=0.1)
    #
    # # print(lowess)
    # stdca_coxph_results['smoothed_nb'] = lowess[:,1].copy()
    # # print('\n')
    # # print(stdca_coxph_results)
    #
    # import matplotlib.pyplot as plt
    #
    # # Scatter plot for the regular 'net_benefit'
    # plt.scatter(stdca_coxph_results['threshold'], stdca_coxph_results['net_benefit'], label='net_benefit', alpha=0.5)
    #
    # # Line plot for the smoothed 'net_benefit'
    # plt.plot(stdca_coxph_results['threshold'], stdca_coxph_results['smoothed_nb'], label='smoothed_net_benefit',
    #          color='red')
    #
    # # Add labels and legend
    # plt.xlabel('Threshold')
    # plt.ylabel('Net Benefit')
    # plt.title('Net Benefit vs Threshold')
    # plt.legend()
    #
    # plt.show()

    # sm.nonparametric.lowess(y, x, frac=0.3)

    # def polynomial_smooth(x: List[float], y: List[float], degree: int = 3, alpha: float = None) -> List[float]:
    #     if alpha is not None:
    #         y = lifelines.smoothers.spline_smoother(x, y, alpha=alpha)
    #     else:
    #         y = lowess(y, x, return_sorted=False, frac=1. / degree)
    #     return y
    #
    # stdca_coxph_results['smoothed_nb'] = polynomial_smooth(x=stdca_coxph_results['threshold'],
    #                                                        y=stdca_coxph_results['net_benefit'],
    #                                                        degree=4,
    #                                                        alpha=1)

    # plot_df = stdca_coxph_results
    # y_limits = [-0.05, 0.25]

    # plot_graphs(
    #     plot_df=plot_df,
    #     graph_type='net_benefit',
    #     y_limits=[-0.05, 0.25]
    # )
