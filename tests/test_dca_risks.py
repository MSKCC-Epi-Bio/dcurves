# Load data
import dcurves
from dcurves.load_test_data import load_tutorial_marker_risk_scores

# load risk functions

from dcurves.risks import _calc_binary_risks
from dcurves.risks import _create_risks_df

# load tools
import pandas as pd
import numpy as np

def test_dca_risks_calc():

    r_marker_risks = load_tutorial_marker_risk_scores()

    df_cancer_dx = \
        pd.read_csv(
            "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
        )

    p_marker_risks = \
        _calc_binary_risks(
            data=df_cancer_dx,
            outcome='cancer',
            model='marker'
        )

    r_marker_risks = r_marker_risks['marker_risk'].tolist()

    # print(p_marker_risks==r_marker_risks)

    # # print(type(r_marker_risks['marker_risk'].tolist()), ' ', type(p_marker_risks))
    # # print(sorted(p_marker_risks))
    #

    marker_compare_df = \
        pd.DataFrame({'r_marker_risks': sorted(r_marker_risks),
                      'p_marker_risks': sorted(p_marker_risks)})

    print(marker_compare_df.to_string())

    for i in range(0,len(marker_compare_df)):
        if marker_compare_df['r_marker_risks'][i] != marker_compare_df['p_marker_risks'][i]:
            print(i)

    # print('\n', marker_compare_df)
    # print('r_marker_risks')
    # print('\n', marker_compare_df['r_marker_risks'].describe())
    # print('p_marker_risks')
    # print('\n', marker_compare_df['p_marker_risks'].describe())


