
# load risk functions
from dcurves.dca import _create_initial_df
from dcurves.dca import _calc_prevalence
from dcurves.risks import _create_risks_df
from dcurves.dca import _rectify_model_risk_boundaries
# load tools
import pandas as pd
import numpy as np

def test_create_initial_df():

    df_cancer_dx = pd.read_csv("https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv")

    data = df_cancer_dx
    outcome = 'cancer'
    models_to_prob = None
    time = None
    time_to_outcome_col = None
    modelnames = ['famhistory']
    prevalence = None
    thresholds = np.arange(0, 1.01, 0.01)
    harm = None

    risks_df = \
        _create_risks_df(
            data=data,
            outcome=outcome,
            models_to_prob=models_to_prob,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    rectified_risks_df = \
        _rectify_model_risk_boundaries(
            risks_df=risks_df,
            modelnames=modelnames
        )

    # 3. calculate prevalences

    prevalence_value = \
        _calc_prevalence(
            risks_df=rectified_risks_df,
            outcome=outcome,
            prevalence=prevalence,
            time=time,
            time_to_outcome_col=time_to_outcome_col
        )

    # 4. Create initial dataframe for binary/survival cases

    initial_df = \
        _create_initial_df(
            thresholds=thresholds,
            modelnames=modelnames,
            input_df_rownum=len(rectified_risks_df.index),
            prevalence_value=prevalence_value,
            harm=harm
        )

    assert len(initial_df)%3==0
    assert 'all', 'none' in initial_df['model'].value_counts()
    assert len(risks_df) == initial_df['n'][0]
    assert prevalence_value == initial_df['prevalence'][0]
    assert initial_df['harm'][0] == 0


    print(initial_df['model'].value_counts())
    print('\n', initial_df.to_string())

# Note: Need to add tests for when harm is specified to make sure each model has a different associated harm
