import dcurves

dcurves.dca(
    data=dcurves.load_test_data.load_survival_df(),
    outcome='cancer',
    predictors=['cancerpredmarker'],
    thresh_vals=[0.01, 1.0, 0.01],
    probabilities=[False],
    time=1,
    time_to_outcome_col='ttcancer'
            )

