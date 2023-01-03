# Load Data
from dcurves.load_test_data import load_binary_df, load_survival_df, load_case_control_df

# Load _calc_prevalence
from dcurves.dca import _calc_prevalence

def test_binary_prevalence():

    binary_df = load_binary_df()

    local_prevalence_calc = \
        _calc_prevalence(
            risks_df=binary_df,
            outcome='cancer',
            prevalence=None
        )

    assert local_prevalence_calc == 0.14

def test_survival_prevalence():

    surv_df = load_survival_df()

    local_prevalence_calc = \
        _calc_prevalence(
            risks_df=surv_df,
            outcome='cancer',
            prevalence=None,
            time=1,
            time_to_outcome_col='ttcancer'
        )

    assert round(number=float(local_prevalence_calc),
                 ndigits=6) == 0.147287
