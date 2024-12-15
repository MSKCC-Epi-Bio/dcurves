"""
This module contains tests for the DCA (Decision Curve Analysis) tutorial examples.
It covers various scenarios including binary outcomes, survival analysis, and case-control studies.
"""

import pandas as pd
import statsmodels.api as sm
import lifelines
from sklearn.model_selection import RepeatedKFold

from dcurves import dca
from .load_test_data import load_r_dca_famhistory


def test_python_model():
    """Test creation of a simple GLM model."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )
    mod = sm.GLM.from_formula(
        "cancer ~ famhistory", data=df_cancer_dx, family=sm.families.Binomial()
    )
    mod.fit()


def test_python_famhistory1():
    """Test DCA with family history as a predictor."""
    df_r_dca_famhistory = (
        load_r_dca_famhistory()
        .sort_values(by=["variable", "threshold"], ascending=[True, True])
        .reset_index(drop=True)
    )

    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca_result_df = (
        dca(data=df_cancer_dx, outcome="cancer", modelnames=["famhistory"])
        .sort_values(by=["model", "threshold"], ascending=[True, True])
        .reset_index(drop=True)
    )

    for model in ["all", "none", "famhistory"]:
        r_nb = (
            df_r_dca_famhistory[df_r_dca_famhistory.variable == model]["net_benefit"]
            .round(decimals=6)
            .reset_index(drop=True)
        )
        p_nb = (
            dca_result_df[dca_result_df.model == model]["net_benefit"]
            .round(decimals=6)
            .reset_index(drop=True)
        )

        assert r_nb.equals(p_nb)


def test_python_famhistory2():
    """Test DCA with family history and custom thresholds."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["famhistory"],
        thresholds=[i / 100 for i in range(36)],
    )


def test_python_model_multi():
    """Test creation of a multi-predictor GLM model."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    mod = sm.GLM.from_formula(
        "cancer ~ marker + age + famhistory",
        data=df_cancer_dx,
        family=sm.families.Binomial(),
    )
    mod.fit()


def test_python_dca_multi():
    """Test DCA with multiple predictors."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["famhistory", "cancerpredmarker"],
        thresholds=[i / 100 for i in range(36)],
    )


def test_python_pub_model():
    """Test DCA with a published model."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    df_cancer_dx["logodds_brown"] = (
        0.75 * df_cancer_dx["famhistory"] + 0.26 * df_cancer_dx["age"] - 17.5
    )
    df_cancer_dx["phat_brown"] = 1 / (1 + (1 / (2.718281828 ** df_cancer_dx["logodds_brown"])))

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["phat_brown"],
        thresholds=[i / 100 for i in range(36)],
    )


def test_python_joint():
    """Test creation of joint and conditional risk variables."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    df_cancer_dx["high_risk"] = df_cancer_dx["risk_group"].apply(lambda x: 1 if x == "high" else 0)

    df_cancer_dx["joint"] = df_cancer_dx.apply(
        lambda x: 1 if x["risk_group"] == "high" or x["cancerpredmarker"] > 0.15 else 0,
        axis=1,
    )

    df_cancer_dx["conditional"] = df_cancer_dx.apply(
        lambda x: (
            1
            if x["risk_group"] == "high"
            or (x["risk_group"] == "intermediate" and x["cancerpredmarker"] > 0.15)
            else 0
        ),
        axis=1,
    )


def test_python_dca_joint():
    """Test DCA with joint and conditional risk variables."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    df_cancer_dx["high_risk"] = df_cancer_dx["risk_group"].apply(lambda x: 1 if x == "high" else 0)

    df_cancer_dx["joint"] = df_cancer_dx.apply(
        lambda x: 1 if x["risk_group"] == "high" or x["cancerpredmarker"] > 0.15 else 0,
        axis=1,
    )

    df_cancer_dx["conditional"] = df_cancer_dx.apply(
        lambda x: (
            1
            if x["risk_group"] == "high"
            or (x["risk_group"] == "intermediate" and x["cancerpredmarker"] > 0.15)
            else 0
        ),
        axis=1,
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["high_risk", "joint", "conditional"],
        thresholds=[i / 100 for i in range(36)],
    )


def test_python_dca_harm_simple():
    """Test DCA with simple harm incorporation."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["marker"],
        thresholds=[i / 100 for i in range(36)],
        harm={"marker": 0.0333},
        models_to_prob=["marker"],
    )


def test_python_dca_harm():
    """Test DCA with more complex harm incorporation."""

    # pylint: disable=R0801
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    harm_marker = 0.0333
    intermediate_mask = df_cancer_dx["risk_group"] == "intermediate"
    harm_conditional = intermediate_mask.mean() * harm_marker

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["risk_group"],
        models_to_prob=["risk_group"],
        thresholds=[i / 100 for i in range(36)],
        harm={"risk_group": harm_conditional},
    )
    # pylint: enable=R0801


def test_python_dca_table():
    """Test DCA table generation."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["marker"],
        models_to_prob=["marker"],
        thresholds=[i / 100 for i in range(36)],
    )


def test_python_dca_intervention():
    """Test DCA with intervention avoided calculation."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    dca(
        data=df_cancer_dx,
        outcome="cancer",
        modelnames=["marker"],
        thresholds=[i / 100 for i in range(36)],
        models_to_prob=["marker"],
    )


# pylint: disable=duplicate-code


def test_python_coxph():
    """Test Cox proportional hazards model."""
    df_time_to_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/"
        "main/data/df_time_to_cancer_dx.csv"
    )

    cph = lifelines.CoxPHFitter()
    cph.fit(
        df=df_time_to_cancer_dx,
        duration_col="ttcancer",
        event_col="cancer",
        formula="age + famhistory + marker",
    )

    cph_pred_vals = cph.predict_survival_function(
        df_time_to_cancer_dx[["age", "famhistory", "marker"]], times=[1.5]
    )

    df_time_to_cancer_dx["pr_failure18"] = 1 - cph_pred_vals.iloc[0, :]


# pylint: enable=duplicate-code


def test_python_stdca_coxph():
    """Test standardized DCA with Cox proportional hazards model."""

    # pylint: disable=duplicate-code
    df_time_to_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/"
        "main/data/df_time_to_cancer_dx.csv"
    )

    cph = lifelines.CoxPHFitter()
    cph.fit(
        df=df_time_to_cancer_dx,
        duration_col="ttcancer",
        event_col="cancer",
        formula="age + famhistory + marker",
    )
    # pylint: enable=duplicate-code

    cph_pred_vals = cph.predict_survival_function(
        df_time_to_cancer_dx[["age", "famhistory", "marker"]], times=[1.5]
    )

    df_time_to_cancer_dx["pr_failure18"] = 1 - cph_pred_vals.iloc[0, :]

    dca(
        data=df_time_to_cancer_dx,
        outcome="cancer",
        modelnames=["pr_failure18"],
        thresholds=[i / 100 for i in range(51)],
        time=1.5,
        time_to_outcome_col="ttcancer",
    )


def test_python_dca_case_control():
    """Test DCA with case-control data."""
    df_case_control = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/"
        "main/data/df_cancer_dx_case_control.csv"
    )

    dca(
        data=df_case_control,
        outcome="casecontrol",
        modelnames=["cancerpredmarker"],
        prevalence=0.20,
        thresholds=[i / 100 for i in range(51)],
    )


def test_python_cross_validation():
    """Test DCA with cross-validation."""
    df_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_cancer_dx.csv"
    )

    formula = "cancer ~ marker + age + famhistory"
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=112358)
    cv_predictions = []

    for train_index, test_index in rkf.split(df_cancer_dx):
        train = df_cancer_dx.iloc[train_index].copy()
        test = df_cancer_dx.iloc[test_index].copy()
        model = sm.Logit.from_formula(formula, data=train).fit(disp=0)
        test.loc[:, "cv_prediction"] = model.predict(test)
        cv_predictions.append(test[["patientid", "cv_prediction"]])

    df_predictions = pd.concat(cv_predictions)
    df_mean_predictions = df_predictions.groupby("patientid")["cv_prediction"].mean().reset_index()
    df_cv_pred = pd.merge(df_cancer_dx, df_mean_predictions, on="patientid", how="left")

    df_dca_cv = dca(data=df_cv_pred, modelnames=["cv_prediction"], outcome="cancer")

    assert isinstance(df_dca_cv, pd.DataFrame), "df_dca_cv is not a pandas DataFrame"
