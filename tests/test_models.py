"""Tests for model training / evaluation helpers and model-input preparation.

The end-to-end checks use fast models (logistic regression) so the suite stays quick;
the full model comparison lives in ``python -m src.model_training`` / ``model_evaluation``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config, feature_engineering as fe, model_evaluation as ev, model_training as mt


@pytest.fixture(scope="module")
def split(features_df):
    X, y = fe.get_feature_matrix(features_df)
    return fe.split_data(X, y)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def test_get_models_returns_six_pipelines_with_prep_step(split):
    _, _, y_train, _ = split
    models = mt.get_models(y_train)
    assert set(models) == {
        "Logistic Regression", "Logistic Regression (balanced)", "Random Forest",
        "Random Forest (balanced)", "XGBoost", "XGBoost (balanced)",
    }
    for pipe in models.values():
        assert list(pipe.named_steps) == ["prep", "model"]


def test_xgboost_class_weight_comes_from_training_labels_only(split):
    _, _, y_train, y_test = split
    spw = mt.get_models(y_train)["XGBoost (balanced)"].named_steps["model"].scale_pos_weight
    assert spw == pytest.approx((y_train == 0).sum() / (y_train == 1).sum())
    assert spw != pytest.approx((y_test == 0).sum() / (y_test == 1).sum(), abs=1e-6)


def test_logistic_pipeline_beats_baseline_on_holdout(split):
    X_train, X_test, y_train, y_test = split
    pipe = fe.make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    assert auc > 0.80  # the signal is real: well above the 0.5 of a random ranking


def test_shuffled_labels_give_chance_level_auc_so_no_leakage(split):
    """If any feature leaked the target, a model trained on shuffled labels would still score well
    on held-out *shuffled* labels only by chance - AUC must stay near 0.5."""
    X_train, X_test, y_train, y_test = split
    rng = np.random.default_rng(0)
    y_shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
    pipe = fe.make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_shuffled)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    assert 0.44 < auc < 0.56


def test_baseline_predicts_majority_class(split):
    X_train, X_test, y_train, y_test = split
    base = mt.get_baseline().fit(X_train, y_train)
    assert (base.predict(X_test) == 0).all()


def test_select_champion_uses_cv_auc_only():
    cv = pd.DataFrame({"model": ["A", "B", "C"], "cv_roc_auc_mean": [0.80, 0.85, 0.83]})
    assert mt.select_champion(cv) == "B"
    assert mt.select_champion(cv, ["A", "C"]) == "C"


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def test_classification_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.2, 0.3])
    m = ev.classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))


def test_majority_class_has_high_accuracy_but_zero_recall():
    y = np.array([0] * 74 + [1] * 26)
    m = ev.classification_metrics(y, np.zeros(100), 0.5)
    assert m["accuracy"] == pytest.approx(0.74)
    assert m["recall"] == 0.0 and m["precision"] == 0.0


def test_choose_threshold_is_in_grid_and_improves_f1():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.25).astype(int)
    p = np.clip(0.25 * y + rng.normal(0.25, 0.12, 2000), 0, 1)
    t = ev.choose_threshold(y, p)
    assert 0.1 <= t <= 0.9
    f_best = ev.classification_metrics(y, p, t)["f1"]
    assert f_best >= ev.classification_metrics(y, p, 0.5)["f1"]


def test_lift_table_properties(split):
    X_train, X_test, y_train, y_test = split
    pipe = fe.make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    lift = ev.lift_table(y_test, pipe.predict_proba(X_test)[:, 1])
    assert len(lift) == 10
    assert lift["customers"].sum() == len(y_test)
    assert lift["churners"].sum() == y_test.sum()
    assert lift["cum_share_of_churners"].iloc[-1] == pytest.approx(1.0)
    assert lift["lift"].iloc[0] > 1.5          # top decile much riskier than average
    assert lift["lift"].iloc[0] > lift["lift"].iloc[-1]
    assert lift["cum_share_of_churners"].is_monotonic_increasing


def test_permutation_importance_ranks_contract_and_tenure_high(split):
    X_train, X_test, y_train, y_test = split
    pipe = fe.make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    imp = ev.permutation_importance_table(pipe, X_test, y_test, n_repeats=3)
    assert set(imp["feature"]) == set(fe.MODEL_FEATURES)
    assert {"contract", "tenure_months"} <= set(imp["feature"].head(4))
    gender = imp.set_index("feature").loc["gender", "importance_mean"]
    assert abs(gender) < 0.01


def test_logistic_odds_ratios_signs(split):
    X_train, _, y_train, _ = split
    pipe = fe.make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    odds = ev.logistic_odds_ratios(pipe).set_index("feature")
    assert odds.loc["tenure_months", "odds_ratio"] < 1       # longer tenure -> lower churn odds
    assert odds.loc["internet_service_Fiber optic", "odds_ratio"] > 1
    assert odds.loc["contract_Two year", "odds_ratio"] < 1
    assert np.allclose(odds["odds_ratio"], np.exp(odds["coefficient"]))
