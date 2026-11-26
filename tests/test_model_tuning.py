"""Tests for src/model_tuning.py (hyper-parameter tuning, Phase 2 enhancement 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as fe, model_training as mt
from src import model_tuning as tu


@pytest.fixture(scope="module")
def train_test(features_df):
    X, y = fe.get_feature_matrix(features_df)
    return fe.split_data(X, y)


def test_search_spaces_cover_all_expected_classifiers():
    expected = {"LogisticRegression", "RandomForestClassifier", "XGBClassifier"}
    assert expected <= set(tu.SEARCH_SPACES)


def test_search_space_keys_all_start_with_model_prefix():
    for clf, space in tu.SEARCH_SPACES.items():
        for k in space:
            assert k.startswith("model__"), f"{clf}: key '{k}' missing 'model__' prefix"


def test_tune_pipeline_returns_fitted_pipeline_and_cv_dataframe(train_test):
    X_train, _, y_train, _ = train_test
    models = mt.get_models(y_train)
    # Use Logistic Regression (fastest) with very few iterations
    pipe = models["Logistic Regression"]
    best, results = tu.tune_pipeline(pipe, X_train, y_train, n_iter=3, seed=42)
    assert hasattr(best, "predict_proba")
    assert "mean_test_score" in results.columns and len(results) == 3
    assert results["mean_test_score"].between(0, 1).all()


def test_tuned_pipeline_produces_valid_probabilities(train_test):
    X_train, X_test, y_train, _ = train_test
    models = mt.get_models(y_train)
    best, _ = tu.tune_pipeline(models["Logistic Regression"], X_train, y_train, n_iter=3, seed=42)
    prob = best.predict_proba(X_test)[:, 1]
    assert prob.shape == (len(X_test),)
    assert np.all((prob >= 0) & (prob <= 1))


def test_compare_on_test_returns_two_rows_with_expected_columns(train_test):
    X_train, X_test, y_train, y_test = train_test
    models = mt.get_models(y_train)
    baseline = models["Logistic Regression"].fit(X_train, y_train)
    best, _ = tu.tune_pipeline(models["Logistic Regression"], X_train, y_train, n_iter=3, seed=42)
    comp = tu.compare_on_test(baseline, best, X_test, y_test)
    assert len(comp) == 2
    for col in ("test_roc_auc", "test_accuracy", "test_recall", "test_f1"):
        assert col in comp.columns
        assert comp[col].between(0, 1).all()


def test_tuning_report_is_written(train_test, tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test
    models = mt.get_models(y_train)
    baseline = models["Logistic Regression"].fit(X_train, y_train)
    best, rscv_results = tu.tune_pipeline(models["Logistic Regression"], X_train, y_train, n_iter=3, seed=42)
    comp = tu.compare_on_test(baseline, best, X_test, y_test)
    tu.write_report("Logistic Regression", best, rscv_results, comp)
    report = tmp_path / "tuning_results.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "best inner-cv" in text.lower() or "best hyper-parameters" in text.lower()
    assert "test-set comparison" in text.lower()


def test_no_search_space_raises_value_error(train_test):
    X_train, _, y_train, _ = train_test
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyClassifier
    dummy = Pipeline([("model", DummyClassifier())])
    with pytest.raises(ValueError, match="No search space"):
        tu.tune_pipeline(dummy, X_train, y_train, n_iter=2)
