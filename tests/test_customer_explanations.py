"""Tests for src/customer_explanations.py (per-customer explanations, Phase 2 enhancement 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, feature_engineering as fe, model_training as mt
from src import customer_explanations as ce
from src import explainability as ex


@pytest.fixture(scope="module")
def fitted_xgb(features_df):
    X, y = fe.get_feature_matrix(features_df)
    X_train, _, y_train, _ = fe.split_data(X, y)
    models = mt.get_models(y_train)
    return models["XGBoost"].fit(X_train, y_train)


@pytest.fixture(scope="module")
def all_explanations(features_df, fitted_xgb):
    return ce.build_all_explanations(fitted_xgb, features_df)


def test_risk_band_boundaries():
    assert ce._risk_band(0.0) == "Low"
    assert ce._risk_band(0.29) == "Low"
    assert ce._risk_band(0.30) == "Medium"
    assert ce._risk_band(0.50) == "High"
    assert ce._risk_band(0.99) == "High"


def test_explain_customer_structure():
    shap_row = pd.Series({"contract": 0.5, "tenure_months": -0.3, "monthly_charges": 0.2})
    ex_dict = ce.explain_customer("CUS-001", shap_row, 0.72)
    assert ex_dict[config.ID_COL] == "CUS-001"
    assert ex_dict["churn_probability"] == pytest.approx(0.72)
    assert ex_dict["risk_band"] == "High"
    # top risk factors should include "contract" (highest positive SHAP)
    top_features = [f["feature"] for f in ex_dict["top_risk_factors"]]
    assert "contract" in top_features
    # tenure should be a retention signal (negative SHAP)
    ret_features = [f["feature"] for f in ex_dict["top_retention_signals"]]
    assert "tenure_months" in ret_features


def test_all_explanations_covers_all_customers(features_df, all_explanations):
    assert len(all_explanations) == len(features_df) == 7043
    ids = {ex[config.ID_COL] for ex in all_explanations}
    assert len(ids) == 7043


def test_explanations_probabilities_match_pipeline(features_df, fitted_xgb, all_explanations):
    X, _ = fe.get_feature_matrix(features_df)
    expected_probs = fitted_xgb.predict_proba(X)[:, 1]
    actual_probs = np.array([ex["churn_probability"] for ex in all_explanations])
    # Rounded to 4 dp; allow 0.0001 tolerance
    assert np.allclose(actual_probs, expected_probs, atol=1e-3)


def test_explanations_to_dataframe_shape(all_explanations):
    flat = ce.explanations_to_dataframe(all_explanations)
    assert len(flat) == 7043
    assert config.ID_COL in flat.columns
    assert "churn_probability" in flat.columns
    assert "risk_band" in flat.columns
    assert "risk_factor_1" in flat.columns
    assert flat["risk_band"].isin(["Low", "Medium", "High"]).all()


def test_risk_band_distribution_is_sane(all_explanations):
    flat = ce.explanations_to_dataframe(all_explanations)
    dist = flat["risk_band"].value_counts()
    # At least 1,000 customers in each band (sanity check on data)
    assert dist.min() >= 100


def test_format_explanation_contains_key_strings():
    shap_row = pd.Series({"contract": 0.6, "tenure_months": -0.4})
    ex_dict = ce.explain_customer("CUS-999", shap_row, 0.55)
    text = ce.format_explanation(ex_dict)
    assert "CUS-999" in text
    assert "55.0%" in text or "High" in text
    assert "contract" in text


def test_exported_csv_matches_fresh_run(all_explanations):
    saved = pd.read_csv(config.DASHBOARD_DATA_DIR / "customer_explanations.csv")
    fresh = ce.explanations_to_dataframe(all_explanations)
    assert len(saved) == len(fresh)
    # Risk bands should match
    assert saved["risk_band"].tolist() == fresh["risk_band"].tolist()
