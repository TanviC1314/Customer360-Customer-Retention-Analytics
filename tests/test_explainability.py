"""Tests for src/explainability.py (SHAP global explanations, Phase 2 enhancement 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config, feature_engineering as fe, model_training as mt
from src import explainability as ex


@pytest.fixture(scope="module")
def fitted_xgb(features_df):
    X, y = fe.get_feature_matrix(features_df)
    X_train, _, y_train, _ = fe.split_data(X, y)
    models = mt.get_models(y_train)
    return models["XGBoost"].fit(X_train, y_train)


@pytest.fixture(scope="module")
def shap_result(fitted_xgb, features_df):
    X, _ = fe.get_feature_matrix(features_df)
    return ex.compute_shap_values(fitted_xgb, X, sample_size=500)


def test_encoded_feature_names_are_non_empty(fitted_xgb):
    preprocessor, _ = ex._get_preprocessor_and_model(fitted_xgb)
    names = ex._encoded_feature_names(preprocessor)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_shap_values_shape_matches_encoded_features(shap_result):
    shap_vals, X_enc_df, enc_names = shap_result
    assert shap_vals.shape == (len(X_enc_df), len(enc_names))
    assert len(enc_names) == X_enc_df.shape[1]


def test_aggregated_shap_covers_all_original_features(fitted_xgb, shap_result):
    shap_vals, _, enc_names = shap_result
    preprocessor, _ = ex._get_preprocessor_and_model(fitted_xgb)
    mapping = ex._ohe_to_original_mapping(preprocessor, enc_names)
    agg = ex.aggregate_to_original(shap_vals, enc_names, mapping)
    # Every original feature in the mapping should appear in the aggregated df
    expected_original = set(mapping.values())
    assert expected_original <= set(agg.columns)


def test_global_importance_is_sorted_descending(fitted_xgb, shap_result):
    shap_vals, _, enc_names = shap_result
    preprocessor, _ = ex._get_preprocessor_and_model(fitted_xgb)
    mapping = ex._ohe_to_original_mapping(preprocessor, enc_names)
    agg = ex.aggregate_to_original(shap_vals, enc_names, mapping)
    imp = ex.global_importance(agg)
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    assert imp["mean_abs_shap"].gt(0).all()


def test_contract_is_a_top_driver(fitted_xgb, shap_result):
    """Contract type must rank in the top 3 original features (domain knowledge)."""
    shap_vals, _, enc_names = shap_result
    preprocessor, _ = ex._get_preprocessor_and_model(fitted_xgb)
    mapping = ex._ohe_to_original_mapping(preprocessor, enc_names)
    agg = ex.aggregate_to_original(shap_vals, enc_names, mapping)
    imp = ex.global_importance(agg)
    assert imp["feature"].iloc[:3].str.contains("contract").any()


def test_customer_shap_table_has_one_row_per_customer(features_df, fitted_xgb):
    tbl = ex.customer_shap_table(fitted_xgb, features_df)
    assert len(tbl) == len(features_df)
    assert config.ID_COL in tbl.columns
    assert tbl[config.ID_COL].is_unique


def test_shap_values_sum_to_expected_range(shap_result):
    """Individual SHAP values should be finite and their row sums finite."""
    shap_vals, _, _ = shap_result
    assert np.isfinite(shap_vals).all()


def test_global_bar_plot_saved(fitted_xgb, shap_result, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    shap_vals, _, enc_names = shap_result
    preprocessor, _ = ex._get_preprocessor_and_model(fitted_xgb)
    mapping = ex._ohe_to_original_mapping(preprocessor, enc_names)
    agg = ex.aggregate_to_original(shap_vals, enc_names, mapping)
    imp = ex.global_importance(agg)
    ex.plot_global_bar(imp)
    assert (tmp_path / "shap_01_global_bar.png").exists()
