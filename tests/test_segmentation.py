"""Tests for src/segmentation.py (K-Means customer segmentation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, segmentation as sg


@pytest.fixture(scope="module")
def result(features_df):
    return sg.segment(features_df, k=5)


def test_cluster_inputs_exclude_the_target(features_df):
    inputs = sg.build_cluster_inputs(features_df)
    assert list(inputs.columns) == sg.CLUSTER_FEATURES
    assert not {"churn", config.TARGET} & set(inputs.columns)
    assert not inputs.isna().any().any()


def test_clustering_is_unaffected_by_churn_values(features_df):
    """Flipping every churn label must not move a single customer between groups."""
    flipped = features_df.copy()
    flipped[config.TARGET] = 1 - flipped[config.TARGET]
    a = sg.fit_kmeans(sg.scale(sg.build_cluster_inputs(features_df)), 5).labels_
    b = sg.fit_kmeans(sg.scale(sg.build_cluster_inputs(flipped)), 5).labels_
    assert np.array_equal(a, b)


def test_scaled_inputs_have_zero_mean_unit_variance(features_df):
    Z = sg.scale(sg.build_cluster_inputs(features_df))
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1)


def test_every_customer_gets_exactly_one_segment(features_df, result):
    assignments, profile, _ = result
    assert len(assignments) == len(features_df) == 7043
    assert assignments[config.ID_COL].is_unique
    assert profile["customers"].sum() == 7043
    assert profile["share_of_customers_pct"].sum() == pytest.approx(100, abs=0.1)
    assert profile["share_of_churners_pct"].sum() == pytest.approx(100, abs=0.1)
    assert profile["cluster_name"].is_unique


def test_profile_reconciles_with_assignments(features_df, result):
    assignments, profile, _ = result
    merged = features_df.merge(assignments, on=config.ID_COL)
    expected = merged.groupby("cluster")[config.TARGET].mean() * 100
    for _, row in profile.iterrows():
        assert row["churn_rate_pct"] == pytest.approx(expected[row["cluster"]], abs=0.01)
    active = merged.loc[merged[config.TARGET] == 0, "monthly_charges"].sum()
    assert profile["active_mrr"].sum() == pytest.approx(active, abs=1)


def test_clusters_are_ordered_by_churn_rate(result):
    _, profile, _ = result
    assert profile["cluster"].tolist() == list(range(len(profile)))
    assert profile["churn_rate_pct"].is_monotonic_decreasing


def test_segmentation_is_reproducible(features_df, result):
    again, _, _ = sg.segment(features_df, k=5)
    assert again["cluster"].tolist() == result[0]["cluster"].tolist()


def test_choose_k_respects_minimum_and_stability():
    scores = pd.DataFrame(
        {"k": [2, 3, 4, 5], "silhouette": [0.90, 0.30, 0.50, 0.40], "stability_ari_mean": [1.0, 0.95, 0.50, 0.99]}
    )
    # k=2 is too coarse, k=4 is unstable, so k=5 wins over k=3
    assert sg.choose_k(scores) == 5
    scores["stability_ari_mean"] = 0.1  # nothing stable: fall back to best silhouette with k >= 3
    assert sg.choose_k(scores) == 4


def test_name_cluster_rules():
    row = pd.Series({"avg_tenure_months": 8, "avg_monthly_charges": 90, "pct_month_to_month": 95, "pct_autopay": 0})
    assert sg.name_cluster(row) == "New high-spend, month-to-month, manual pay"
    row = pd.Series({"avg_tenure_months": 60, "avg_monthly_charges": 25, "pct_month_to_month": 0, "pct_autopay": 90})
    assert sg.name_cluster(row) == "Long-standing low-spend, contracted, auto-pay"


def test_exported_cluster_file_matches_a_fresh_run(result):
    saved = pd.read_csv(config.DASHBOARD_DATA_DIR / "customer_clusters.csv")
    fresh = result[0]
    assert len(saved) == len(fresh)
    assert saved["cluster"].tolist() == fresh["cluster"].tolist()
