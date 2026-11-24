"""Tests for the Power BI data export: reconciliation with the raw source and model-field integrity."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, export_dashboard_data as ex

D = config.DASHBOARD_DATA_DIR


@pytest.fixture(scope="module")
def fact() -> pd.DataFrame:
    return pd.read_csv(D / "customers.csv")


@pytest.fixture(scope="module")
def kpi() -> pd.Series:
    return pd.read_csv(D / "kpi_summary.csv").iloc[0]


def test_all_expected_files_exist_and_are_non_empty():
    expected = ["customers", "segment_summary", "feature_importance", "retention_opportunities", "kpi_summary",
                "model_comparison", "dim_tenure_group", "dim_charge_band", "dim_contract", "dim_risk_band", "dim_segment"]
    for name in expected:
        path = D / f"{name}.csv"
        assert path.exists(), name
        assert len(pd.read_csv(path)) > 0, name


def test_fact_table_reconciles_with_raw(raw_df, fact):
    assert len(fact) == len(raw_df) == 7043
    assert fact["customer_id"].is_unique
    assert set(fact["customer_id"]) == set(raw_df["customerID"])
    assert fact["churn_flag"].sum() == (raw_df["Churn"] == "Yes").sum()


def test_kpi_summary_matches_raw_source(raw_df, kpi):
    """These are the numbers the Power BI DAX measures must reproduce (see powerbi_guide.md)."""
    churned = raw_df["Churn"] == "Yes"
    assert kpi["total_customers"] == len(raw_df)
    assert kpi["churned_customers"] == churned.sum()
    assert kpi["active_customers"] == (~churned).sum()
    assert kpi["churn_rate_pct"] == pytest.approx(churned.mean() * 100, abs=0.01)
    assert kpi["avg_monthly_charges"] == pytest.approx(raw_df["MonthlyCharges"].mean(), abs=0.01)
    assert kpi["avg_tenure_months"] == pytest.approx(raw_df["tenure"].mean(), abs=0.01)
    assert kpi["active_mrr"] == pytest.approx(raw_df.loc[~churned, "MonthlyCharges"].sum(), abs=0.01)
    assert kpi["lost_mrr"] == pytest.approx(raw_df.loc[churned, "MonthlyCharges"].sum(), abs=0.01)


def test_documented_reconciliation_values_in_guide_match_data(kpi):
    """The numbers printed in the Power BI guide must equal the exported KPI file."""
    guide = (config.PROJECT_ROOT / "dashboard" / "powerbi_guide.md").read_text(encoding="utf-8")
    for text in [f"{int(kpi['total_customers']):,}", f"{int(kpi['churned_customers']):,}", f"{kpi['churn_rate_pct']:.2f}%",
                 f"${kpi['avg_monthly_charges']:.2f}", f"{kpi['avg_tenure_months']:.2f}",
                 f"${kpi['active_mrr']:,.2f}", f"${kpi['lost_mrr']:,.2f}",
                 f"{int(kpi['high_risk_active_customers'])}", f"${kpi['model_scored_revenue_at_risk_mrr']:,.2f}"]:
        assert text in guide, f"{text} not found in powerbi_guide.md"


def test_risk_scores_valid_and_bands_consistent(fact):
    assert fact["churn_risk_score"].between(0, 1).all()
    assert set(fact["risk_band"]) == {"Low", "Medium", "High"}
    assert (fact.loc[fact.risk_band == "High", "churn_risk_score"] >= 0.5).all()
    assert (fact.loc[fact.risk_band == "Low", "churn_risk_score"] < 0.3).all()
    mid = fact.loc[fact.risk_band == "Medium", "churn_risk_score"]
    assert mid.between(0.3, 0.5, inclusive="left").all()


def test_risk_score_has_real_signal_out_of_fold(fact):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(fact["churn_flag"], fact["churn_risk_score"])
    assert 0.80 < auc < 0.90  # consistent with the evaluated model (and not suspiciously perfect)


def test_revenue_at_risk_only_counts_active_high_risk(fact, kpi):
    at_risk = fact[fact["monthly_revenue_at_risk"] > 0]
    assert (at_risk["churn_flag"] == 0).all()
    assert (at_risk["risk_band"] == "High").all()
    assert (at_risk["monthly_revenue_at_risk"] == at_risk["monthly_charges"]).all()
    assert fact["monthly_revenue_at_risk"].sum() == pytest.approx(kpi["model_scored_revenue_at_risk_mrr"], abs=0.01)
    assert len(at_risk) == kpi["high_risk_active_customers"]


def test_dimension_tables_cover_all_fact_values(fact):
    for dim, col in [("dim_tenure_group", "tenure_group"), ("dim_charge_band", "monthly_charge_band"),
                     ("dim_contract", "contract"), ("dim_risk_band", "risk_band"), ("dim_segment", "customer_segment")]:
        dim_values = set(pd.read_csv(D / f"{dim}.csv")[col])
        assert set(fact[col].unique()) <= dim_values, f"{dim} missing values"
        assert pd.read_csv(D / f"{dim}.csv")[col].is_unique


def test_segment_assignment_rules():
    tenure = pd.Series([60, 20, 20, 6, 6])
    monthly = pd.Series([50.0, 95.0, 40.0, 80.0, 30.0])
    assert ex.assign_segment(tenure, monthly).tolist() == [
        "Loyal (49+ months)", "Established high-spend", "Established standard",
        "New high-spend (<=12 months)", "New standard (<=12 months)"]
    # boundaries: 13 months / $70 belong to the "established high-spend" group, 12 months does not
    assert ex.assign_segment(pd.Series([13, 12, 49]), pd.Series([70.0, 70.0, 70.0])).tolist() == [
        "Established high-spend", "New high-spend (<=12 months)", "Loyal (49+ months)"]


def test_risk_band_boundaries():
    s = pd.Series([0.0, 0.2999, 0.30, 0.4999, 0.50, 1.0])
    assert ex.assign_risk_band(s).tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_segment_summary_is_consistent_with_fact(fact):
    seg = pd.read_csv(D / "segment_summary.csv")
    assert (seg["customers"] >= 50).all()
    assert seg["churn_rank"].tolist() == list(range(1, len(seg) + 1))
    assert seg["churn_rate_pct"].is_monotonic_decreasing
    top = seg.iloc[0]
    sub = fact[(fact.contract == top.contract) & (fact.tenure_group == top.tenure_group)
               & (fact.internet_service == top.internet_service)]
    assert len(sub) == top["customers"]
    assert sub["churn_flag"].sum() == top["churned"]


def test_retention_opportunities_reconcile_with_raw(raw_df):
    opp = pd.read_csv(D / "retention_opportunities.csv").set_index("opportunity")
    m2m = raw_df[raw_df["Contract"] == "Month-to-month"]
    row = opp.loc["Move month-to-month customers to longer contracts"]
    assert row["target_churn_rate_pct"] == pytest.approx((m2m["Churn"] == "Yes").mean() * 100, abs=0.01)
    assert row["active_customers"] == (m2m["Churn"] == "No").sum()
    assert row["churn_gap_pct_points"] == pytest.approx(row["target_churn_rate_pct"] - row["comparison_churn_rate_pct"], abs=0.02)
    assert (opp["active_customers"] > 0).all()
    assert opp["suggested_action"].str.len().min() > 20
