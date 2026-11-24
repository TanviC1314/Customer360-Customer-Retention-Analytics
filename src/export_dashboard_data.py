"""Build the Power BI-ready data files in ``dashboard/dashboard_data/``.

Star-schema style output: one fact table (``customers.csv``) plus small dimension and
summary tables. Churn-risk scores are **out-of-fold** predictions from the champion model
(each customer is scored by a model that never saw that customer), so the dashboard does
not show optimistic in-sample scores.

    python -m src.export_dashboard_data
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, feature_engineering as fe, model_training as mt

OUT = config.DASHBOARD_DATA_DIR

RISK_BANDS = [("Low", 0.0, 0.30), ("Medium", 0.30, 0.50), ("High", 0.50, 1.0001)]
SEGMENT_ORDER = [
    "Loyal (49+ months)",
    "Established high-spend",
    "Established standard",
    "New high-spend (<=12 months)",
    "New standard (<=12 months)",
]


def assign_segment(tenure: pd.Series, monthly: pd.Series) -> pd.Series:
    """Lifecycle x spend segmentation (identical rules to SQL query Q08)."""
    seg = np.select(
        [tenure >= 49, (tenure >= 13) & (monthly >= 70), tenure >= 13, monthly >= 70],
        [SEGMENT_ORDER[0], SEGMENT_ORDER[1], SEGMENT_ORDER[2], SEGMENT_ORDER[3]],
        default=SEGMENT_ORDER[4],
    )
    return pd.Series(seg, index=tenure.index)


def assign_risk_band(score: pd.Series) -> pd.Series:
    labels = pd.cut(score, bins=[b[1] for b in RISK_BANDS] + [RISK_BANDS[-1][2]],
                    labels=[b[0] for b in RISK_BANDS], right=False)
    return labels.astype("string")


def out_of_fold_scores(df: pd.DataFrame, model_name: str) -> pd.Series:
    """5-fold out-of-fold churn probability for every customer using the named model."""
    X, y = fe.get_feature_matrix(df)
    pipe = clone(mt.get_models(y)[model_name])
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.Series(proba, index=df.index, name="churn_risk_score")


def build_fact_table(df: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["churn_status"] = np.where(out[config.TARGET] == 1, "Churned", "Retained")
    out["customer_segment"] = assign_segment(out["tenure_months"], out["monthly_charges"])
    out["churn_risk_score"] = scores.round(4)
    out["risk_band"] = assign_risk_band(out["churn_risk_score"])
    # Revenue at risk = monthly charges of customers who are STILL ACTIVE and scored High risk
    out["monthly_revenue_at_risk"] = np.where(
        (out[config.TARGET] == 0) & (out["risk_band"] == "High"), out["monthly_charges"], 0.0
    )
    out["tenure_group_order"] = out["tenure_group"].map({k: i + 1 for i, k in enumerate(config.TENURE_LABELS)})
    out["charge_band_order"] = out["monthly_charge_band"].map({k: i + 1 for i, k in enumerate(config.CHARGE_LABELS)})
    out["contract_order"] = out["contract"].map({k: i + 1 for i, k in enumerate(config.CONTRACT_ORDER)})
    return out


def segment_summary(fact: pd.DataFrame, min_customers: int = 50) -> pd.DataFrame:
    """Contract x tenure group x internet segments with churn, revenue and active exposure."""
    keys = ["contract", "tenure_group", "internet_service"]
    g = fact.groupby(keys, observed=True)
    seg = g.agg(customers=("customer_id", "count"), churned=(config.TARGET, "sum"),
                avg_monthly_charges=("monthly_charges", "mean")).reset_index()
    active = fact[fact[config.TARGET] == 0].groupby(keys, observed=True).agg(
        active_customers=("customer_id", "count"), active_mrr=("monthly_charges", "sum")).reset_index()
    seg = seg.merge(active, on=keys, how="left").fillna({"active_customers": 0, "active_mrr": 0})
    seg["churn_rate_pct"] = (seg["churned"] / seg["customers"] * 100).round(2)
    seg = seg[seg["customers"] >= min_customers].sort_values("churn_rate_pct", ascending=False)
    seg["avg_monthly_charges"] = seg["avg_monthly_charges"].round(2)
    seg["active_mrr"] = seg["active_mrr"].round(2)
    seg.insert(0, "churn_rank", range(1, len(seg) + 1))
    seg["segment_label"] = seg["contract"] + " | " + seg["tenure_group"] + " | " + seg["internet_service"]
    return seg.reset_index(drop=True)


def retention_opportunities(fact: pd.DataFrame) -> pd.DataFrame:
    """Sizing of the main retention levers. Gaps are OBSERVED associations, not causal effects;
    ``illustrative_mrr_protected`` = target MRR x churn-rate gap, an upper-bound illustration."""
    rate = lambda s: s[config.TARGET].mean() * 100
    active = fact[fact[config.TARGET] == 0]
    internet = fact[fact["internet_service"] != "No internet"]
    rows = []

    def add(name, action, tgt_mask_all, cmp_mask_all, cmp_label, tgt_label):
        tgt_all, cmp_all = fact[tgt_mask_all], fact[cmp_mask_all]
        tgt_active = tgt_all[tgt_all[config.TARGET] == 0]
        gap = rate(tgt_all) - rate(cmp_all)
        rows.append({
            "opportunity": name, "target_group": tgt_label, "active_customers": len(tgt_active),
            "active_mrr": round(tgt_active["monthly_charges"].sum(), 2),
            "target_churn_rate_pct": round(rate(tgt_all), 2),
            "comparison_group": cmp_label, "comparison_churn_rate_pct": round(rate(cmp_all), 2),
            "churn_gap_pct_points": round(gap, 2),
            "illustrative_mrr_protected": round(tgt_active["monthly_charges"].sum() * gap / 100, 2),
            "suggested_action": action,
        })

    add("Move month-to-month customers to longer contracts",
        "Offer a discount or perk for 12/24-month commitment, timed at months 3-9 of tenure.",
        fact["contract"] == "Month-to-month", fact["contract"] != "Month-to-month",
        "One/Two-year contracts", "Month-to-month contract")
    add("Move electronic-check payers to automatic payment",
        "Incentivise bank-transfer / card auto-pay (one-off bill credit); simplifies payment and signals commitment.",
        fact["payment_method"] == "Electronic check", fact["payment_method"].str.contains("automatic"),
        "Automatic payment methods", "Pays by electronic check")
    tenure_new = fact["tenure_months"] <= 12
    add("Strengthen the first-year onboarding experience",
        "Welcome journey, 30/90-day check-ins, proactive service-quality follow-up in the first 12 months.",
        tenure_new, ~tenure_new, "Customers with 13+ months tenure", "Tenure 0-12 months")
    no_support = (fact["internet_service"] != "No internet") & (fact["num_support_services"] == 0)
    with_support = (fact["internet_service"] != "No internet") & (fact["num_support_services"] > 0)
    add("Add a protection/support service for internet customers without one",
        "Bundle online security / tech support free for 3 months, then at a reduced price.",
        no_support, with_support, "Internet customers with 1+ support service", "Internet, no support service")
    fiber = fact["internet_service"] == "Fiber optic"
    add("Review fibre price-for-value and service quality",
        "Investigate fibre complaints/outages, review price vs competitors, target fibre month-to-month customers first.",
        fiber, fact["internet_service"] == "DSL", "DSL customers", "Fibre optic internet")
    return pd.DataFrame(rows).sort_values("illustrative_mrr_protected", ascending=False).reset_index(drop=True)


def kpi_summary(fact: pd.DataFrame) -> pd.DataFrame:
    act = fact[fact[config.TARGET] == 0]
    ch = fact[fact[config.TARGET] == 1]
    return pd.DataFrame([{
        "total_customers": len(fact),
        "churned_customers": int(len(ch)),
        "active_customers": int(len(act)),
        "churn_rate_pct": round(len(ch) / len(fact) * 100, 2),
        "avg_monthly_charges": round(fact["monthly_charges"].mean(), 2),
        "avg_tenure_months": round(fact["tenure_months"].mean(), 2),
        "active_mrr": round(act["monthly_charges"].sum(), 2),
        "lost_mrr": round(ch["monthly_charges"].sum(), 2),
        "high_risk_active_customers": int((fact["monthly_revenue_at_risk"] > 0).sum()),
        "model_scored_revenue_at_risk_mrr": round(fact["monthly_revenue_at_risk"].sum(), 2),
    }])


def run() -> dict[str, pd.DataFrame]:
    metrics = json.loads((config.REPORTS_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    champion = metrics["champion"]
    df = mt.load_modelling_data()
    scores = out_of_fold_scores(df, champion)
    fact = build_fact_table(df, scores)

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "customers": fact,
        "segment_summary": segment_summary(fact),
        "retention_opportunities": retention_opportunities(fact),
        "kpi_summary": kpi_summary(fact),
        "dim_tenure_group": pd.DataFrame({"tenure_group": config.TENURE_LABELS, "sort_order": range(1, 5)}),
        "dim_charge_band": pd.DataFrame({"monthly_charge_band": config.CHARGE_LABELS, "sort_order": range(1, 5)}),
        "dim_contract": pd.DataFrame({"contract": config.CONTRACT_ORDER, "sort_order": range(1, 4)}),
        "dim_risk_band": pd.DataFrame({
            "risk_band": [b[0] for b in RISK_BANDS], "sort_order": [1, 2, 3],
            "score_range": ["< 0.30", "0.30 - 0.50", ">= 0.50"]}),
        "dim_segment": pd.DataFrame({"customer_segment": SEGMENT_ORDER, "sort_order": range(1, 6)}),
    }
    perm = pd.read_csv(config.REPORTS_DIR / "tables" / "permutation_importance.csv")
    perm["importance_pct_of_total"] = (perm["importance_mean"].clip(lower=0) / perm["importance_mean"].clip(lower=0).sum() * 100).round(1)
    perm["model"] = champion
    tables["feature_importance"] = perm
    tm = pd.DataFrame(metrics["test_metrics"])
    tables["model_comparison"] = tm
    for name, tbl in tables.items():
        tbl.to_csv(OUT / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = run()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(t["kpi_summary"].T)
    print(t["customers"]["risk_band"].value_counts())
    print(t["retention_opportunities"].drop(columns=["suggested_action"]).to_string(index=False))
    print(f"\nWrote {len(t)} files to {OUT}")
