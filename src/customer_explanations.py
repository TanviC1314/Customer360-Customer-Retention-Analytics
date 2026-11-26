"""Per-customer churn-risk explanations (Phase 2, enhancement 4 & 5).

Produces a human-readable explanation for each customer's churn-risk score:
top factors pushing the prediction up and down, and the overall risk band.

Risk bands match ``src/export_dashboard_data.py``:
  Low    < 0.30
  Medium 0.30–0.50
  High   >= 0.50

Usage:
    python -m src.customer_explanations               # generates reports + dashboard CSV
    python -m src.customer_explanations --id CUS-0001 # explain one customer
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, feature_engineering as fe, model_training as mt
from .explainability import (
    _get_preprocessor_and_model,
    aggregate_to_original,
    compute_shap_values,
    _encoded_feature_names,
    _ohe_to_original_mapping,
    customer_shap_table,
)
from .export_dashboard_data import RISK_BANDS

# --------------------------------------------------------------------------- #
# Core explanation builder
# --------------------------------------------------------------------------- #

def _risk_band(prob: float) -> str:
    for band, lo, hi in RISK_BANDS:
        if lo <= prob < hi:
            return band
    return RISK_BANDS[-1][0]  # catch exactly 1.0


def explain_customer(
    customer_id: str,
    shap_row: pd.Series,
    churn_prob: float,
    top_n: int = 5,
) -> dict:
    """Return a structured explanation dict for a single customer.

    Parameters
    ----------
    shap_row : pd.Series
        Original-feature SHAP values for this customer (index = feature name).
    churn_prob : float
        Model churn probability for this customer.
    top_n : int
        Number of top features to surface for each direction.
    """
    shap_row = shap_row.astype(float)
    drivers_up = shap_row.nlargest(top_n)       # pushes churn probability up
    drivers_down = shap_row.nsmallest(top_n)    # pushes churn probability down (retention signals)
    return {
        config.ID_COL: customer_id,
        "churn_probability": round(float(churn_prob), 4),
        "risk_band": _risk_band(churn_prob),
        "top_risk_factors": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in drivers_up.items()
            if v > 0
        ],
        "top_retention_signals": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in drivers_down.items()
            if v < 0
        ],
    }


def build_all_explanations(
    pipe,
    df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Return one explanation dict per customer in ``df``."""
    X, _ = fe.get_feature_matrix(df)
    probs = pipe.predict_proba(X)[:, 1]

    shap_tbl = customer_shap_table(pipe, df)
    id_col = shap_tbl[config.ID_COL].to_numpy()
    feature_cols = [c for c in shap_tbl.columns if c != config.ID_COL]

    explanations = []
    for i, (cid, prob) in enumerate(zip(id_col, probs)):
        shap_row = shap_tbl.iloc[i][feature_cols]
        explanations.append(explain_customer(str(cid), shap_row, float(prob), top_n=top_n))
    return explanations


def explanations_to_dataframe(explanations: list[dict]) -> pd.DataFrame:
    """Flatten the list of explanation dicts into a wide DataFrame (one row per customer).

    Columns: customer_id, churn_probability, risk_band,
             risk_factor_1 … risk_factor_N, retention_signal_1 … retention_signal_N.
    """
    rows = []
    for ex in explanations:
        row: dict = {
            config.ID_COL: ex[config.ID_COL],
            "churn_probability": ex["churn_probability"],
            "risk_band": ex["risk_band"],
        }
        for i, f in enumerate(ex["top_risk_factors"], 1):
            row[f"risk_factor_{i}"] = f["feature"]
            row[f"risk_factor_{i}_shap"] = f["shap_value"]
        for i, f in enumerate(ex["top_retention_signals"], 1):
            row[f"retention_signal_{i}"] = f["feature"]
            row[f"retention_signal_{i}_shap"] = f["shap_value"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Single-customer text report
# --------------------------------------------------------------------------- #

def format_explanation(ex: dict) -> str:
    cid = ex[config.ID_COL]
    prob = ex["churn_probability"]
    band = ex["risk_band"]
    lines = [
        f"Customer: {cid}",
        f"Churn probability: {prob:.1%}  |  Risk band: {band}",
        "",
        "Top churn-risk drivers (pushing probability UP):",
    ]
    for item in ex["top_risk_factors"]:
        lines.append(f"  + {item['feature']:30s}  SHAP {item['shap_value']:+.4f}")
    if not ex["top_risk_factors"]:
        lines.append("  (none above zero)")
    lines.append("")
    lines.append("Top retention signals (pushing probability DOWN):")
    for item in ex["top_retention_signals"]:
        lines.append(f"  - {item['feature']:30s}  SHAP {item['shap_value']:+.4f}")
    if not ex["top_retention_signals"]:
        lines.append("  (none below zero)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _load_fitted_pipe(X_train, y_train):
    models = mt.get_models(y_train)
    cv_results = mt.cross_validate_models(models, X_train, y_train)
    champion_name = mt.select_champion(cv_results)
    return models[champion_name].fit(X_train, y_train), champion_name


def main(customer_id: str | None = None) -> None:
    df = mt.load_modelling_data()
    X, y = fe.get_feature_matrix(df)
    X_train, _, y_train, _ = fe.split_data(X, y)

    pipe, champion = _load_fitted_pipe(X_train, y_train)
    print(f"Champion: {champion}")

    if customer_id:
        # Single-customer mode
        row = df[df[config.ID_COL] == customer_id]
        if row.empty:
            raise ValueError(f"Customer '{customer_id}' not found in the dataset.")
        sub_X, _ = fe.get_feature_matrix(row)
        prob = float(pipe.predict_proba(sub_X)[:, 1][0])
        tbl = customer_shap_table(pipe, row)
        feature_cols = [c for c in tbl.columns if c != config.ID_COL]
        shap_row = tbl.iloc[0][feature_cols]
        ex = explain_customer(customer_id, shap_row, prob)
        print(format_explanation(ex))
    else:
        print("Building explanations for all customers…")
        explanations = build_all_explanations(pipe, df)
        flat = explanations_to_dataframe(explanations)
        config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = config.DASHBOARD_DATA_DIR / "customer_explanations.csv"
        flat.to_csv(out, index=False)
        print(f"Saved {len(flat)} rows → {out}")

        # Quick summary
        band_counts = flat["risk_band"].value_counts()
        print("\nRisk band distribution:")
        print(band_counts.to_string())
        avg_prob = flat["churn_probability"].mean()
        print(f"\nMean churn probability: {avg_prob:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="customer_id", default=None, help="Explain a single customer")
    args = parser.parse_args()
    main(customer_id=args.customer_id)
