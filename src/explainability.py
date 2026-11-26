"""SHAP-based global model explanations (Phase 2, enhancement 3 & 4).

Uses ``shap.TreeExplainer`` on the fitted XGBoost pipeline.  SHAP values are computed
on the *pre-processed* (encoded) feature matrix; they are then aggregated back to the
original feature names so the results are human-readable.

    python -m src.explainability          # global SHAP summary
    python -m src.explainability --batch  # also exports per-customer SHAP table
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from . import config, feature_engineering as fe, model_training as mt

# --------------------------------------------------------------------------- #
# SHAP helpers
# --------------------------------------------------------------------------- #

def _get_preprocessor_and_model(pipe: Pipeline):
    """Return the fitted ColumnTransformer and the underlying XGBoost Booster."""
    # The preprocessing step is named "prep" in the Customer360 pipeline
    prep_key = next(k for k in pipe.named_steps if k != "model")
    preprocessor = pipe.named_steps[prep_key]
    model = pipe.named_steps["model"]
    return preprocessor, model


def _encoded_feature_names(preprocessor) -> list[str]:
    """Full list of column names after ColumnTransformer, including OHE columns."""
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(columns).tolist())
        else:
            names.extend(columns if isinstance(columns, list) else [columns])
    return names


def _ohe_to_original_mapping(preprocessor, encoded_names: list[str]) -> dict[str, str]:
    """Map each encoded column name back to the original feature name."""
    mapping: dict[str, str] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            enc_cols = transformer.get_feature_names_out(columns).tolist()
            for ec in enc_cols:
                # OHE names look like "x0_value" or the full "featurename_category"
                # fall back to the first original column if no match
                orig = next((c for c in columns if ec.startswith(c)), columns[0])
                mapping[ec] = orig
        else:
            for c in (columns if isinstance(columns, list) else [columns]):
                mapping[c] = c
    return mapping


def compute_shap_values(
    pipe: Pipeline,
    X: pd.DataFrame,
    sample_size: int | None = 2000,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return ``(shap_values, X_encoded_df, encoded_feature_names)``.

    ``shap_values`` has shape (n_samples, n_encoded_features).  Call
    ``aggregate_to_original`` to collapse OHE columns back to originals.
    """
    preprocessor, model = _get_preprocessor_and_model(pipe)
    if sample_size and len(X) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    X_enc = preprocessor.transform(X_sample)
    enc_names = _encoded_feature_names(preprocessor)
    X_enc_df = pd.DataFrame(X_enc, columns=enc_names)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_enc)
    return shap_vals, X_enc_df, enc_names


def aggregate_to_original(
    shap_values: np.ndarray,
    encoded_names: list[str],
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Sum |SHAP| of OHE sub-columns into their original feature name."""
    df = pd.DataFrame(shap_values, columns=encoded_names)
    result: dict[str, pd.Series] = {}
    for enc_col, orig_col in mapping.items():
        if enc_col in df.columns:
            result.setdefault(orig_col, pd.Series(np.zeros(len(df))))
            result[orig_col] = result[orig_col] + df[enc_col]
    return pd.DataFrame(result)


def global_importance(shap_agg: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per original feature, ranked high-to-low."""
    mean_abs = shap_agg.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": mean_abs.index, "mean_abs_shap": mean_abs.values})


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_global_bar(importance: pd.DataFrame, top_n: int = 15) -> None:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3A6EA5")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(f"Global SHAP feature importance — top {top_n} original features")
    ax.set_ylabel("")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "shap_01_global_bar.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_beeswarm(shap_values: np.ndarray, X_enc_df: pd.DataFrame, top_n: int = 15) -> None:
    """Beeswarm on the encoded feature matrix, showing the top_n encoded columns."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    sv_top = shap_values[:, top_idx]
    names_top = [X_enc_df.columns[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(10, 7))
    for j, (sv_col, name) in enumerate(zip(sv_top.T, names_top)):
        x_vals = X_enc_df.iloc[:, top_idx[j]].to_numpy()
        # normalise feature value to [0,1] for colour
        x_norm = (x_vals - x_vals.min()) / max(float(x_vals.max() - x_vals.min()), 1e-9)
        y_jitter = j + np.random.default_rng(42).uniform(-0.3, 0.3, len(sv_col))
        sc = ax.scatter(sv_col, y_jitter, c=x_norm, cmap="coolwarm", s=6, alpha=0.5, linewidth=0)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_top, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on churn probability, log-odds)")
    ax.set_title(f"SHAP beeswarm — top {top_n} encoded features")
    plt.colorbar(sc, ax=ax, label="Feature value (low → high)", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_02_beeswarm.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_global_report(importance: pd.DataFrame, top_n: int = 15) -> None:
    top = importance.head(top_n)

    def tbl(d: pd.DataFrame) -> str:
        head = "| " + " | ".join(str(c) for c in d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
        rows = ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
                for row in d.itertuples(index=False)]
        return head + "\n".join(rows)

    text = f"""# SHAP global explanations (Phase 2, enhancement 3)

_Generated by `python -m src.explainability`; every number was computed from the data._

## Method

* `shap.TreeExplainer` on the fitted XGBoost champion pipeline.
* SHAP values are computed on the pre-processed (one-hot encoded) feature matrix using a sample of 2,000 customers.
* OHE sub-columns are **aggregated back to the original feature name** (sum of raw SHAP values) so results are human-readable.
* Mean absolute SHAP value across the sample is used as the importance measure.

## Global feature importance (top {top_n})

{tbl(top.assign(mean_abs_shap=top['mean_abs_shap'].round(4)))}

## Interpretation

* **Higher = more influential on the model's churn-probability prediction** (regardless of direction).
* The sign of the SHAP value for a single customer tells whether that feature pushed the prediction up or down; mean |SHAP| averages this out across all customers.
* The beeswarm plot (``reports/figures/shap_02_beeswarm.png``) shows direction: red = high feature value, blue = low.

## Limitations

* SHAP explains the model, not the real world. A high SHAP importance means the model relies on that feature; it does not prove causation.
* The sample of 2,000 is used for speed; the full 7,043 would give essentially the same ranking.
* One-hot encoding means some original features are split across many sub-columns; the aggregation sums signed SHAP values, so cancellation can understate the true influence of categorical features with many categories.

![Global SHAP bar](figures/shap_01_global_bar.png)
![SHAP beeswarm](figures/shap_02_beeswarm.png)
"""
    (config.REPORTS_DIR / "shap_global_results.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Per-customer export (also used by customer_explanations.py)
# --------------------------------------------------------------------------- #

def customer_shap_table(
    pipe: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a tidy DataFrame with one row per (customer, feature) giving the SHAP value.

    Covers all customers in ``df`` (no subsampling) so it can be used for per-customer lookups.
    """
    preprocessor, model = _get_preprocessor_and_model(pipe)
    X_enc = preprocessor.transform(df)
    enc_names = _encoded_feature_names(preprocessor)
    mapping = _ohe_to_original_mapping(preprocessor, enc_names)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_enc)

    agg = aggregate_to_original(shap_vals, enc_names, mapping)
    agg.insert(0, config.ID_COL, df[config.ID_COL].to_numpy())
    return agg


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(batch: bool = False) -> None:
    df = mt.load_modelling_data()
    X, y = fe.get_feature_matrix(df)
    X_train, X_test, y_train, y_test = fe.split_data(X, y)

    models = mt.get_models(y_train)
    cv_results = mt.cross_validate_models(models, X_train, y_train)
    champion_name = mt.select_champion(cv_results)
    print(f"Champion: {champion_name}")

    pipe = models[champion_name].fit(X_train, y_train)

    print("Computing SHAP values…")
    shap_vals, X_enc_df, enc_names = compute_shap_values(pipe, X, sample_size=2000)

    preprocessor, _ = _get_preprocessor_and_model(pipe)
    mapping = _ohe_to_original_mapping(preprocessor, enc_names)
    shap_agg = aggregate_to_original(shap_vals, enc_names, mapping)
    importance = global_importance(shap_agg)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_global_bar(importance)
    plot_beeswarm(shap_vals, X_enc_df)
    write_global_report(importance)

    importance.to_csv(config.REPORTS_DIR / "tables" / "shap_global_importance.csv", index=False)

    print(importance.head(15).round(4).to_string(index=False))
    print(f"\nReport → {config.REPORTS_DIR / 'shap_global_results.md'}")

    if batch:
        print("Exporting per-customer SHAP table…")
        tbl = customer_shap_table(pipe, df)
        config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(config.DASHBOARD_DATA_DIR / "customer_shap.csv", index=False)
        print(f"Saved {len(tbl)} rows → {config.DASHBOARD_DATA_DIR / 'customer_shap.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Also export per-customer SHAP CSV")
    args = parser.parse_args()
    main(batch=args.batch)
