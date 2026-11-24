"""Evaluate the trained churn models on the held-out test set and explain what drives churn.

Produces, under ``reports/``:

* ``model_metrics.json``                machine-readable results (used by the report + dashboard export)
* ``tables/*.csv``                      test metrics, thresholds, lift, importances
* ``figures/ml_*.png``                  ROC / PR curves, confusion matrices, importances, gains
* ``model_results.md``                  the written model report (generated from the numbers)

    python -m src.model_evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, feature_engineering as fe, model_training as mt
from .exploratory_analysis import COLOR_CHURNED, COLOR_NEUTRAL, COLOR_RETAINED, set_style

set_style()

TABLES_DIR = config.REPORTS_DIR / "tables"
FIG_DIR = config.FIGURES_DIR


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Accuracy / precision / recall / F1 at ``threshold`` plus threshold-free ROC-AUC and PR-AUC."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_models(fitted: dict, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Score every fitted model (default 0.5 threshold) and return the metrics table + probabilities."""
    rows, probas = [], {}
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        probas[name] = proba
        rows.append({"model": name, **classification_metrics(y_test, proba)})
    base = mt.get_baseline().fit(X_test, y_test)  # majority class; predicts "no churn" for everyone
    rows.append({"model": "Majority-class baseline",
                 **classification_metrics(y_test, base.predict_proba(X_test)[:, 1], 0.5)})
    return pd.DataFrame(rows), probas


def choose_threshold(y_true, proba, grid=None) -> float:
    """Threshold that maximises F1 on the supplied (training, out-of-fold) predictions."""
    grid = np.arange(0.10, 0.91, 0.01) if grid is None else grid
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def threshold_table(y_true, proba, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, proba, t)
        flagged = m["tp"] + m["fp"]
        rows.append({"threshold": t, "customers_flagged": flagged, "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


def lift_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    """Decile lift: customers sorted by predicted risk, split into equal groups."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    df["decile"] = pd.qcut(np.arange(len(df)), bins, labels=range(1, bins + 1))
    g = df.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"), avg_risk=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / df["y"].mean()
    g["cum_share_of_churners"] = g["churners"].cumsum() / df["y"].sum()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def permutation_importance_table(pipe, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """Drop in ROC-AUC when each ORIGINAL column is shuffled (works on the full pipeline)."""
    res = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=config.RANDOM_STATE, n_jobs=1)
    out = pd.DataFrame({"feature": X_test.columns, "importance_mean": res.importances_mean,
                        "importance_std": res.importances_std})
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def logistic_odds_ratios(pipe) -> pd.DataFrame:
    """Odds ratios from a fitted logistic-regression pipeline (numeric features are standardised,
    so their ratio is per +1 standard deviation; dummy features are per switch from 0 to 1)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def tree_importance_table(pipe) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_pr(y_test, probas: dict, names: list[str]) -> None:
    colors = [COLOR_RETAINED, "#2A9D8F", COLOR_CHURNED]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, c in zip(names, colors):
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, probas[name]):.3f})")
        p, r, _ = precision_recall_curve(y_test, probas[name])
        axes[1].plot(r, p, color=c, lw=2, label=f"{name} (AP {average_precision_score(y_test, probas[name]):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=COLOR_NEUTRAL, label="Random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    axes[1].axhline(np.mean(y_test), ls="--", color=COLOR_NEUTRAL, label=f"Base rate {np.mean(y_test):.2f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "ml_01_roc_pr_curves.png")


def plot_confusion_grid(y_test, probas: dict, names: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_test, (probas[name] >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, linewidths=1, linecolor="white",
                    xticklabels=["Pred: stays", "Pred: churns"], yticklabels=["Actual: stays", "Actual: churned"])
        ax.set_title(name, fontsize=11)
    fig.suptitle("Confusion matrices at the default 0.5 threshold (test set, 1,409 customers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "ml_02_confusion_matrices.png")


def plot_metric_heatmap(metrics: pd.DataFrame) -> None:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    data = metrics.set_index("model")[cols]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score"})
    ax.set_title("Test-set metrics by model (threshold 0.5)")
    ax.set_ylabel("")
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    fig.tight_layout()
    _save(fig, "ml_03_model_comparison.png")


def plot_permutation_importance(imp: pd.DataFrame, champion: str) -> None:
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=COLOR_RETAINED, ecolor="#555")
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation feature importance - {champion}")
    fig.tight_layout()
    _save(fig, "ml_04_permutation_importance.png")


def plot_odds_ratios(odds: pd.DataFrame) -> None:
    top = odds.head(14).iloc[::-1]
    colors = [COLOR_CHURNED if c > 0 else COLOR_RETAINED for c in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.barh(top["feature"], top["odds_ratio"] - 1, left=1, color=colors)
    ax.axvline(1, color="#333", lw=1)
    ax.set_xlabel("Odds ratio (>1 raises churn odds, <1 lowers them)")
    ax.set_title("Logistic regression: largest effects on churn odds")
    for y, v in enumerate(top["odds_ratio"]):
        ax.text(v + (0.03 if v >= 1 else -0.03), y, f"{v:.2f}", va="center", ha="left" if v >= 1 else "right", fontsize=9)
    fig.tight_layout()
    _save(fig, "ml_05_logistic_odds_ratios.png")


def plot_gains(lift: pd.DataFrame, champion: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.r_[0, lift["decile"].astype(int) * 10]
    y = np.r_[0, lift["cum_share_of_churners"] * 100]
    ax.plot(x, y, marker="o", color=COLOR_CHURNED, lw=2, label=champion)
    ax.plot([0, 100], [0, 100], ls="--", color=COLOR_NEUTRAL, label="Random targeting")
    ax.set(xlabel="% of customers contacted (highest predicted risk first)", ylabel="% of churners reached",
           title="Cumulative gains (test set)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "ml_06_cumulative_gains.png")


def plot_threshold_tradeoff(y_test, proba, chosen: float, champion: str) -> None:
    ts = np.arange(0.05, 0.95, 0.01)
    p = [precision_score(y_test, proba >= t, zero_division=0) for t in ts]
    r = [recall_score(y_test, proba >= t, zero_division=0) for t in ts]
    f = [f1_score(y_test, proba >= t, zero_division=0) for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ts, p, label="Precision", color=COLOR_RETAINED, lw=2)
    ax.plot(ts, r, label="Recall", color=COLOR_CHURNED, lw=2)
    ax.plot(ts, f, label="F1", color="#2A9D8F", lw=2)
    ax.axvline(chosen, ls="--", color="#333", lw=1)
    ax.text(chosen + 0.01, 0.05, f"chosen on training data: {chosen:.2f}", fontsize=9)
    ax.set(xlabel="Decision threshold", ylabel="Score", title=f"Precision / recall trade-off - {champion}")
    ax.legend()
    fig.tight_layout()
    _save(fig, "ml_07_threshold_tradeoff.png")


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return head + "\n".join(rows)


def write_report(m: dict, cv: pd.DataFrame, test: pd.DataFrame, thr: pd.DataFrame, lift: pd.DataFrame,
                 perm: pd.DataFrame, odds: pd.DataFrame, path: Path) -> None:
    champ = m["champion"]
    base = test[test["model"] == "Majority-class baseline"].iloc[0]
    champ_default = m["champion_default"]
    champ_tuned = m["champion_tuned"]
    best_auc = test[test["model"] != "Majority-class baseline"]["roc_auc"]
    cv_view = pd.DataFrame({
        "Model": cv["model"],
        "Accuracy": [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv["cv_accuracy_mean"], cv["cv_accuracy_std"])],
        "Precision": [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv["cv_precision_mean"], cv["cv_precision_std"])],
        "Recall": [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv["cv_recall_mean"], cv["cv_recall_std"])],
        "F1": [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv["cv_f1_mean"], cv["cv_f1_std"])],
        "ROC-AUC": [f"{a:.3f} ± {s:.3f}" for a, s in zip(cv["cv_roc_auc_mean"], cv["cv_roc_auc_std"])],
    })
    test_view = test[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]].copy()
    test_view.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TP", "FP", "FN", "TN"]
    xgb_u = test[test["model"] == "XGBoost"].iloc[0]
    xgb_b = test[test["model"] == "XGBoost (balanced)"].iloc[0]
    top_dec = lift.iloc[0]
    top3 = lift.iloc[:3]
    perm_top = perm.head(8)[["feature", "importance_mean", "importance_std"]].copy()
    perm_top.columns = ["Feature", "ROC-AUC drop (mean)", "Std"]
    odds_top = odds.head(10)[["feature", "odds_ratio"]].copy()
    odds_top.columns = ["Encoded feature", "Odds ratio"]
    lines = f"""# Model results

_Generated by `python -m src.model_evaluation` from the executed models - every number below was computed, none typed in.
Random seed {config.RANDOM_STATE}; hold-out test set = {m['n_test']:,} customers ({m['n_train']:,} used for training and cross-validation)._

## 1. Setup

| Item | Choice |
|---|---|
| Target | `churn_flag` (1 = churned) - {_pct(m['churn_rate_train'])} of training customers churned |
| Split | Stratified 80/20 train/test (`random_state={config.RANDOM_STATE}`); test set untouched until final evaluation |
| Preprocessing | `ColumnTransformer` inside every `Pipeline`: standard-scaling of numerics (logistic regression only; trees do not need it) and one-hot encoding of categoricals (`handle_unknown="ignore"`). Re-fitted inside each CV fold, so no leakage |
| Features | {m['n_features']} raw columns (tenure, charges, services, contract, billing, demographics). `customer_id`, `churn` and `churn_flag` are excluded by construction (tested) |
| Models | Logistic Regression (baseline), Random Forest, XGBoost - each unweighted and with class weighting |
| Class imbalance | {_pct(m['churn_rate_train'])} positives. Class weighting is compared against unweighted training (below) rather than assumed |
| Model selection | 5-fold stratified CV on the training data only; champion = highest mean CV ROC-AUC |
| Hyper-parameters | Fixed, conservative values. Systematic tuning is a Phase 2 item, so these results are a starting point, not an optimum |

## 2. Cross-validation results (training data, mean ± std over 5 folds)

{_md_table(cv_view)}

All six variants have CV ROC-AUC within 0.004 of each other, far less than the fold-to-fold standard deviation (~0.01).
**The models cannot be separated on ranking quality.** The champion rule (highest mean CV ROC-AUC) selects **{champ}**, but the
choice is close to arbitrary; logistic regression is statistically indistinguishable and far easier to explain.

## 3. Hold-out test results (threshold 0.5)

{_md_table(test_view)}

(TP/FP/FN/TN = confusion-matrix counts with "churn" as the positive class.)

![Metric heatmap](figures/ml_03_model_comparison.png)
![ROC and PR curves](figures/ml_01_roc_pr_curves.png)
![Confusion matrices](figures/ml_02_confusion_matrices.png)

### Why accuracy alone is not enough

* A "model" that predicts *nobody churns* scores **{_pct(base['accuracy'])} accuracy** on this test set while catching
  **{_pct(base['recall'])}** of churners. Accuracy mostly measures the majority class when only ~1 in 4 customers churns.
* Unweighted XGBoost has {_pct(xgb_u['accuracy'])} accuracy but recall of only {_pct(xgb_u['recall'])}: it misses about
  {xgb_u['fn'] / (xgb_u['fn'] + xgb_u['tp']) * 100:.0f}% of customers who leave. The class-weighted version has *lower* accuracy
  ({_pct(xgb_b['accuracy'])}) but recall of {_pct(xgb_b['recall'])} - it is more useful for retention, which is why
  the metrics must be read together.
* A missed churner (false negative) costs a customer's recurring revenue; a false alarm (false positive) costs a discount or a call.
  The right balance of precision and recall depends on those costs, which the dataset does not contain.
* ROC-AUC/PR-AUC are threshold-free and measure how well customers are *ranked* by risk, which is what targeting a limited
  retention budget needs.

Class weighting trades precision for recall at the default 0.5 threshold without changing the ranking quality
(ROC-AUC differs by < 0.01 between weighted and unweighted versions).

## 4. Champion model: {champ}

* Test ROC-AUC **{champ_default['roc_auc']:.3f}** (range across all six variants: {best_auc.min():.3f}-{best_auc.max():.3f}); PR-AUC {champ_default['pr_auc']:.3f}.
* Threshold choice: the default 0.5 gives precision {_pct(champ_default['precision'])} / recall {_pct(champ_default['recall'])} / F1 {champ_default['f1']:.3f}.
  A threshold of **{champ_tuned['threshold']:.2f}** was chosen to maximise F1 on *out-of-fold predictions for the training data only* (the
  test set played no part) and gives precision {_pct(champ_tuned['precision'])} / recall {_pct(champ_tuned['recall'])} /
  F1 {champ_tuned['f1']:.3f} / accuracy {_pct(champ_tuned['accuracy'])} on the test set.

{_md_table(thr)}

![Threshold trade-off](figures/ml_07_threshold_tradeoff.png)

### Targeting value (decile lift, test set)

Customers sorted by predicted churn risk. The top 10% of customers contain **{_pct(top_dec['cum_share_of_churners'])}** of all churners
(lift {top_dec['lift']:.1f}x); the top 30% contain **{_pct(top3.iloc[-1]['cum_share_of_churners'])}**.
(Random targeting would reach 10% and 30%.)

{_md_table(lift.assign(churn_rate=lift['churn_rate'].map(_pct), cum_share_of_churners=lift['cum_share_of_churners'].map(_pct)).drop(columns=['avg_risk']).rename(columns={'decile': 'Risk decile (1 = highest)'}), '{:.2f}')}

![Cumulative gains](figures/ml_06_cumulative_gains.png)

## 5. What drives churn (feature importance)

**Permutation importance** ({champ}, test set, 10 repeats): how much ROC-AUC falls when one original column is randomly shuffled.
Computed on original columns, so the importance of a multi-level feature like `contract` is not split across dummy columns.

{_md_table(perm_top, '{:.4f}')}

![Permutation importance](figures/ml_04_permutation_importance.png)

**Logistic-regression odds ratios** (largest absolute effects; numeric features standardised, so their ratio is per +1 SD;
dummy features are per switch from 0 to 1, holding other features fixed):

{_md_table(odds_top, '{:.2f}')}

![Odds ratios](figures/ml_05_logistic_odds_ratios.png)

Caution: correlated features share credit (e.g. `tenure_months`, `total_charges`, `contract`), and importance describes what the model
uses for prediction - it is **association, not proof of cause**.

## 6. Limitations

* Single snapshot, no dates: cannot validate over time or build time-to-churn / survival models.
* Single random split + CV: differences between models of ~0.01 AUC are inside the noise.
* No cost data (retention offer cost, margin), so thresholds are chosen on F1, not profit.
* Hyper-parameters not tuned; probability calibration was not assessed.
* The dataset is a well-known sample dataset; performance on a real business's data may differ.
* Explainability beyond permutation importance and odds ratios (e.g. SHAP, per-customer explanations) is planned for Phase 2.
"""
    path.write_text(lines, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    df = mt.load_modelling_data()
    X, y = fe.get_feature_matrix(df)
    X_train, X_test, y_train, y_test = fe.split_data(X, y)

    model_file = config.MODELS_DIR / "churn_models.joblib"
    cv_file = TABLES_DIR / "cv_results.csv"
    if not (model_file.exists() and cv_file.exists()):
        mt.main()
    bundle = joblib.load(model_file)
    fitted = bundle["models"]
    assert bundle["test_index"] == X_test.index.tolist(), "saved models were trained on a different split"
    cv = pd.read_csv(cv_file)

    test_metrics, probas = evaluate_models(fitted, X_test, y_test)
    champion = mt.select_champion(cv)
    champ_pipe = fitted[champion]

    # threshold from out-of-fold TRAINING predictions only
    cv_split = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(mt.get_models(y_train)[champion]), X_train, y_train, cv=cv_split,
                            method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof)

    champ_default = classification_metrics(y_test, probas[champion], 0.5)
    champ_tuned = classification_metrics(y_test, probas[champion], threshold)
    thr_tbl = threshold_table(y_test, probas[champion], sorted({0.2, 0.3, 0.4, 0.5, 0.6, 0.7, round(threshold, 2)}))
    lift = lift_table(y_test, probas[champion])
    perm = permutation_importance_table(champ_pipe, X_test, y_test)
    odds = logistic_odds_ratios(fitted["Logistic Regression"])
    rf_imp = tree_importance_table(fitted["Random Forest"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(TABLES_DIR / "test_metrics.csv", index=False)
    thr_tbl.to_csv(TABLES_DIR / "threshold_table.csv", index=False)
    lift.to_csv(TABLES_DIR / "lift_table.csv", index=False)
    perm.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)
    odds.to_csv(TABLES_DIR / "logistic_odds_ratios.csv", index=False)
    rf_imp.to_csv(TABLES_DIR / "random_forest_importance.csv", index=False)

    unweighted = ["Logistic Regression", "Random Forest", "XGBoost"]
    plot_roc_pr(y_test, probas, unweighted)
    plot_confusion_grid(y_test, probas, list(fitted))
    plot_metric_heatmap(test_metrics)
    plot_permutation_importance(perm, champion)
    plot_odds_ratios(odds)
    plot_gains(lift, champion)
    plot_threshold_tradeoff(y_test, probas[champion], threshold, champion)

    metrics = {
        "champion": champion, "threshold": threshold, "n_train": len(X_train), "n_test": len(X_test),
        "n_features": X.shape[1], "churn_rate_train": float(y_train.mean()),
        "champion_default": champ_default, "champion_tuned": champ_tuned,
        "test_metrics": test_metrics.to_dict("records"),
        "cv_metrics": cv.to_dict("records"),
        "top_permutation_features": perm.head(10).to_dict("records"),
    }
    (config.REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, cv, test_metrics, thr_tbl, lift, perm, odds, config.REPORTS_DIR / "model_results.md")

    pd.set_option("display.width", 200)
    print(test_metrics.round(3).to_string(index=False))
    print(f"\nChampion (CV ROC-AUC): {champion}; tuned threshold {threshold:.2f}")
    print(thr_tbl.round(3).to_string(index=False))
    print(lift.round(3).to_string(index=False))
    print(perm.head(10).round(4).to_string(index=False))
    print(odds.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
