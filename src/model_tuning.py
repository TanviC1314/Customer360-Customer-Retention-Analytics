"""Hyper-parameter tuning for the churn models (Phase 2, enhancement 2).

Uses ``RandomizedSearchCV`` inside a leakage-safe pipeline so the preprocessor is
re-fitted inside every inner CV fold. The champion from Phase 1 CV is tuned first;
the tuned pipeline is then compared with the untuned champion on the held-out test set.

    python -m src.model_tuning
"""

from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config, feature_engineering as fe, model_training as mt

warnings.filterwarnings("ignore", category=UserWarning)

# Search spaces keyed on the classifier class name (the pipeline step is always "model")
SEARCH_SPACES: dict[str, dict] = {
    "LogisticRegression": {
        "model__C": loguniform(1e-3, 1e3),
        "model__solver": ["lbfgs", "saga"],
        "model__max_iter": [1000, 2000, 5000],
    },
    "RandomForestClassifier": {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": randint(1, 20),
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBClassifier": {
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 8),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-4, 10),
        "model__reg_lambda": loguniform(1e-4, 10),
    },
}

N_ITER = 30
CV_INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)


def _classifier_name(pipe: Pipeline) -> str:
    return type(pipe.named_steps["model"]).__name__


def tune_pipeline(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = N_ITER,
    seed: int = config.RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame]:
    """Random search over ``pipe``; return (best_pipeline, cv_results_df)."""
    cname = _classifier_name(pipe)
    space = SEARCH_SPACES.get(cname, {})
    if not space:
        raise ValueError(f"No search space defined for {cname}")
    rscv = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=CV_INNER,
        refit=True,
        n_jobs=-1,
        random_state=seed,
        return_train_score=False,
    )
    t0 = time.time()
    rscv.fit(X_train, y_train)
    elapsed = time.time() - t0
    results = pd.DataFrame(rscv.cv_results_)
    results["elapsed_seconds"] = elapsed
    return rscv.best_estimator_, results


def compare_on_test(
    baseline: Pipeline,
    tuned: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """AUC, accuracy, recall (threshold 0.5) on the held-out test set."""
    from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

    rows = []
    for label, pipe in [("baseline (phase 1)", baseline), ("tuned", tuned)]:
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "model": label,
            "test_roc_auc": roc_auc_score(y_test, prob),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def _best_params_table(best_pipe: Pipeline) -> pd.DataFrame:
    params = {k.replace("model__", ""): v for k, v in best_pipe.get_params().items() if k.startswith("model__")}
    return pd.DataFrame(list(params.items()), columns=["parameter", "value"])


def write_report(
    champion_name: str,
    best_pipe: Pipeline,
    cv_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_auc = cv_results["mean_test_score"].max()
    best_std = cv_results.loc[cv_results["mean_test_score"].idxmax(), "std_test_score"]
    baseline_test = comparison.loc[comparison["model"].str.startswith("baseline"), "test_roc_auc"].iloc[0]
    tuned_test = comparison.loc[comparison["model"] == "tuned", "test_roc_auc"].iloc[0]
    delta = tuned_test - baseline_test
    sign = "+" if delta >= 0 else ""
    params_tbl = _best_params_table(best_pipe)

    def tbl(d: pd.DataFrame, fmts: dict | None = None) -> str:
        fmts = fmts or {}
        head = "| " + " | ".join(str(c) for c in d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
        rows = []
        for _, row in d.iterrows():
            cells = []
            for c in d.columns:
                v = row[c]
                cells.append(fmts.get(c, "{}").format(v) if isinstance(v, float) else str(v))
            rows.append("| " + " | ".join(cells) + " |")
        return head + "\n".join(rows)

    comp_fmts = {c: "{:.4f}" for c in comparison.columns if c != "model"}
    text = f"""# Hyper-parameter tuning (Phase 2, enhancement 2)

_Generated by `python -m src.model_tuning`; every number below was computed from the data._

## Method

* **Champion from Phase 1:** {champion_name} (selected by CV ROC-AUC on training data).
* **Tuning:** `RandomizedSearchCV`, {N_ITER} iterations, 5-fold stratified inner CV, scoring = ROC-AUC.
* The preprocessor is inside the pipeline and re-fitted in every inner fold — no leakage.
* Best hyper-parameters are selected on training data only; test set is touched exactly once for the final comparison.

## Best hyper-parameters

Best inner-CV ROC-AUC: **{best_auc:.4f} ± {best_std:.4f}** (std across 5 folds)

{tbl(params_tbl)}

## Test-set comparison (threshold 0.5)

{tbl(comparison, comp_fmts)}

**AUC change:** {sign}{delta:.4f} ({sign}{delta * 100:.2f} pp). {"Improvement is marginal — the fixed Phase 1 hyper-parameters were already close to the optimum for this dataset." if abs(delta) < 0.01 else "Meaningful improvement from tuning."}

## Interpretation

Hyper-parameter tuning is not guaranteed to improve the held-out score: with only ~5,600 training rows and 5-fold CV, the landscape is noisy. Any gain under ~0.01 AUC is within estimation uncertainty and should not be over-interpreted. The tuned pipeline is saved to disk for downstream use.

## Limitations

* Random search covers {N_ITER} configurations out of a large space; a grid search or Bayesian optimisation would be more exhaustive.
* The inner CV score is an optimistic estimate (search over many candidates inflates the apparent best score).
* Results are specific to this 80 / 20 split; a different seed would give slightly different numbers.
"""
    (config.REPORTS_DIR / "tuning_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = mt.load_modelling_data()
    X, y = fe.get_feature_matrix(df)
    X_train, X_test, y_train, y_test = fe.split_data(X, y)

    models = mt.get_models(y_train)
    # Identify champion from Phase 1 CV (re-run a quick CV)
    cv_results = mt.cross_validate_models(models, X_train, y_train)
    champion_name = mt.select_champion(cv_results)
    print(f"Phase 1 champion: {champion_name}")

    champion_pipe = models[champion_name]
    print(f"Tuning {champion_name} ({N_ITER} iterations)…")
    best_pipe, rscv_results = tune_pipeline(champion_pipe, X_train, y_train)

    # Fit the untuned champion on the full training set for the fair comparison
    baseline_pipe = models[champion_name].fit(X_train, y_train)
    comparison = compare_on_test(baseline_pipe, best_pipe, X_test, y_test)

    write_report(champion_name, best_pipe, rscv_results, comparison)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"tuned_pipeline": best_pipe, "champion_name": champion_name}, config.MODELS_DIR / "tuned_model.joblib")
    (config.REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.REPORTS_DIR / "tables" / "tuning_comparison.csv", index=False)

    pd.set_option("display.width", 160)
    print(comparison.round(4).to_string(index=False))
    print(f"\nTuning report → {config.REPORTS_DIR / 'tuning_results.md'}")


if __name__ == "__main__":
    main()
