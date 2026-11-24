"""Train and cross-validate the churn models.

Design choices (all documented in ``reports/model_results.md``):

* One stratified 80/20 train/test split (``random_state=42``). The test set is not touched
  until final evaluation; model *selection* uses 5-fold stratified cross-validation on the
  training data only.
* Preprocessing (scaling, one-hot encoding) lives inside each ``Pipeline`` so it is re-fitted
  inside every CV fold - no leakage from validation folds or the test set.
* Class imbalance (~26.5 % churners): each model is trained both unweighted and with class
  weighting (``class_weight="balanced"`` / XGBoost ``scale_pos_weight``) so the effect is
  measured, not assumed.
* Fixed, sensible hyper-parameters. Systematic tuning is a planned Phase 2 enhancement.

    python -m src.model_training
"""

from __future__ import annotations

import time

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, feature_engineering as fe

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_modelling_data() -> pd.DataFrame:
    """Clean dataset + engineered features (re-created from the clean CSV if missing)."""
    if config.FEATURES_FILE.exists():
        return pd.read_csv(config.FEATURES_FILE)
    return fe.engineer_features(pd.read_csv(config.CLEAN_FILE))


def get_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Unfitted pipelines for every model variant, keyed by display name."""
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos  # XGBoost's class-weight equivalent, computed from TRAINING labels only
    rs = config.RANDOM_STATE

    def xgb(**kw):
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=rs, n_jobs=-1, **kw,
        )

    def rf(**kw):
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=rs, n_jobs=-1, **kw
        )

    return {
        "Logistic Regression": fe.make_pipeline(LogisticRegression(max_iter=2000, random_state=rs), True),
        "Logistic Regression (balanced)": fe.make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs), True),
        "Random Forest": fe.make_pipeline(rf(), False),
        "Random Forest (balanced)": fe.make_pipeline(rf(class_weight="balanced_subsample"), False),
        "XGBoost": fe.make_pipeline(xgb(), False),
        "XGBoost (balanced)": fe.make_pipeline(xgb(scale_pos_weight=spw), False),
    }


def get_baseline() -> Pipeline:
    """Majority-class dummy: shows what 'accuracy' means when you predict nobody churns."""
    return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """5-fold stratified CV on the training set; returns mean and std for each metric."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        t0 = time.time()
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            row[f"cv_{m}_mean"] = res[f"test_{m}"].mean()
            row[f"cv_{m}_std"] = res[f"test_{m}"].std()
        row["fit_seconds_per_fold"] = (time.time() - t0) / config.CV_FOLDS
        rows.append(row)
    return pd.DataFrame(rows)


def select_champion(cv_results: pd.DataFrame, candidates: list[str] | None = None) -> str:
    """Pick the best model by CV ROC-AUC (training data only - never the test set)."""
    df = cv_results if candidates is None else cv_results[cv_results["model"].isin(candidates)]
    return df.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"]


def fit_models(models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    return {name: pipe.fit(X_train, y_train) for name, pipe in models.items()}


def main() -> None:
    df = load_modelling_data()
    X, y = fe.get_feature_matrix(df)
    X_train, X_test, y_train, y_test = fe.split_data(X, y)
    print(f"Train {X_train.shape}, test {X_test.shape}; churn rate train {y_train.mean():.3f} / test {y_test.mean():.3f}")

    models = get_models(y_train)
    cv_results = cross_validate_models(models, X_train, y_train)
    print(cv_results.round(4).to_string(index=False))

    fitted = fit_models(models, X_train, y_train)
    config.MODELS_DIR.mkdir(exist_ok=True)
    tables = config.REPORTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(tables / "cv_results.csv", index=False)
    joblib.dump(
        {"models": fitted, "test_index": X_test.index.tolist(), "train_index": X_train.index.tolist()},
        config.MODELS_DIR / "churn_models.joblib",
    )
    print(f"Saved fitted models to {config.MODELS_DIR / 'churn_models.joblib'} (gitignored)")


if __name__ == "__main__":
    main()
