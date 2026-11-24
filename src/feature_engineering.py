"""Feature engineering and model-input preparation.

All engineered features are *row-local* (computed only from the same customer's own
columns, never from the target or from other rows), so they can be created before the
train/test split without leaking information. Statistics-based steps (scaling, one-hot
encoding) live inside scikit-learn pipelines and are fitted on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

AUTOPAY_METHODS = {"Bank transfer (automatic)", "Credit card (automatic)"}

# --------------------------------------------------------------------------- #
# Engineered features
# --------------------------------------------------------------------------- #


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket ``tenure_months`` into ordered lifecycle stages."""
    out = df.copy()
    out["tenure_group"] = pd.cut(
        out["tenure_months"], bins=config.TENURE_BINS, labels=config.TENURE_LABELS
    ).astype("string")
    return out


def add_charge_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket ``monthly_charges`` into price bands (left-open, right-closed)."""
    out = df.copy()
    out["monthly_charge_band"] = pd.cut(
        out["monthly_charges"], bins=config.CHARGE_BINS, labels=config.CHARGE_LABELS, right=False
    ).astype("string")
    return out


def add_service_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count add-on services held and flag customers with any protection/support service."""
    out = df.copy()
    out["num_addon_services"] = (out[config.INTERNET_ADDON_COLS] == "Yes").sum(axis=1).astype("int64")
    out["num_support_services"] = (out[config.SUPPORT_SERVICE_COLS] == "Yes").sum(axis=1).astype("int64")
    out["has_support_service"] = np.where(out["num_support_services"] > 0, "Yes", "No")
    out["has_support_service"] = out["has_support_service"].astype("string")
    return out


def add_billing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Billing behaviour flags and average realised monthly spend."""
    out = df.copy()
    out["is_autopay"] = np.where(out["payment_method"].isin(AUTOPAY_METHODS), "Yes", "No")
    out["is_autopay"] = out["is_autopay"].astype("string")
    # tenure 0 -> no billing history, so the average is defined as 0 rather than inf/NaN
    tenure = out["tenure_months"].replace(0, np.nan)
    out["avg_monthly_spend"] = (out["total_charges"] / tenure).fillna(0.0).round(2)
    return out


def add_household_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Customers living with a partner and/or dependents (household stability proxy)."""
    out = df.copy()
    has_household = (out["partner"] == "Yes") | (out["dependents"] == "Yes")
    out["has_household"] = np.where(has_household, "Yes", "No")
    out["has_household"] = out["has_household"].astype("string")
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature-engineering step to a clean DataFrame."""
    out = add_tenure_group(df)
    out = add_charge_band(out)
    out = add_service_counts(out)
    out = add_billing_features(out)
    out = add_household_flag(out)
    return out


# --------------------------------------------------------------------------- #
# Model inputs
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_addon_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must never be used as predictors: identifiers and the target (and its text twin).
EXCLUDED_FROM_MODEL = [config.ID_COL, "churn", config.TARGET]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for modelling, guaranteeing no target/ID leakage."""
    missing = [c for c in MODEL_FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    X = df[MODEL_FEATURES].copy()
    leaked = set(X.columns) & set(EXCLUDED_FROM_MODEL)
    if leaked:  # pragma: no cover - guarded by the constants above
        raise ValueError(f"Leaking columns in feature matrix: {leaked}")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("object")
    return X, df[config.TARGET].astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (preserves the churn ratio in both parts)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Column transformer: optional scaling for numerics, one-hot for categoricals.

    ``handle_unknown="ignore"`` keeps prediction robust to unseen categories, and
    ``drop="if_binary"`` avoids redundant columns for two-level features. The transformer is
    unfitted; fit it inside a Pipeline on training data only.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, scale_numeric: bool = True) -> Pipeline:
    """Preprocessing + estimator as one object, so CV re-fits preprocessing per fold."""
    return Pipeline([("prep", build_preprocessor(scale_numeric)), ("model", estimator)])


def encode_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Fully numeric version of the dataset for tools that cannot handle text.

    Uses only fixed, data-independent encodings (Yes/No -> 1/0, contract -> ordinal,
    remaining categoricals -> one-hot), so it is safe to produce before any split.
    The modelling pipeline does *not* use this file; it encodes inside ``make_pipeline``.
    """
    X, y = get_feature_matrix(df)
    out = pd.DataFrame(index=df.index)
    out[config.ID_COL] = df[config.ID_COL]
    for col in NUMERIC_FEATURES:
        out[col] = X[col]
    yes_no = [c for c in CATEGORICAL_FEATURES if set(X[c].unique()) <= {"Yes", "No"}]
    for col in yes_no:
        out[col] = (X[col] == "Yes").astype(int)
    out["is_male"] = (X["gender"] == "Male").astype(int)
    out["contract_ordinal"] = X["contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).astype(int)
    dummies = pd.get_dummies(X[["internet_service", "payment_method"]], dtype=int)
    dummies.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "") for c in dummies.columns]
    out = pd.concat([out, dummies], axis=1)
    out[config.TARGET] = y
    return out


def run() -> pd.DataFrame:
    """Create ``telco_churn_features.csv`` and ``telco_churn_encoded.csv`` from the clean file."""
    clean = pd.read_csv(config.CLEAN_FILE)
    feats = engineer_features(clean)
    feats.to_csv(config.FEATURES_FILE, index=False)
    encode_for_export(feats).to_csv(config.ENCODED_FILE, index=False)
    return feats


if __name__ == "__main__":
    feats = run()
    print(f"Wrote {feats.shape[0]:,} rows x {feats.shape[1]} columns to {config.FEATURES_FILE}")
