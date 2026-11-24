"""Data inspection, cleaning and validation for the IBM Telco Customer Churn dataset.

The raw CSV is never modified. ``clean_data`` returns a new DataFrame together with a
log of every decision taken, which ``write_cleaning_report`` turns into
``reports/data_cleaning_report.md``.

Run end-to-end::

    python -m src.data_cleaning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Loading and inspection
# --------------------------------------------------------------------------- #


def load_raw_data(path: Path | str = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as stored (no type coercion beyond pandas defaults)."""
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a structural profile of a (raw or clean) DataFrame.

    Blank strings are counted separately from NaN because the raw Telco file encodes
    missing ``TotalCharges`` as a single space rather than an empty field.
    """
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    blank_counts = {
        col: int(df[col].astype("string").str.strip().eq("").sum()) for col in object_cols
    }
    id_col = next((c for c in ("customerID", config.ID_COL) if c in df.columns), None)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "blank_strings": {c: n for c, n in blank_counts.items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()) if id_col else None,
    }


# --------------------------------------------------------------------------- #
# Cleaning steps (each is a small pure function so it can be unit-tested)
# --------------------------------------------------------------------------- #


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to snake_case; unknown columns are left unchanged."""
    return df.rename(columns=config.COLUMN_RENAMES)


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every text column."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip()
    return out


def clean_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert ``total_charges`` to float and resolve blanks.

    * Blank / non-numeric values for customers with ``tenure_months == 0`` are real
      "not billed yet" cases and are set to ``0.0``.
    * Any other unparsable value (none exist in the IBM file) is estimated as
      ``monthly_charges * tenure_months`` so no row is silently dropped.
    """
    out = df.copy()
    raw = out["total_charges"]
    numeric = pd.to_numeric(raw.astype("string").str.strip().replace("", pd.NA), errors="coerce")
    bad = numeric.isna()
    new_customer = bad & (out["tenure_months"] == 0)
    other_bad = bad & ~new_customer

    numeric = numeric.astype(float)
    numeric[new_customer] = 0.0
    numeric[other_bad] = (out["monthly_charges"] * out["tenure_months"])[other_bad]
    out["total_charges"] = numeric
    return out, {
        "blank_total_charges": int(bad.sum()),
        "set_to_zero_new_customers": int(new_customer.sum()),
        "estimated_from_monthly_x_tenure": int(other_bad.sum()),
    }


def standardise_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse redundant categories and normalise binary columns to Yes/No.

    * ``"No internet service"`` / ``"No phone service"`` duplicate information already
      held in ``internet_service`` / ``phone_service``; they are mapped to ``"No"`` in the
      dependent columns so each service column is a clean Yes/No flag.
    * ``internet_service == "No"`` is relabelled ``"No internet"`` so it cannot be
      confused with a Yes/No answer.
    * ``senior_citizen`` (0/1 in the source) becomes Yes/No like the other flags.
    """
    out = df.copy()
    changes: dict[str, int] = {}

    for col in config.INTERNET_ADDON_COLS:
        mask = out[col] == "No internet service"
        changes[col] = int(mask.sum())
        out.loc[mask, col] = "No"

    mask = out["multiple_lines"] == "No phone service"
    changes["multiple_lines"] = int(mask.sum())
    out.loc[mask, "multiple_lines"] = "No"

    mask = out["internet_service"] == "No"
    changes["internet_service"] = int(mask.sum())
    out.loc[mask, "internet_service"] = "No internet"

    sc = out["senior_citizen"]
    if pd.api.types.is_numeric_dtype(sc):
        out["senior_citizen"] = sc.map({0: "No", 1: "Yes"}).astype("string")
    return out, {"recoded_values": changes}


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target ``churn_flag`` (1 = churned, 0 = retained)."""
    out = df.copy()
    out[config.TARGET] = out["churn"].map({"Yes": 1, "No": 0}).astype("int64")
    return out


def validate_clean_data(df: pd.DataFrame) -> list[str]:
    """Return a list of validation problems (an empty list means the data is valid)."""
    problems: list[str] = []

    if df[config.ID_COL].duplicated().any():
        problems.append(f"{int(df[config.ID_COL].duplicated().sum())} duplicate customer IDs")
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        problems.append(f"missing values remain in {cols}")

    for col, allowed in config.ALLOWED_VALUES.items():
        if col in df.columns:
            unexpected = set(df[col].dropna().unique()) - allowed
            if unexpected:
                problems.append(f"{col}: unexpected values {sorted(unexpected)}")

    if not df["tenure_months"].between(0, 72).all():
        problems.append("tenure_months outside 0-72")
    if not (df["monthly_charges"] > 0).all():
        problems.append("monthly_charges must be > 0")
    if not (df["total_charges"] >= 0).all():
        problems.append("total_charges must be >= 0")

    # Internal consistency rules between related columns
    no_net = df["internet_service"] == "No internet"
    if (df.loc[no_net, config.INTERNET_ADDON_COLS] == "Yes").any().any():
        problems.append("internet add-ons marked Yes for customers without internet")
    if ((df["phone_service"] == "No") & (df["multiple_lines"] == "Yes")).any():
        problems.append("multiple_lines is Yes without phone service")
    if config.TARGET in df.columns and not df[config.TARGET].isin([0, 1]).all():
        problems.append("churn_flag must be 0/1")
    return problems


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class CleaningLog:
    """Everything needed to document a cleaning run."""

    raw_profile: dict = field(default_factory=dict)
    clean_profile: dict = field(default_factory=dict)
    total_charges: dict = field(default_factory=dict)
    categorical: dict = field(default_factory=dict)
    whitespace_cells_changed: int = 0
    charge_consistency: dict = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    monthly_range: tuple[float, float] = (0.0, 0.0)


def charge_consistency_check(df: pd.DataFrame) -> dict:
    """Compare ``total_charges`` with ``monthly_charges * tenure_months`` (a sanity check).

    Total charges legitimately differ from this product (price changes, discounts,
    partial months), so this is reported for information, not used to alter data.
    """
    expected = df["monthly_charges"] * df["tenure_months"]
    mask = df["tenure_months"] > 0
    rel = ((df["total_charges"] - expected).abs() / expected.where(mask)).dropna()
    return {
        "customers_checked": int(mask.sum()),
        "median_abs_pct_diff": round(float(rel.median() * 100), 2),
        "share_within_10pct": round(float((rel <= 0.10).mean() * 100), 1),
        "max_abs_pct_diff": round(float(rel.max() * 100), 1),
    }


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Run the full cleaning pipeline on a raw Telco DataFrame."""
    log = CleaningLog(raw_profile=inspect_dataset(raw))

    df = rename_columns(raw)
    before = df.copy()
    df = strip_whitespace(df)
    # count text cells that actually changed (whitespace only)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    log.whitespace_cells_changed = int(
        sum((before[c].astype("string") != df[c]).fillna(False).sum() for c in text_cols)
    )

    df, log.total_charges = clean_total_charges(df)
    df, log.categorical = standardise_categoricals(df)
    df["tenure_months"] = df["tenure_months"].astype("int64")
    df["monthly_charges"] = df["monthly_charges"].astype("float64")
    df = encode_target(df)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string")

    log.charge_consistency = charge_consistency_check(df)
    log.monthly_range = (float(df["monthly_charges"].min()), float(df["monthly_charges"].max()))
    log.validation_problems = validate_clean_data(df)
    log.clean_profile = inspect_dataset(df)
    return df.reset_index(drop=True), log


def write_cleaning_report(log: CleaningLog, path: Path | str) -> None:
    """Render the cleaning decisions as a Markdown report."""
    raw, tc, cat = log.raw_profile, log.total_charges, log.categorical["recoded_values"]
    rows = "\n".join(f"| `{c}` | {t} |" for c, t in raw["dtypes"].items())
    recoded = "\n".join(f"| `{c}` | {n:,} |" for c, n in cat.items())
    cc = log.charge_consistency
    problems = "None - all validation checks passed." if not log.validation_problems else (
        "\n".join(f"- {p}" for p in log.validation_problems)
    )
    text = f"""# Data cleaning report

_Generated by `python -m src.data_cleaning`; do not edit by hand._

**Source:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (never modified).
**Output:** `data/processed/telco_churn_clean.csv`.

## 1. Inspection of the raw file

- Dimensions: **{raw['n_rows']:,} rows x {raw['n_columns']} columns**
- Duplicate rows: **{raw['duplicate_rows']}**; duplicate customer IDs: **{raw['duplicate_ids']}**
- Missing values (NaN): **{sum(raw['missing_values'].values())}**
- Blank-string values: **{json.dumps(raw['blank_strings'])}**

| Raw column | Raw dtype |
|---|---|
{rows}

Note `TotalCharges` is read as text, which is the symptom of the blank-value problem described below.

## 2. Decisions

| # | Issue found | Decision | Rationale |
|---|---|---|---|
| 1 | `TotalCharges` stored as text; **{tc['blank_total_charges']} blank (single-space) values** | Converted to float. Blanks set to **0.0** ({tc['set_to_zero_new_customers']} rows), none estimated ({tc['estimated_from_monthly_x_tenure']}) | Every blank belongs to a customer with `tenure = 0` (signed up, not yet billed), so 0 is the true value. Dropping them would remove {tc['blank_total_charges']} genuine, never-churned customers; mean-imputation would invent revenue. |
| 2 | No duplicates | None needed; uniqueness of `customer_id` is enforced in validation and as a database primary key | |
| 3 | `SeniorCitizen` is 0/1 while other flags are Yes/No | Converted to Yes/No | Consistent categorical handling. |
| 4 | `No internet service` repeated across 6 add-on columns; `No phone service` in `MultipleLines` | Mapped to `No` (see counts below) | The information already lives in `InternetService` / `PhoneService`, so the extra level is redundant and would only add collinear dummy columns. |
| 5 | `InternetService = "No"` is ambiguous next to Yes/No columns | Relabelled `No internet` | Clarity. |
| 6 | Column names mix CamelCase / lowercase | Renamed to snake_case (`tenure` -> `tenure_months`) | SQL- and Python-friendly. |
| 7 | Whitespace | Stripped from all text cells ({log.whitespace_cells_changed} cells changed, of which {tc['blank_total_charges']} are the blank `TotalCharges` cells) | Prevents phantom categories such as `"Yes "`. |
| 8 | Target `Churn` is Yes/No text | Kept as `churn` and added `churn_flag` (1/0) | Numeric target for modelling and rates. |

Values recoded in step 3/4/5:

| Column | Rows recoded |
|---|---|
{recoded}

## 3. Checks that changed nothing (documented for completeness)

- Categorical columns contain only the expected labels (no case/spelling variants such as `yes`/`Yes`).
- `tenure_months` is within 0-72, `monthly_charges` is between ${log.monthly_range[0]:.2f} and ${log.monthly_range[1]:.2f} (all > 0).
- `total_charges` vs `monthly_charges x tenure_months`: median absolute difference **{cc['median_abs_pct_diff']}%**,
  **{cc['share_within_10pct']}%** of customers within 10% (max {cc['max_abs_pct_diff']}%).
  Differences are expected (price changes, partial months), so no values were altered.
- Outliers: charge and tenure distributions are bounded and plausible; no values were removed or capped.

## 4. Validation of the cleaned data

{problems}

Cleaned dataset: **{log.clean_profile['n_rows']:,} rows x {log.clean_profile['n_columns']} columns**, no missing values.
"""
    Path(path).write_text(text, encoding="utf-8")


def run(
    raw_path: Path | str = config.RAW_FILE,
    out_path: Path | str = config.CLEAN_FILE,
    report_path: Path | str | None = config.REPORTS_DIR / "data_cleaning_report.md",
) -> pd.DataFrame:
    """Load the raw file, clean it, validate it and write the processed CSV + report."""
    raw = load_raw_data(raw_path)
    clean, log = clean_data(raw)
    if log.validation_problems:
        raise ValueError(f"Cleaned data failed validation: {log.validation_problems}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    if report_path is not None:
        write_cleaning_report(log, report_path)
    return clean


if __name__ == "__main__":
    out = run()
    print(f"Wrote {len(out):,} clean rows to {config.CLEAN_FILE}")
    print(np.round(out[["tenure_months", "monthly_charges", "total_charges"]].describe(), 2))
