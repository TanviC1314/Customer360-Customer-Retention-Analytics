"""Tests for src/data_cleaning.py - both unit tests on tiny frames and checks against the real file."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src import config, data_cleaning


# --------------------------------------------------------------------------- #
# Unit tests on the tiny raw-schema frame
# --------------------------------------------------------------------------- #
def test_total_charges_blank_for_new_customer_becomes_zero(mini_raw):
    df = data_cleaning.strip_whitespace(data_cleaning.rename_columns(mini_raw))
    out, info = data_cleaning.clean_total_charges(df)
    assert out["total_charges"].dtype == float
    assert out.loc[0, "total_charges"] == 0.0
    assert info == {
        "blank_total_charges": 1,
        "set_to_zero_new_customers": 1,
        "estimated_from_monthly_x_tenure": 0,
    }
    assert out.loc[1, "total_charges"] == pytest.approx(1086.0)


def test_total_charges_blank_for_existing_customer_is_estimated_not_dropped(mini_raw):
    mini_raw.loc[1, "TotalCharges"] = ""  # a billed customer with a missing value
    df = data_cleaning.strip_whitespace(data_cleaning.rename_columns(mini_raw))
    out, info = data_cleaning.clean_total_charges(df)
    assert len(out) == len(df)
    assert out.loc[1, "total_charges"] == pytest.approx(90.5 * 12)
    assert info["estimated_from_monthly_x_tenure"] == 1
    assert not out["total_charges"].isna().any()


def test_standardise_categoricals_collapses_redundant_levels(mini_raw):
    df = data_cleaning.rename_columns(mini_raw)
    out, info = data_cleaning.standardise_categoricals(df)
    for col in config.INTERNET_ADDON_COLS:
        assert set(out[col]) <= {"Yes", "No"}
    assert out.loc[0, "multiple_lines"] == "No"
    assert out.loc[2, "internet_service"] == "No internet"
    assert list(out["senior_citizen"]) == ["No", "Yes", "No", "No"]
    assert info["recoded_values"]["online_security"] == 1


def test_strip_whitespace_removes_padding():
    df = pd.DataFrame({"contract": [" Month-to-month ", "One year"], "n": [1, 2]})
    out = data_cleaning.strip_whitespace(df)
    assert out["contract"].tolist() == ["Month-to-month", "One year"]
    assert out["n"].tolist() == [1, 2]


def test_encode_target_maps_yes_no():
    out = data_cleaning.encode_target(pd.DataFrame({"churn": ["Yes", "No", "No"]}))
    assert out[config.TARGET].tolist() == [1, 0, 0]
    assert out[config.TARGET].dtype == "int64"


def test_clean_data_on_mini_frame_passes_validation(mini_raw):
    clean, log = data_cleaning.clean_data(mini_raw)
    assert log.validation_problems == []
    assert len(clean) == 4
    assert clean["tenure_months"].dtype == "int64"
    assert clean["total_charges"].dtype == "float64"
    assert not clean.isna().any().any()


# --------------------------------------------------------------------------- #
# Validation rules catch bad data
# --------------------------------------------------------------------------- #
def test_validation_flags_duplicate_ids_and_bad_values(clean_df):
    bad = clean_df.head(5).copy()
    bad.loc[bad.index[1], config.ID_COL] = bad.loc[bad.index[0], config.ID_COL]
    bad.loc[bad.index[2], "contract"] = "Three year"
    bad.loc[bad.index[3], "tenure_months"] = 99
    bad.loc[bad.index[4], "monthly_charges"] = -1.0
    problems = " | ".join(data_cleaning.validate_clean_data(bad))
    assert "duplicate customer IDs" in problems
    assert "contract" in problems
    assert "tenure_months" in problems
    assert "monthly_charges" in problems


def test_validation_flags_inconsistent_service_flags(clean_df):
    bad = clean_df.head(2).copy()
    bad.loc[bad.index[0], "internet_service"] = "No internet"
    bad.loc[bad.index[0], "tech_support"] = "Yes"
    bad.loc[bad.index[1], "phone_service"] = "No"
    bad.loc[bad.index[1], "multiple_lines"] = "Yes"
    problems = " | ".join(data_cleaning.validate_clean_data(bad))
    assert "without internet" in problems
    assert "without phone service" in problems


def test_inspect_dataset_counts_blank_strings(mini_raw):
    profile = data_cleaning.inspect_dataset(mini_raw)
    assert profile["blank_strings"] == {"TotalCharges": 1}
    assert profile["duplicate_ids"] == 0
    assert profile["n_rows"] == 4


# --------------------------------------------------------------------------- #
# Checks against the real raw file (reconciliation with the source)
# --------------------------------------------------------------------------- #
def test_raw_file_dimensions(raw_df):
    assert raw_df.shape == (7043, 21)


def test_raw_file_is_unmodified():
    digest = hashlib.sha256(config.RAW_FILE.read_bytes()).hexdigest()
    assert digest == "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def test_real_data_blank_total_charges_are_all_zero_tenure(raw_df):
    blank = raw_df["TotalCharges"].astype("string").str.strip() == ""
    assert blank.sum() == 11
    assert (raw_df.loc[blank, "tenure"] == 0).all()
    assert (raw_df.loc[blank, "Churn"] == "No").all()


def test_clean_data_keeps_all_customers_and_ids(raw_df, clean_df):
    assert len(clean_df) == len(raw_df) == 7043
    assert set(clean_df[config.ID_COL]) == set(raw_df["customerID"])
    assert clean_df[config.ID_COL].is_unique


def test_clean_data_has_no_missing_values_and_correct_dtypes(clean_df):
    assert not clean_df.isna().any().any()
    assert clean_df["total_charges"].dtype == "float64"
    assert clean_df["monthly_charges"].dtype == "float64"
    assert clean_df["tenure_months"].dtype == "int64"
    assert clean_df[config.TARGET].dtype == "int64"


def test_clean_data_reconciles_with_raw_counts(raw_df, clean_df):
    """Churn counts and revenue totals must be identical before and after cleaning."""
    assert clean_df[config.TARGET].sum() == (raw_df["Churn"] == "Yes").sum() == 1869
    assert clean_df["monthly_charges"].sum() == pytest.approx(raw_df["MonthlyCharges"].sum())
    raw_total = pd.to_numeric(raw_df["TotalCharges"], errors="coerce").sum()
    assert clean_df["total_charges"].sum() == pytest.approx(raw_total)
    # values per customer unchanged
    merged = raw_df.merge(clean_df, left_on="customerID", right_on=config.ID_COL)
    assert (merged["tenure"] == merged["tenure_months"]).all()
    assert np.allclose(merged["MonthlyCharges"], merged["monthly_charges"])
    assert (merged["Contract"] == merged["contract"]).all()
    assert (merged["Churn"] == merged["churn"]).all()


def test_cleaning_does_not_mutate_input(raw_df):
    snapshot = raw_df.copy(deep=True)
    data_cleaning.clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_cleaned_data_passes_validation(clean_bundle):
    _, log = clean_bundle
    assert log.validation_problems == []


def test_processed_file_matches_pipeline_output(clean_df):
    """The committed processed CSV equals a fresh run of the pipeline."""
    saved = pd.read_csv(config.CLEAN_FILE)
    assert saved.shape == clean_df.shape
    assert list(saved.columns) == list(clean_df.columns)
    assert saved[config.TARGET].sum() == clean_df[config.TARGET].sum()
    assert np.allclose(saved["total_charges"], clean_df["total_charges"])
