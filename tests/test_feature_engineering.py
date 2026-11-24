"""Tests for src/feature_engineering.py: engineered features, model inputs, leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config, feature_engineering as fe


def _row(**overrides) -> pd.DataFrame:
    base = {
        "customer_id": "X-1",
        "tenure_months": 10,
        "monthly_charges": 50.0,
        "total_charges": 500.0,
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "payment_method": "Mailed check",
        "partner": "No",
        "dependents": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


# --------------------------------------------------------------------------- #
# Feature logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tenure, expected",
    [(0, "0-12 months"), (12, "0-12 months"), (13, "13-24 months"), (24, "13-24 months"),
     (25, "25-48 months"), (48, "25-48 months"), (49, "49-72 months"), (72, "49-72 months")],
)
def test_tenure_group_boundaries(tenure, expected):
    assert fe.add_tenure_group(_row(tenure_months=tenure))["tenure_group"].iloc[0] == expected


@pytest.mark.parametrize(
    "charge, expected",
    [(18.25, "<$35"), (34.99, "<$35"), (35.0, "$35-70"), (69.99, "$35-70"),
     (70.0, "$70-90"), (89.99, "$70-90"), (90.0, "$90+"), (118.75, "$90+")],
)
def test_charge_band_boundaries(charge, expected):
    assert fe.add_charge_band(_row(monthly_charges=charge))["monthly_charge_band"].iloc[0] == expected


def test_service_counts():
    df = _row(online_security="Yes", tech_support="Yes", streaming_tv="Yes")
    out = fe.add_service_counts(df)
    assert out["num_addon_services"].iloc[0] == 3
    assert out["num_support_services"].iloc[0] == 2  # streaming is not a support service
    assert out["has_support_service"].iloc[0] == "Yes"
    none = fe.add_service_counts(_row())
    assert none["num_addon_services"].iloc[0] == 0
    assert none["has_support_service"].iloc[0] == "No"


def test_billing_features_handle_zero_tenure():
    out = fe.add_billing_features(_row(tenure_months=0, total_charges=0.0))
    assert out["avg_monthly_spend"].iloc[0] == 0.0
    assert np.isfinite(out["avg_monthly_spend"]).all()
    out = fe.add_billing_features(_row(tenure_months=10, total_charges=505.0))
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(50.5)


@pytest.mark.parametrize(
    "method, expected",
    [("Bank transfer (automatic)", "Yes"), ("Credit card (automatic)", "Yes"),
     ("Electronic check", "No"), ("Mailed check", "No")],
)
def test_autopay_flag(method, expected):
    assert fe.add_billing_features(_row(payment_method=method))["is_autopay"].iloc[0] == expected


def test_household_flag():
    assert fe.add_household_flag(_row())["has_household"].iloc[0] == "No"
    assert fe.add_household_flag(_row(partner="Yes"))["has_household"].iloc[0] == "Yes"
    assert fe.add_household_flag(_row(dependents="Yes"))["has_household"].iloc[0] == "Yes"


def test_engineer_features_does_not_mutate_input_or_change_rows(clean_df):
    before = clean_df.copy(deep=True)
    out = fe.engineer_features(clean_df)
    pd.testing.assert_frame_equal(clean_df, before)
    assert len(out) == len(clean_df)
    assert out[config.ID_COL].tolist() == clean_df[config.ID_COL].tolist()
    assert not out.isna().any().any()


def test_engineered_features_on_real_data(features_df):
    assert set(features_df["tenure_group"].unique()) == set(config.TENURE_LABELS)
    assert features_df["tenure_group"].value_counts().sum() == 7043
    assert features_df["num_addon_services"].between(0, 6).all()
    # customers without internet cannot hold internet add-ons
    no_net = features_df["internet_service"] == "No internet"
    assert (features_df.loc[no_net, "num_addon_services"] == 0).all()


# --------------------------------------------------------------------------- #
# Model inputs and leakage guards
# --------------------------------------------------------------------------- #
def test_feature_matrix_excludes_target_and_id(features_df):
    X, y = fe.get_feature_matrix(features_df)
    assert list(X.columns) == fe.MODEL_FEATURES
    for banned in (config.ID_COL, "churn", config.TARGET):
        assert banned not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y) == 7043


def test_feature_matrix_requires_columns():
    with pytest.raises(KeyError):
        fe.get_feature_matrix(pd.DataFrame({"tenure_months": [1]}))


def test_split_is_stratified_and_disjoint(features_df):
    X, y = fe.get_feature_matrix(features_df)
    X_tr, X_te, y_tr, y_te = fe.split_data(X, y)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert y_tr.mean() == pytest.approx(y.mean(), abs=0.002)
    assert y_te.mean() == pytest.approx(y.mean(), abs=0.002)


def test_split_is_reproducible(features_df):
    X, y = fe.get_feature_matrix(features_df)
    a = fe.split_data(X, y)[1].index
    b = fe.split_data(X, y)[1].index
    assert a.equals(b)


def test_preprocessor_fits_on_train_only_and_handles_unseen_category(features_df):
    X, y = fe.get_feature_matrix(features_df)
    X_tr, X_te, y_tr, _ = fe.split_data(X, y)
    pipe = fe.make_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_tr, y_tr)

    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    # scaler statistics come from the training rows, not the full data
    assert scaler.mean_[0] == pytest.approx(X_tr["tenure_months"].mean())
    assert scaler.mean_[0] != pytest.approx(X["tenure_months"].mean(), abs=1e-9)

    unseen = X_te.head(3).copy()
    unseen["payment_method"] = "Crypto"
    proba = pipe.predict_proba(unseen)[:, 1]
    assert proba.shape == (3,) and np.isfinite(proba).all()


def test_transformed_matrix_is_numeric_and_finite(features_df):
    X, y = fe.get_feature_matrix(features_df)
    X_tr, _, y_tr, _ = fe.split_data(X, y)
    prep = fe.build_preprocessor(scale_numeric=True).fit(X_tr)
    Z = prep.transform(X_tr)
    assert Z.shape[0] == len(X_tr)
    assert np.isfinite(Z).all()
    names = prep.get_feature_names_out()
    assert len(names) == Z.shape[1]
    assert "tenure_months" in names


def test_unscaled_preprocessor_keeps_raw_numeric_values(features_df):
    X, y = fe.get_feature_matrix(features_df)
    prep = fe.build_preprocessor(scale_numeric=False).fit(X)
    Z = prep.transform(X)
    assert np.allclose(Z[:, 0], X["tenure_months"].to_numpy())


def test_encode_for_export_is_fully_numeric(features_df):
    enc = fe.encode_for_export(features_df)
    non_id = enc.drop(columns=[config.ID_COL])
    assert all(pd.api.types.is_numeric_dtype(t) for t in non_id.dtypes)
    assert not enc.isna().any().any()
    assert enc[config.TARGET].sum() == 1869
    assert set(enc["contract_ordinal"].unique()) == {0, 1, 2}
