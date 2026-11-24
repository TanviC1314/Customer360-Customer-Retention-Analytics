"""Shared fixtures.

``raw_df`` loads the real IBM file (the source of truth); ``mini_raw`` is a tiny hand-written
frame in the raw schema used to test individual cleaning rules in isolation (test input only,
never used for analysis).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_cleaning, feature_engineering  # noqa: E402


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return data_cleaning.load_raw_data(config.RAW_FILE)


@pytest.fixture(scope="session")
def clean_bundle(raw_df):
    return data_cleaning.clean_data(raw_df)


@pytest.fixture(scope="session")
def clean_df(clean_bundle) -> pd.DataFrame:
    return clean_bundle[0]


@pytest.fixture(scope="session")
def features_df(clean_df) -> pd.DataFrame:
    return feature_engineering.engineer_features(clean_df)


@pytest.fixture()
def mini_raw() -> pd.DataFrame:
    """Four customers in the raw schema, covering the awkward cases."""
    return pd.DataFrame(
        {
            "customerID": ["A-1", "A-2", "A-3", "A-4"],
            "gender": ["Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [0, 12, 40, 5],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "Yes", "No", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["No", "No", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "No"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Two year", "Month-to-month", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [25.0, 90.5, 20.0, 60.0],
            "TotalCharges": [" ", "1086.0", "800.0", "300.5"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )
