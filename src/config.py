"""Central configuration: paths, schema constants and database settings.

Database credentials are read from environment variables (see ``.env.example``);
nothing sensitive is hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "dashboard_data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_FILE = DATA_PROCESSED_DIR / "telco_churn_clean.csv"
FEATURES_FILE = DATA_PROCESSED_DIR / "telco_churn_features.csv"
ENCODED_FILE = DATA_PROCESSED_DIR / "telco_churn_encoded.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "churn_flag"

# --- schema -----------------------------------------------------------------
ID_COL = "customer_id"

# Raw column name -> clean snake_case name
COLUMN_RENAMES = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

# Add-on services that are only meaningful for internet subscribers
INTERNET_ADDON_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
# "Support" style services (protect / help the customer) vs entertainment add-ons
SUPPORT_SERVICE_COLS = ["online_security", "online_backup", "device_protection", "tech_support"]
STREAMING_COLS = ["streaming_tv", "streaming_movies"]

BINARY_YES_NO_COLS = [
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
]

ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "senior_citizen": {"Yes", "No"},
    "partner": {"Yes", "No"},
    "dependents": {"Yes", "No"},
    "phone_service": {"Yes", "No"},
    "multiple_lines": {"Yes", "No"},
    "internet_service": {"DSL", "Fiber optic", "No internet"},
    "online_security": {"Yes", "No"},
    "online_backup": {"Yes", "No"},
    "device_protection": {"Yes", "No"},
    "tech_support": {"Yes", "No"},
    "streaming_tv": {"Yes", "No"},
    "streaming_movies": {"Yes", "No"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": {"Yes", "No"},
    "payment_method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "churn": {"Yes", "No"},
}

# Ordered category labels used by charts and engineered features
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
CHARGE_BINS = [0, 35, 70, 90, float("inf")]
CHARGE_LABELS = ["<$35", "$35-70", "$70-90", "$90+"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]


# --- database ---------------------------------------------------------------
def load_env_file(path: Path | None = None) -> None:
    """Load ``.env`` into ``os.environ`` (without overriding existing variables)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        return
    load_dotenv(path or PROJECT_ROOT / ".env", override=False)


def database_url() -> str:
    """Build the SQLAlchemy URL from environment variables."""
    load_env_file()
    try:
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
    except KeyError as exc:
        raise RuntimeError(
            f"Missing environment variable {exc}. Copy .env.example to .env and set the values."
        ) from exc
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5434")
    db = os.environ.get("POSTGRES_DB", "customer360")
    from urllib.parse import quote_plus

    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"
