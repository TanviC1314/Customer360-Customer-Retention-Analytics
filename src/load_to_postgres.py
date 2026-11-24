"""Create the PostgreSQL schema, load the cleaned dataset and run the validation queries.

Credentials come from environment variables / ``.env`` (see ``.env.example``)::

    docker compose up -d
    python -m src.load_to_postgres
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config

YES_NO_TO_BOOL = {"Yes": True, "No": False}

CUSTOMER_COLS = ["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure_months", "churned"]
SERVICE_COLS = ["customer_id", "phone_service", "multiple_lines", "internet_service", *config.INTERNET_ADDON_COLS]
BILLING_COLS = ["customer_id", "contract", "paperless_billing", "payment_method", "monthly_charges", "total_charges"]
BOOL_COLS = [
    "senior_citizen", "partner", "dependents", "churned", "phone_service", "multiple_lines",
    *config.INTERNET_ADDON_COLS, "paperless_billing",
]


def split_tables(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the flat cleaned dataset into the three relational tables (pure function)."""
    df = clean.copy()
    df["churned"] = df["churn"]
    for col in BOOL_COLS:
        df[col] = df[col].map(YES_NO_TO_BOOL).astype(bool)
    return {
        "customers": df[CUSTOMER_COLS].copy(),
        "services": df[SERVICE_COLS].copy(),
        "billing": df[BILLING_COLS].copy(),
    }


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or config.database_url(), future=True)


def wait_for_database(engine: Engine, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as exc:  # noqa: BLE001 - surface the last error after the timeout
            if time.time() > deadline:
                raise RuntimeError(f"PostgreSQL not reachable: {exc}") from exc
            time.sleep(2)


def run_sql_script(engine: Engine, path: Path) -> None:
    """Execute a multi-statement SQL file (used for the schema)."""
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(Path(path).read_text(encoding="utf-8"))
        raw.commit()
    finally:
        raw.close()


def run_query_file(engine: Engine, path: Path) -> pd.DataFrame:
    """Run a single-statement query file and return the result as a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_query(text(Path(path).read_text(encoding="utf-8")), conn)


def load_tables(engine: Engine, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Insert tables in dependency order (customers first); returns rows loaded per table."""
    counts = {}
    for name in ("customers", "services", "billing"):
        tables[name].to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
        counts[name] = len(tables[name])
    return counts


def run_validation(engine: Engine) -> pd.DataFrame:
    return run_query_file(engine, config.SQL_DIR / "validation_queries.sql")


def main() -> int:
    clean = pd.read_csv(config.CLEAN_FILE)
    engine = get_engine()
    wait_for_database(engine)
    run_sql_script(engine, config.SQL_DIR / "schema.sql")
    counts = load_tables(engine, split_tables(clean))
    print("Loaded rows:", counts)

    report = run_validation(engine)
    print(report.to_string(index=False))
    failed = report[report["violations"] != 0]
    if not failed.empty:
        print(f"\n{len(failed)} validation check(s) FAILED", file=sys.stderr)
        return 1
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
