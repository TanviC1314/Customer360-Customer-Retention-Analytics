"""Tests for the PostgreSQL layer.

* Tests that need no database (query-file structure, table splitting) always run.
* Database tests are skipped automatically when PostgreSQL is not reachable
  (start it with ``docker compose up -d`` then ``python -m src.load_to_postgres``).
  They reload nothing - they query the loaded tables and compare against pandas results
  computed from the cleaned CSV, so SQL and Python are cross-validated.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from src import config, load_to_postgres as loader, run_sql_analysis as sqlrun


# --------------------------------------------------------------------------- #
# No database required
# --------------------------------------------------------------------------- #
def test_query_file_has_at_least_20_numbered_queries():
    queries = sqlrun.parse_queries()
    ids = [q["id"] for q in queries]
    assert len(queries) >= 20
    assert ids == [f"Q{i:02d}" for i in range(1, len(queries) + 1)]  # contiguous numbering
    assert all(q["sql"].rstrip().endswith(";") for q in queries)


def test_query_file_uses_required_sql_features():
    sql = " ".join(q["sql"] for q in sqlrun.parse_queries()).upper()
    for feature in ("JOIN", "GROUP BY", "CASE WHEN", "WITH ", " OVER (", "(SELECT "):
        assert feature in sql, f"{feature.strip()} not used in analysis queries"


def test_split_tables_shapes_and_types(clean_df):
    tables = loader.split_tables(clean_df)
    assert {k: len(v) for k, v in tables.items()} == {"customers": 7043, "services": 7043, "billing": 7043}
    assert tables["customers"]["churned"].dtype == bool
    assert tables["customers"]["churned"].sum() == 1869
    assert tables["services"]["tech_support"].dtype == bool
    assert tables["billing"]["customer_id"].is_unique
    # every relational column exists exactly once across the three tables (besides the key)
    cols = [c for t in tables.values() for c in t.columns if c != "customer_id"]
    assert len(cols) == len(set(cols))


def test_schema_file_defines_keys_and_constraints():
    schema = (config.SQL_DIR / "schema.sql").read_text(encoding="utf-8").upper()
    assert schema.count("PRIMARY KEY") >= 4
    assert "REFERENCES CUSTOMERS" in schema
    assert "CHECK" in schema


# --------------------------------------------------------------------------- #
# Database-backed tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine():
    try:
        eng = loader.get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1 FROM customers LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL with loaded data not available: {exc}")
    return eng


@pytest.fixture(scope="module")
def results(engine):
    return {q["id"]: sqlrun.run_query(engine, q["sql"]) for q in sqlrun.parse_queries()}


def test_validation_queries_all_pass(engine):
    report = loader.run_validation(engine)
    assert len(report) >= 10
    assert (report["violations"] == 0).all(), report[report["violations"] != 0]


def test_every_analysis_query_runs_and_returns_rows(results):
    for qid, df in results.items():
        assert len(df) > 0, f"{qid} returned no rows"


def test_q01_q02_match_pandas(results, clean_df):
    q1, q2 = results["Q01"].iloc[0], results["Q02"].iloc[0]
    assert q1["total_customers"] == len(clean_df)
    assert q1["churned_customers"] == clean_df[config.TARGET].sum()
    assert q2["churn_rate_pct"] == pytest.approx(clean_df[config.TARGET].mean() * 100, abs=0.01)


@pytest.mark.parametrize("qid, col, key", [
    ("Q03", "contract", "contract"),
    ("Q04", "payment_method", "payment_method"),
    ("Q05", "internet_service", "internet_service"),
])
def test_group_churn_rates_match_pandas(results, clean_df, qid, col, key):
    expected = clean_df.groupby(col)[config.TARGET].mean().mul(100)
    got = results[qid].set_index(key)["churn_rate_pct"].astype(float)
    assert set(got.index) == set(expected.index)
    for k, v in got.items():
        assert v == pytest.approx(expected[k], abs=0.01), f"{qid} {k}"


def test_q06_q07_averages_match_pandas(results, clean_df):
    q6 = results["Q06"].set_index("status")
    q7 = results["Q07"].set_index("status")
    for status, label in (("Churned", "Yes"), ("Retained", "No")):
        sub = clean_df[clean_df["churn"] == label]
        assert q6.loc[status, "avg_monthly_charges"] == pytest.approx(sub["monthly_charges"].mean(), abs=0.01)
        assert q7.loc[status, "avg_tenure_months"] == pytest.approx(sub["tenure_months"].mean(), abs=0.01)


def test_q10_tenure_bands_match_python_feature(results, features_df):
    expected = features_df.groupby("tenure_group")[config.TARGET].agg(["count", "sum"])
    got = results["Q10"].set_index("tenure_band")
    for band, row in expected.iterrows():
        assert got.loc[band, "customers"] == row["count"]
        assert got.loc[band, "churned"] == row["sum"]
    assert got["cumulative_share_of_churners_pct"].iloc[-1] == pytest.approx(100.0)


def test_q08_segments_partition_all_customers(results):
    q8 = results["Q08"]
    assert q8["customers"].sum() == 7043
    assert q8["share_of_customers_pct"].sum() == pytest.approx(100.0, abs=0.2)


def test_q12_revenue_reconciles_with_pandas(results, clean_df):
    q12 = results["Q12"]
    assert q12["active_mrr"].sum() == pytest.approx(clean_df.loc[clean_df[config.TARGET] == 0, "monthly_charges"].sum(), abs=3)
    assert q12["lost_mrr"].sum() == pytest.approx(clean_df.loc[clean_df[config.TARGET] == 1, "monthly_charges"].sum(), abs=3)


def test_q16_revenue_at_risk_is_bounded_by_active_mrr(results):
    row = results["Q16"].iloc[0]
    assert 0 < row["exposed_mrr"] < row["active_mrr"]
    assert row["exposed_annualised_revenue"] == pytest.approx(row["exposed_mrr"] * 12, abs=12)


def test_q19_rollup_totals(results):
    q19 = results["Q19"]
    grand = q19[q19["grouping_level"] == 3].iloc[0]
    assert grand["customers"] == 7043 and grand["churned"] == 1869
    detail = q19[q19["grouping_level"] == 0]
    assert detail["customers"].sum() == 7043


def test_q22_more_support_services_means_lower_churn(results):
    rates = results["Q22"]["churn_rate_pct"].astype(float).tolist()
    assert rates == sorted(rates, reverse=True)


def test_q09_deciles_cover_everyone(results):
    q9 = results["Q09"]
    assert q9["customers"].sum() == 7043
    assert q9["share_of_revenue_pct"].sum() == pytest.approx(100.0, abs=0.3)
