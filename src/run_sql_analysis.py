"""Execute every query in ``sql/analysis_queries.sql`` against PostgreSQL.

    python -m src.run_sql_analysis            # print all results
    python -m src.run_sql_analysis --export    # also write reports/sql_results/Qnn.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config
from .load_to_postgres import get_engine

QUERY_FILE = config.SQL_DIR / "analysis_queries.sql"
HEADER = re.compile(r"^-- (Q\d{2}) \| (.+)$", re.MULTILINE)


def parse_queries(path: Path = QUERY_FILE) -> list[dict]:
    """Split the SQL file into ``{id, title, sql}`` entries (one per ``-- Qnn | title`` header)."""
    content = path.read_text(encoding="utf-8")
    matches = list(HEADER.finditer(content))
    queries = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        queries.append({"id": m.group(1), "title": m.group(2).strip(), "sql": content[m.start():end].strip()})
    return queries


def run_query(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write each result to reports/sql_results/")
    args = ap.parse_args()
    engine = get_engine()
    out_dir = config.REPORTS_DIR / "sql_results"
    if args.export:
        out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for q in parse_queries():
        df = run_query(engine, q["sql"])
        print(f"\n=== {q['id']} | {q['title']}  ({len(df)} rows)")
        print(df.to_string(index=False))
        if args.export:
            df.to_csv(out_dir / f"{q['id']}.csv", index=False)


if __name__ == "__main__":
    main()
