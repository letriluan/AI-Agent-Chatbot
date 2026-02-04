#!/usr/bin/env python3
from __future__ import annotations

"""
Part 1: CSV -> SQLite loader.

This script is intentionally generic. It loads any folder of CSV files into SQLite,
creates tables with inferred column types, and applies primary and foreign key
constraints only when the data supports them.

What it does:
- Reads every *.csv in a folder
- Normalizes table and column names for SQLite safety
- Infers primary and foreign keys conservatively
- Creates tables using explicit SQL DDL (not pandas auto tables)
- Inserts data
- Keeps the script reusable across datasets (no hardcoded schema)
"""


import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Name + type helpers
# -----------------------------
def norm(name: str) -> str:
    """Normalize identifiers: lowercase, underscores, remove special chars."""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def sqlite_type(series: pd.Series) -> str:
    """Best-effort column type for SQLite."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


# -----------------------------
# PK / FK inference
# -----------------------------
def infer_primary_key(table: str, df: pd.DataFrame) -> Optional[str]:

    """
    Try to infer a primary key column.
    """

    if df.empty:
        return None

    t = norm(table)
    cols = list(df.columns)

    preferred = [c for c in cols if norm(c) in ("id", f"{t}_id")]
    candidates = preferred + [c for c in cols if c not in preferred]

    # Strict: unique + non-null
    for c in candidates:
        s = df[c]
        if s.notna().all() and s.nunique(dropna=False) == len(df):
            return c

    # Relaxed: unique among non-null, but avoid very sparse columns
    for c in candidates:
        s = df[c]
        if s.notna().mean() < 0.90:
            continue
        nn = s.dropna()
        if len(nn) > 0 and nn.nunique() == len(nn):
            return c

    return None


def base_table_name(t: str) -> str:
    """
    Reduce table name to a base for matching.
    Strips common prefixes and plural 's'.
    """
    n = norm(t)
    for p in ("sales_", "dim_", "fact_", "tbl_"):
        if n.startswith(p):
            n = n[len(p) :]
            break
    if n.endswith("s") and len(n) > 3:
        n = n[:-1]
    return n


def infer_foreign_keys(
    tables: Dict[str, pd.DataFrame],
    pks: Dict[str, Optional[str]],
) -> List[Tuple[str, str, str, str]]:
    
    """
    Infer FK relationships conservatively.

    Returns:
      (src_table, src_col, ref_table, ref_pk_col)
    """

    pk_sets: Dict[str, set[str]] = {}
    for t, pk in pks.items():
        if pk:
            pk_sets[t] = set(tables[t][pk].dropna().astype(str).tolist())

    fks: List[Tuple[str, str, str, str]] = []

    for src_t, df in tables.items():
        for src_c in df.columns:
            src_cn = norm(src_c)

            # Only consider ID-like columns
            if not (src_cn.endswith("_id") or src_cn.endswith("id")):
                continue

            src_vals = df[src_c].dropna().astype(str)
            if src_vals.empty:
                continue
            src_set = set(src_vals.tolist())

            for ref_t, ref_pk in pks.items():
                if not ref_pk or ref_t == src_t:
                    continue

                base = base_table_name(ref_t)
                ref_pk_n = norm(ref_pk)

                # Common naming patterns (real datasets are messy)
                name_match = src_cn in {
                    f"{base}_id",
                    f"{base}id",
                    f"{base}s_id",
                    f"{base}s" + "id",
                    ref_pk_n,
                }
                if not name_match:
                    continue

                parent_set = pk_sets.get(ref_t, set())
                if not parent_set:
                    continue

                # Only add FK if it's guaranteed valid
                if src_set - parent_set:
                    continue

                fks.append((src_t, src_c, ref_t, ref_pk))

    # de-dupe
    seen = set()
    out: List[Tuple[str, str, str, str]] = []
    for fk in fks:
        if fk not in seen:
            seen.add(fk)
            out.append(fk)
    return out


# -----------------------------
# Schema build + insert
# -----------------------------
def create_tables(conn: sqlite3.Connection, tables: Dict[str, pd.DataFrame],
                  pks: Dict[str, Optional[str]],
                  fks: List[Tuple[str, str, str, str]]) -> None:
    """Create tables with inferred PK/FK constraints."""
    fk_by_src: Dict[str, List[Tuple[str, str, str]]] = {}
    for src_t, src_c, ref_t, ref_c in fks:
        fk_by_src.setdefault(src_t, []).append((src_c, ref_t, ref_c))

    # Clean rebuild each run
    for t in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{t}";')

    for t, df in tables.items():
        col_defs = [f'"{c}" {sqlite_type(df[c])}' for c in df.columns]

        constraints: List[str] = []
        if pks.get(t):
            constraints.append(f'PRIMARY KEY("{pks[t]}")')

        for src_c, ref_t, ref_c in fk_by_src.get(t, []):
            constraints.append(f'FOREIGN KEY("{src_c}") REFERENCES "{ref_t}"("{ref_c}")')

        ddl = f'CREATE TABLE "{t}" (\n  ' + ",\n  ".join(col_defs + constraints) + "\n);"
        conn.execute(ddl)

    conn.commit()


def insert_all(conn: sqlite3.Connection, tables: Dict[str, pd.DataFrame]) -> None:

    """
    Insert data into all tables.
    """

    conn.execute("PRAGMA foreign_keys=OFF;")

    for t, df in tables.items():
        if df.empty:
            continue

        cols = list(df.columns)
        placeholders = ",".join(["?"] * len(cols))
        sql = (
            f'INSERT INTO "{t}" ('
            + ",".join([f'"{c}"' for c in cols])
            + f") VALUES ({placeholders});"
        )

        rows = df.where(pd.notna(df), None).values.tolist()
        conn.executemany(sql, rows)

    conn.commit()


def validate_foreign_keys(conn: sqlite3.Connection) -> None:
    """Turn FK enforcement back on and report any FK violations."""
    conn.execute("PRAGMA foreign_keys=ON;")
    problems = conn.execute("PRAGMA foreign_key_check;").fetchall()
    if problems:
        print("⚠️ Foreign key check found issues (dataset might be inconsistent):")
        for row in problems[:10]:
            print("  -", row)
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
        print()
    else:
        print("✅ Foreign key check passed.\n")


# -----------------------------
# Main
# -----------------------------
def main(data_dir: Path, db_path: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir.resolve()}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data folder: {data_dir.resolve()}")
    print(f"DB output  : {db_path.resolve()}")
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()

    tables: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_files:
        table_name = norm(csv_path.stem)
        df = pd.read_csv(csv_path)
        df.columns = [norm(c) for c in df.columns]
        tables[table_name] = df

    # Infer PK/FK
    pks = {t: infer_primary_key(t, df) for t, df in tables.items()}
    fks = infer_foreign_keys(tables, pks)

    print("🔎 Inferred keys summary:")
    for t, pk in pks.items():
        print(f"  - {t}: PK = {pk or 'None'}")
    if fks:
        print("  - Foreign keys:")
        for src_t, src_c, ref_t, ref_c in fks:
            print(f"      {src_t}.{src_c} -> {ref_t}.{ref_c}")
    else:
        print("  - Foreign keys: None inferred")
    print()

    conn = sqlite3.connect(str(db_path))
    try:
        create_tables(conn, tables, pks, fks)
        insert_all(conn, tables)

        # Row counts
        for t in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"Loaded table '{t}' | Rows: {count:,}")
        print()

        validate_foreign_keys(conn)

        print("✅ SUCCESS: Database created.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset-agnostic CSV → SQLite loader (Part 1).")
    parser.add_argument("--data", default="data", help="Folder containing CSV files")
    parser.add_argument("--db", default="db/app.sqlite", help="SQLite database output path")
    args = parser.parse_args()
    main(Path(args.data), Path(args.db))
