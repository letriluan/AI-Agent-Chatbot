from __future__ import annotations

"""
src/agent/sql_tools.py

Safe SQL execution for the agent.

- Read-only queries only (SELECT / WITH)
- Blocks dangerous keywords
- Blocks multi-statement queries
- Adds a LIMIT if missing
- Adds a time limit via SQLite progress handler
- Returns a consistent JSON-friendly structure
"""

import re
import sqlite3
import time
from typing import Any, Dict, List

_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

_BANNED = re.compile(
    r"\b(drop|delete|update|insert|alter|truncate|attach|pragma|reindex|vacuum|create|replace)\b",
    re.IGNORECASE,
)

_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


def is_safe_sql(sql: str) -> bool:
    """Return True only for a single, read-only SELECT/WITH query."""
    s = (sql or "").strip()

    # Block empty
    if not s:
        return False

    # Block multi-statement: any semicolon after stripping trailing semicolons
    s_no_trail = s.rstrip().rstrip(";")
    if ";" in s_no_trail:
        return False

    if not _READ_ONLY.match(s_no_trail):
        return False
    if _BANNED.search(s_no_trail):
        return False

    # Hard length guard (avoid huge prompt-generated SQL)
    if len(s_no_trail) > 20_000:
        return False

    return True


def _enforce_limit(sql: str, max_rows: int) -> str:
    s = (sql or "").strip().rstrip(";")
    if _HAS_LIMIT.search(s):
        return s
    return f"{s} LIMIT {int(max_rows)}"


def run_sql(db_path: str, sql: str, max_rows: int = 50, timeout_seconds: float = 2.0) -> Dict[str, Any]:
    """
    Executes read-only SQL and returns:
      { ok: bool, columns: [...], rows: [...], error?: str }
    """
    sql = (sql or "").strip()

    if not is_safe_sql(sql):
        return {"ok": False, "columns": [], "rows": [], "error": "Unsafe or non-read-only SQL blocked."}

    sql = _enforce_limit(sql, max_rows=max_rows)

    conn = sqlite3.connect(db_path)
    start = time.monotonic()

    def _progress_handler() -> int:
        # Return 1 to interrupt
        if (time.monotonic() - start) > timeout_seconds:
            return 1
        return 0

    try:
        conn.row_factory = sqlite3.Row
        # Call progress handler periodically (every N VM instructions)
        conn.set_progress_handler(_progress_handler, 10_000)

        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        columns = [d[0] for d in cur.description] if cur.description else []
        out_rows: List[List[Any]] = []
        for r in rows:
            out_rows.append([r[c] for c in columns])
        return {"ok": True, "columns": columns, "rows": out_rows}
    except Exception as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            msg = f"Query timed out after {timeout_seconds:.1f}s."
        return {"ok": False, "columns": [], "rows": [], "error": msg}
    finally:
        try:
            conn.set_progress_handler(None, 0)
        except Exception:
            pass
        conn.close()
