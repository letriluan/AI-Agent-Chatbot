"""Bounded SELECT execution with read-only SQLite permissions and an authorizer."""
import re
import sqlite3
import time

from .database import connect_readonly
from .privacy import _is_person_name_column, _is_pii_column

_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
_BANNED = re.compile(r"\b(drop|delete|update|insert|alter|truncate|attach|detach|pragma|reindex|vacuum|create|replace|load_extension)\b", re.I)


def is_safe_sql(sql: str) -> bool:
    s = (sql or "").strip().removesuffix(";").strip()
    return bool(s and len(s) <= 20_000 and ";" not in s and _READ_ONLY.match(s) and not _BANNED.search(s))


def _authorize(action, arg1, arg2, database, trigger):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    if action not in allowed:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and (arg1 or "").lower().startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    # Use the source column, not its model-selected alias, for access decisions.
    if action == sqlite3.SQLITE_READ and (_is_person_name_column(arg2 or "") or _is_pii_column(arg2 or "")):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def run_sql(db_path: str, sql: str, max_rows: int = 50, timeout_seconds: float = 5.0) -> dict:
    error = {"ok": False, "columns": [], "rows": []}
    if not is_safe_sql(sql):
        return {**error, "error": "Unsafe or non-read-only SQL blocked."}
    if max_rows < 1 or timeout_seconds <= 0:
        return {**error, "error": "Query limits must be positive."}
    conn = None
    started = time.monotonic()
    try:
        conn = connect_readonly(db_path)
        conn.set_authorizer(_authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout_seconds), 1000)
        # An outer bound also applies when the model supplied an inner LIMIT.
        query = (sql or "").strip().removesuffix(";").strip()
        cur = conn.execute(f"SELECT * FROM ({query}) AS bounded_result LIMIT ?", (int(max_rows),))
        rows = [list(row) for row in cur.fetchmany(max_rows)]
        return {"ok": True, "columns": [d[0] for d in cur.description], "rows": rows}
    except (sqlite3.Error, OSError) as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            message = f"Query timed out after {timeout_seconds:.1f}s."
        return {**error, "error": message}
    finally:
        if conn is not None:
            conn.close()
