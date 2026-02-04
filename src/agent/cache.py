# src/agent/cache.py
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Normalization + keys
# -----------------------------
_PUNCT = re.compile(r"[\t\n\r]+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


def norm_question(q: str) -> str:
    """Normalize user question to improve cache hit rate."""
    q = (q or "").strip().lower()
    q = _PUNCT.sub(" ", q)
    q = _NON_ALNUM.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def db_fingerprint(db_path: str) -> str:
    """Invalidate cache when DB file changes (mtime + size)."""
    try:
        st = os.stat(db_path)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except FileNotFoundError:
        return "missing-db"


def cache_key(question: str, db_fp: str, model_name: str) -> str:
    raw = f"{norm_question(question)}||{db_fp}||{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def q2sql_key(question: str, db_fp: str, model_name: str) -> str:
    raw = f"q2sql||{norm_question(question)}||{db_fp}||{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sql_key(sql: str, db_fp: str) -> str:
    raw = f"sql||{(sql or '').strip()}||{db_fp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# -----------------------------
# SQLite helpers
# -----------------------------
def _ensure_parent_dir(db_path: str) -> None:
    p = Path(db_path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    # Final answer cache (question -> answer)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_cache (
          cache_key TEXT PRIMARY KEY,
          question   TEXT NOT NULL,
          answer     TEXT NOT NULL,
          sql        TEXT,
          image_b64  TEXT,
          db_fingerprint TEXT NOT NULL,
          model_name TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_cache_created ON qa_cache(created_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_cache_model ON qa_cache(model_name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_cache_dbfp ON qa_cache(db_fingerprint);")

    # Question -> SQL cache (question -> sql)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS q2sql_cache (
          cache_key TEXT PRIMARY KEY,
          question  TEXT NOT NULL,
          sql       TEXT NOT NULL,
          db_fingerprint TEXT NOT NULL,
          model_name TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_q2sql_created ON q2sql_cache(created_at);")

    # SQL -> Result cache (sql -> result json)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sql_result_cache (
          cache_key TEXT PRIMARY KEY,
          sql       TEXT NOT NULL,
          result_json TEXT NOT NULL,
          db_fingerprint TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sqlres_created ON sql_result_cache(created_at);")
    conn.commit()


def _get_with_age(
    conn: sqlite3.Connection,
    query: str,
    params: Tuple[Any, ...],
    max_age_seconds: int,
) -> Optional[Tuple[Tuple[Any, ...], float]]:
    row = conn.execute(query, params).fetchone()
    if not row:
        return None
    *payload, created_at = row
    try:
        created = datetime.datetime.fromisoformat(created_at)
    except Exception:
        return None
    age = (datetime.datetime.utcnow() - created).total_seconds()
    if age > max_age_seconds:
        return None
    return tuple(payload), float(age)


# -----------------------------
# Final answer cache
# -----------------------------
def get_cached_answer(cache_db_path: str, key: str, max_age_seconds: int) -> Optional[Dict[str, Any]]:
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        got = _get_with_age(
            conn,
            "SELECT answer, sql, image_b64, created_at FROM qa_cache WHERE cache_key=?",
            (key,),
            max_age_seconds,
        )
        if not got:
            return None
        (answer, sql, image_b64), age = got
        return {"answer": answer, "sql": sql or "", "image_b64": image_b64, "age_seconds": age}
    finally:
        conn.close()


def set_cached_answer(
    cache_db_path: str,
    key: str,
    question: str,
    answer: str,
    sql: str,
    image_b64: Optional[str],
    db_fp: str,
    model_name: str,
) -> None:
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO qa_cache
            (cache_key, question, answer, sql, image_b64, db_fingerprint, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (key, question, answer, sql, image_b64, db_fp, model_name, _utcnow_iso()),
        )
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# Question -> SQL cache
# -----------------------------
def get_cached_sql(cache_db_path: str, key: str, max_age_seconds: int) -> Optional[Dict[str, Any]]:
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        got = _get_with_age(
            conn,
            "SELECT sql, created_at FROM q2sql_cache WHERE cache_key=?",
            (key,),
            max_age_seconds,
        )
        if not got:
            return None
        (sql,), age = got
        return {"sql": sql, "age_seconds": age}
    finally:
        conn.close()


def set_cached_sql(
    cache_db_path: str,
    key: str,
    question: str,
    sql: str,
    db_fp: str,
    model_name: str,
) -> None:
    if not sql:
        return
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO q2sql_cache
            (cache_key, question, sql, db_fingerprint, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (key, question, sql, db_fp, model_name, _utcnow_iso()),
        )
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# SQL -> Result cache
# -----------------------------
def get_cached_sql_result(cache_db_path: str, key: str, max_age_seconds: int) -> Optional[Dict[str, Any]]:
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        got = _get_with_age(
            conn,
            "SELECT result_json, created_at FROM sql_result_cache WHERE cache_key=?",
            (key,),
            max_age_seconds,
        )
        if not got:
            return None
        (result_json,), age = got
        try:
            result = json.loads(result_json)
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        result["age_seconds"] = age
        return result
    finally:
        conn.close()


def set_cached_sql_result(cache_db_path: str, key: str, sql: str, result: Dict[str, Any], db_fp: str) -> None:
    if not sql or not isinstance(result, dict):
        return
    _ensure_parent_dir(cache_db_path)
    conn = sqlite3.connect(cache_db_path)
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO sql_result_cache
            (cache_key, sql, result_json, db_fingerprint, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (key, sql, json.dumps(result, ensure_ascii=False), db_fp, _utcnow_iso()),
        )
        conn.commit()
    finally:
        conn.close()
