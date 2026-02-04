from __future__ import annotations

"""
src/agent/schema.py

SQLite schema helpers.

- Reads tables/views and their columns at runtime
- Extracts foreign key relationships from SQLite metadata
- Formats schema text for prompt context
"""

import sqlite3
from typing import Dict, List, Tuple

Relationship = Tuple[str, str, str, str]


def _list_user_objects(conn: sqlite3.Connection) -> List[Tuple[str, str]]:

    """Return (name, type) for user-defined tables/views, excluding sqlite_* internals."""

    rows = conn.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table','view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name;
        """
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def get_schema(db_path: str) -> Dict[str, List[str]]:

    """
    Return a mapping: object_name -> list of column names.

    Includes tables and views (views are still useful for prompt context).
    """

    conn = sqlite3.connect(db_path)
    try:
        schema: Dict[str, List[str]] = {}
        for name, _typ in _list_user_objects(conn):
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            schema[name] = [str(c[1]) for c in cols]
        return schema
    finally:
        conn.close()


def get_relationships(db_path: str) -> List[Relationship]:
    """
    Return foreign key relationships from SQLite metadata.

    Note: FK metadata exists only if FK constraints were created when building the DB.
    """
    conn = sqlite3.connect(db_path)
    try:
        rels: List[Relationship] = []
        for name, typ in _list_user_objects(conn):
            if typ != "table":
                continue
            fks = conn.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
            for fk in fks:
                to_table = str(fk[2])
                from_col = str(fk[3])
                to_col = str(fk[4])
                rels.append((name, from_col, to_table, to_col))
        return rels
    finally:
        conn.close()

def format_schema(schema: Dict[str, List[str]], relationships: List[Relationship]) -> str:

    """Format schema + relationships into a prompt-friendly string."""
    
    lines: List[str] = []
    for table in sorted(schema.keys()):
        lines.append(f"{table}:")
        for c in schema[table]:
            lines.append(f"  - {c}")
        lines.append("")

    if relationships:
        lines.append("RELATIONSHIPS (preferred joins):")
        for ft, fc, tt, tc in relationships:
            lines.append(f"- {ft}.{fc} = {tt}.{tc}")
        lines.append("")

    return "\n".join(lines)
