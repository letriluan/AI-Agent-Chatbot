"""Read-only database access; cache and ingestion use separate write connections."""
from pathlib import Path
import sqlite3


def connect_readonly(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).resolve(strict=True)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1.0)
    conn.execute("PRAGMA query_only=ON")
    return conn
