import sqlite3
import pytest
from src.agent.database import connect_readonly
from src.agent.sql_tools import run_sql


@pytest.mark.parametrize("sql", ["DROP TABLE orders", "DELETE FROM orders", "ATTACH DATABASE 'x' AS x", "PRAGMA user_version", "SELECT 1; SELECT 2", "WITH x AS (SELECT 1) DELETE FROM orders", "SELECT load_extension('x')"])
def test_denies_unsafe_queries(db, sql):
    assert not run_sql(db, sql)["ok"]
    assert run_sql(db, "SELECT COUNT(*) FROM orders")["rows"] == [[6]]


def test_permissions_are_enforced_without_regex(db):
    conn = connect_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM orders")
    finally:
        conn.close()


def test_missing_database_is_not_created(tmp_path):
    p = tmp_path / "missing.sqlite"
    assert not run_sql(str(p), "SELECT 1")["ok"]
    assert not p.exists()


def test_limits_existing_limit_and_cte(db):
    assert len(run_sql(db, "SELECT * FROM orders LIMIT 999", max_rows=2)["rows"]) == 2
    assert run_sql(db, "WITH x AS (SELECT amount FROM orders) SELECT SUM(amount) FROM x;")["rows"] == [[210.0]]


def test_timeout(db):
    result = run_sql(db, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<100000000) SELECT SUM(n) FROM x", timeout_seconds=0.001)
    assert not result["ok"] and "timed out" in result["error"]


def test_schema_internals_blocked(db):
    assert not run_sql(db, "SELECT * FROM sqlite_master")["ok"]


def test_sensitive_column_cannot_be_exposed_with_alias(db):
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE contacts (email TEXT)")
        conn.execute("INSERT INTO contacts VALUES ('person@example.com')")
        conn.commit()
    finally:
        conn.close()
    assert not run_sql(db, "SELECT email AS harmless FROM contacts")["ok"]
    assert not run_sql(db, "SELECT substr(email,1) AS harmless FROM contacts")["ok"]
    assert run_sql(db, "SELECT COUNT(*) FROM contacts")["rows"] == [[1]]
