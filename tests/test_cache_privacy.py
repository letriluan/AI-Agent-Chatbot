import sqlite3
from src.agent import cache
from src.agent.privacy import redact_result


def test_distinct_numbers_and_unicode():
    assert cache.norm_question("amount 1.5") != cache.norm_question("amount 15")
    assert cache.norm_question("amount -5") != cache.norm_question("amount 5")
    assert cache.norm_question("đơn hàng") != cache.norm_question("hàng")


def test_database_fingerprint_includes_path(db, tmp_path):
    p = tmp_path / "second.sqlite"
    from pathlib import Path
    p.write_bytes(Path(db).read_bytes())
    assert cache.db_fingerprint(db) != cache.db_fingerprint(str(p))


def test_cache_expiration(tmp_path):
    db = str(tmp_path / "cache.sqlite")
    cache.set_cached_answer(db, "k", "question", "answer", "SELECT 1", None, "fp", "model")
    assert cache.get_cached_answer(db, "k", 100)["answer"] == "answer"
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE qa_cache SET created_at='2000-01-01T00:00:00'")
        conn.commit()
    finally:
        conn.close()
    assert cache.get_cached_answer(db, "k", 1) is None


def test_privacy_redacts_customer_fields_keeps_business_fields():
    r = redact_result({"ok": True, "columns": ["customer_name", "email", "supplier_name", "total"], "rows": [["Alice", "alice@example.com", "Acme", 10]]})
    assert r["rows"] == [["[REDACTED_NAME]", "[REDACTED_PII]", "Acme", 10]]
