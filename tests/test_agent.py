import base64
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from src.agent import graph
from src.server.app import app


@pytest.fixture
def agent(db, tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "DB_PATH", db)
    monkeypatch.setattr(graph, "CACHE_DB", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(graph, "_GRAPH", None)
    return graph


def fake(agent, monkeypatch, outputs):
    items = iter(outputs)
    calls = []
    class LLM:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content=next(items), usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
    monkeypatch.setattr(agent, "_make_llm", LLM)
    return calls


def test_repair_chart_and_cache(agent, monkeypatch):
    calls = fake(agent, monkeypatch, ["SELECT missing FROM orders", "SELECT status, COUNT(*) FROM orders GROUP BY status", "SELECT status, COUNT(*) FROM orders GROUP BY status", "Four orders are delivered."])
    a = agent.ask("Plot number of orders by status")
    assert a["ok"] and a["attempts"] == 2
    assert base64.b64decode(a["image_b64"]).startswith(b"\x89PNG")
    assert a["metrics"]["llm_calls"] == 4 and a["metrics"]["total_tokens"] == 20
    b = agent.ask("Plot number of orders by status")
    assert b["cache_hit"] and b["image_b64"] == a["image_b64"]
    assert b["metrics"]["llm_calls"] == 0 and len(calls) == 4


@pytest.mark.parametrize("question", ["How many orders are there?", "Plot number of orders by status"])
def test_failures_are_not_cached_or_summarised(agent, monkeypatch, question):
    calls = fake(agent, monkeypatch, ["SELECT invalid FROM orders"] * 6)
    for _ in range(2):
        r = agent.ask(question)
        assert not r["ok"] and r["attempts"] == 3
        assert "could not execute" in r["answer"]
    assert len(calls) == 6


def test_vague_question_does_not_call_llm(agent, monkeypatch):
    calls = fake(agent, monkeypatch, [])
    assert not agent.ask("hello")["sql"]
    assert not calls


def test_api_validation_and_sanitized_errors(agent, monkeypatch):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/").status_code == 200
        for q in ["", "   ", "x" * 2001]:
            assert client.post("/ask", json={"question": q}).status_code == 422
        def broken(_):
            raise RuntimeError("secret-key-in-provider-error")
        monkeypatch.setattr("src.server.app.ask", broken)
        r = client.post("/ask", json={"question": "How many orders are there?"})
        assert r.status_code == 502 and "secret-key" not in r.text


def test_expired_answer_chart_is_rebuilt(agent, monkeypatch):
    from src.agent import cache
    calls = fake(agent, monkeypatch, ["SELECT status, COUNT(*) FROM orders GROUP BY status", "SELECT status, COUNT(*) FROM orders GROUP BY status", "Four delivered."] * 2)
    a = agent.ask("Plot number of orders by status")
    import sqlite3
    conn = sqlite3.connect(agent.CACHE_DB)
    try:
        conn.execute("DELETE FROM qa_cache")
        conn.commit()
    finally:
        conn.close()
    b = agent.ask("Plot number of orders by status")
    assert a["image_b64"] and b["image_b64"] and len(calls) == 6
    assert cache.get_cached_answer(agent.CACHE_DB, "nonexistent", 10) is None


def test_failed_cached_sql_recovers(agent, monkeypatch):
    import hashlib
    from pathlib import Path
    from src.agent import cache
    import os
    q = "How many orders are there?"
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b") + "|v2|" + hashlib.sha256(Path(agent.__file__).with_name("prompts.py").read_bytes()).hexdigest()[:16]
    fp = cache.db_fingerprint(agent.DB_PATH)
    key = cache.q2sql_key(q, fp, model)
    cache.set_cached_sql(agent.CACHE_DB, key, q, "SELECT missing FROM orders", fp, model)
    calls = fake(agent, monkeypatch, ["SELECT COUNT(*) FROM orders", "There are six orders."])
    result = agent.ask(q)
    assert result["ok"] and "six" in result["answer"] and len(calls) == 2


def test_metrics_log_omits_question_sql_and_rows(agent, monkeypatch, caplog):
    fake(agent, monkeypatch, ["SELECT COUNT(*) FROM orders", "There are six orders."])
    with caplog.at_level("INFO", logger="agent.metrics"):
        result = agent.ask("How many orders exist, private-marker-123?")
    assert result["request_id"] in caplog.text
    assert "private-marker-123" not in caplog.text
    assert "SELECT COUNT" not in caplog.text
    assert "There are six" not in caplog.text


def test_grounded_code_answer_survives_query_cache(agent, monkeypatch):
    import sqlite3
    calls = fake(agent, monkeypatch, ["SELECT SUM(o.amount) AS total_amount FROM orders o JOIN customers c ON c.id=o.customer_id WHERE c.state='WA'"])
    q = "What is the total amount for WA customers?"
    first = agent.ask(q)
    assert "WA" in first["answer"] and "Washington" not in first["answer"]
    assert "$" not in first["answer"] and len(calls) == 1
    conn = sqlite3.connect(agent.CACHE_DB)
    try:
        conn.execute("DELETE FROM qa_cache")
        conn.commit()
    finally:
        conn.close()
    second = agent.ask(q)
    assert "WA" in second["answer"] and "Washington" not in second["answer"]
    assert "$" not in second["answer"] and len(calls) == 1


def test_unsupported_currency_removed_from_live_and_cached_answer(agent, monkeypatch):
    calls = fake(agent, monkeypatch, ["SELECT AVG(amount) AS average_amount FROM orders", "The average is $35.00 USD."])
    q = "What is the average order amount?"
    for _ in range(2):
        answer = agent.ask(q)["answer"]
        assert "35.00" in answer and "$" not in answer and "USD" not in answer
    assert len(calls) == 2


def test_month_line_image_survives_answer_cache(agent, monkeypatch):
    sql = "SELECT strftime('%Y-%m',ordered_at) AS month,SUM(amount) AS total_amount FROM orders GROUP BY 1 ORDER BY 1"
    calls = fake(agent, monkeypatch, [sql, sql, "Monthly totals are 30, 70 and 110."])
    question = "Plot monthly total amount as a line chart"
    first = agent.ask(question)
    second = agent.ask(question)
    assert base64.b64decode(first["image_b64"]).startswith(b"\x89PNG")
    assert second["image_b64"] == first["image_b64"] and second["cache_hit"]
    assert second["metrics"]["llm_calls"] == 0 and len(calls) == 3
