import pytest
from evals.run import cases, equivalent
from src.agent.sql_tools import run_sql


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_reference_cases(db, case):
    result = run_sql(db, case["reference_sql"])
    assert result["ok"]
    assert equivalent(result["rows"], case["expected_rows"], case.get("ordered", True))


def test_comparator_preserves_duplicates_and_order():
    assert not equivalent([[1], [1]], [[1]], ordered=False)
    assert not equivalent([[1], [2]], [[2], [1]])


def test_selected_offline_case_is_not_model_accuracy():
    from evals.run import run
    report = run("offline", selected_ids=["q01"])
    assert report["case_count"] == report["reference_passed"] == 1
    assert report["agent_quality_measured"] is False


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_graph_against_reference_with_scripted_llm(db, tmp_path, monkeypatch, case):
    """Tests orchestration only; scripted SQL is not model-quality evidence."""
    from types import SimpleNamespace
    from src.agent import graph
    monkeypatch.setattr(graph, "DB_PATH", db)
    monkeypatch.setattr(graph, "CACHE_DB", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(graph, "_GRAPH", None)
    outputs = iter([case["reference_sql"], "Answer from the supplied SQL result."])
    class Scripted:
        def invoke(self, messages):
            return SimpleNamespace(content=next(outputs))
    monkeypatch.setattr(graph, "_make_llm", Scripted)
    result = graph.ask(case["question"])
    assert result["ok"] and result["attempts"] == 1
    actual = run_sql(db, result["sql"])
    assert equivalent(actual["rows"], case["expected_rows"])
