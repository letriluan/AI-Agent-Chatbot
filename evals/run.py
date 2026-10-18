"""Offline validates reference SQL; live additionally measures the actual agent."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import math
import tempfile
import time
import hashlib

from .fixture import create_database
from src.agent.sql_tools import run_sql


def cases():
    return json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))


def equivalent(actual, expected, ordered=True):
    def row(r):
        return tuple(round(x, 6) if isinstance(x, float) else x for x in r)
    a, b = [row(r) for r in actual], [row(r) for r in expected]
    return a == b if ordered else Counter(a) == Counter(b)


@contextmanager
def configured_graph(db, cache):
    from src.agent import graph
    previous = graph.DB_PATH, graph.CACHE_DB, graph._GRAPH
    graph.DB_PATH, graph.CACHE_DB, graph._GRAPH = str(db), str(cache), None
    try:
        yield graph
    finally:
        graph.DB_PATH, graph.CACHE_DB, graph._GRAPH = previous


def run(mode="offline", delay_seconds=10.0, selected_ids=None):
    items = [item for item in cases() if not selected_ids or item["id"] in selected_ids]
    if not items:
        raise ValueError("No matching evaluation cases.")
    records = []
    if mode == "live":
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GROQ_API_KEY", "")
        if not key or "YOUR" in key.upper() or "your_" in key.lower():
            raise ValueError("Live evaluation requires a valid local GROQ_API_KEY.")
    with tempfile.TemporaryDirectory() as temp:
        db = Path(temp) / "fixture.sqlite"
        create_database(db)
        with configured_graph(db, Path(temp) / "cache.sqlite") as graph:
            for item in items:
                if mode == "live" and records and delay_seconds > 0:
                    time.sleep(delay_seconds)
                expected = run_sql(str(db), item["reference_sql"])
                valid = expected["ok"] and equivalent(expected["rows"], item["expected_rows"], item.get("ordered", True))
                record = {"id": item["id"], "reference_valid": valid}
                if not valid:
                    raise AssertionError(f"Reference fixture mismatch: {item['id']}")
                if mode == "live":
                    start = time.perf_counter()
                    try:
                        response = graph.ask(item["question"])
                        result = run_sql(str(db), response.get("sql", ""))
                        record.update(correct=bool(result["ok"] and equivalent(result["rows"], expected["rows"], item.get("ordered", True))), attempts=response.get("attempts", 0), metrics=response.get("metrics", {}))
                        record["generated_sql"] = response.get("sql", "")
                        record["actual_rows"] = result.get("rows", [])
                        record["question"] = item["question"]
                        record["reference_sql"] = item["reference_sql"]
                        record["expected_rows"] = expected["rows"]
                        record["answer"] = response.get("answer", "")
                        cached = graph.ask(item["question"])
                        record["cached_latency_ms"] = cached["metrics"]["latency_ms"]
                        record["cache_hit"] = cached.get("cache_hit", False)
                    except Exception as exc:
                        record.update(correct=False, error_type=type(exc).__name__)
                    record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
                records.append(record)
                if mode == "live":
                    print(json.dumps({"case": item["id"], "correct": record.get("correct"), "error_type": record.get("error_type")}), flush=True)
    report = {"mode": mode, "case_count": len(records), "reference_passed": sum(x["reference_valid"] for x in records), "agent_quality_measured": mode == "live", "records": records}
    if mode == "live":
        report["model"] = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        report["prompt_sha256"] = hashlib.sha256(Path(graph.__file__).with_name("prompts.py").read_bytes()).hexdigest()
        latencies = [r["metrics"]["latency_ms"] for r in records if "metrics" in r]
        repair = [r for r in records if r.get("attempts", 0) > 1]
        report.update(execution_accuracy=sum(r.get("correct", False) for r in records) / len(records), repair_cases=len(repair), repair_success_rate=sum(r["correct"] for r in repair) / len(repair) if repair else None, median_cold_latency_ms=statistics.median(latencies) if latencies else None)
        cached = [r["cached_latency_ms"] for r in records if "cached_latency_ms" in r]
        report.update(
            first_attempt_accuracy=sum(r.get("correct", False) and r.get("attempts") == 1 for r in records) / len(records),
            p95_cold_latency_ms=sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            median_cached_latency_ms=statistics.median(cached) if cached else None,
            cache_hits=sum(r.get("cache_hit", False) for r in records),
            monetary_cost_measured=False,
        )
        completed = [r for r in records if "error_type" not in r]
        report["provider_error_cases"] = sum("error_type" in r for r in records)
        report["execution_accuracy_on_completed"] = sum(r.get("correct", False) for r in completed) / len(completed) if completed else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--delay-seconds", type=float, default=10.0, help="Pause between live cases to reduce rate limits")
    parser.add_argument("--case", action="append", dest="selected_ids", help="Run a selected case ID; repeat for more cases")
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be nonnegative")
    report = run(args.mode, args.delay_seconds, args.selected_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
