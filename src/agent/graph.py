from __future__ import annotations

"""
src/agent/graph.py

Features:
- Multi-level cache:
  (1) question -> final answer
  (2) question -> sql
  (3) sql -> result
- cache_hit info returned + ALSO shown inside the answer text (⚡ served from cache)
- schema cached via lru_cache
- vague question handling with suggestions
- optional explain metadata (only when user asks)
- plotting when user asks (chart/plot/graph)
"""

import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Literal
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from .schema import get_schema, get_relationships, format_schema
from .sql_tools import run_sql
from .privacy import redact_result, redact_text
from .prompts import SQL_SYSTEM, SQL_REPAIR_SYSTEM, ANSWER_SYSTEM, PLOT_SQL_SYSTEM
from .plot_tools import plot_line_png_base64, plot_bar_png_base64
from .cache import (
    db_fingerprint,
    cache_key,
    q2sql_key,
    sql_key,
    norm_question,
    get_cached_answer,
    set_cached_answer,
    get_cached_sql,
    set_cached_sql,
    get_cached_sql_result,
    set_cached_sql_result,
)

DB_PATH = os.environ.get("DB_PATH", "db/app.sqlite")
CACHE_DB = os.environ.get("CACHE_DB", "db/cache.sqlite")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

SQL_MAX_ROWS = int(os.environ.get("SQL_MAX_ROWS", "50"))
SQL_TIMEOUT_SECONDS = float(os.environ.get("SQL_TIMEOUT_SECONDS", "2.0"))


class State(TypedDict, total=False):
    question: str
    schema_text: str
    relationships: List[tuple[str, str, str, str]]
    sql: str
    result: Dict[str, Any]
    attempts: int
    want_plot: bool
    plot_sql: str
    plot_result: Dict[str, Any]
    image_b64: Optional[str]
    answer: str


def _make_llm():
    model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _expects_non_empty(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["top", "most", "highest", "lowest", "rank", "list", "show", "compare"])


def _wants_plot(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["plot", "chart", "graph", "visual", "trend", "line", "bar"])


def _wants_explain(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["explain", "show sql", "show the sql", "how did you", "why this sql", "reasoning"])


def _build_plot_image(plot_result: Dict[str, Any], question: str) -> Optional[str]:
    if not plot_result.get("ok"):
        return None

    cols = plot_result.get("columns", []) or []
    rows = plot_result.get("rows", []) or []
    if len(cols) < 2 or not rows:
        return None

    xs, ys = [], []
    for r in rows:
        try:
            xs.append(str(r[0]))
            ys.append(float(r[1]))
        except Exception:
            continue

    if not xs or not ys:
        return None

    q = question.lower()
    if "bar" in q:
        return plot_bar_png_base64(xs, ys, title="Comparison", xlabel=cols[0], ylabel=cols[1])
    if "line" in q or "trend" in q or "over time" in q:
        return plot_line_png_base64(xs, ys, title="Trend", xlabel=cols[0], ylabel=cols[1])

    if len(xs) > 12:
        return plot_line_png_base64(xs, ys, title="Trend", xlabel=cols[0], ylabel=cols[1])

    return plot_bar_png_base64(xs, ys, title="Comparison", xlabel=cols[0], ylabel=cols[1])


@lru_cache(maxsize=8)
def _cached_schema_text(db_path: str, db_fp: str) -> tuple[str, List[tuple[str, str, str, str]], Dict[str, List[str]]]:
    schema = get_schema(db_path)
    relationships = get_relationships(db_path)
    return format_schema(schema, relationships), relationships, schema


def _make_suggestions(schema: Dict[str, List[str]]) -> List[str]:
    tables = list(schema.keys())[:6]
    if not tables:
        return []
    out: List[str] = []
    t0 = tables[0]
    out.append(f"What are the top 5 rows in {t0}?")
    if len(tables) >= 2:
        out.append(f"How many records are in {tables[1]}?")
    if len(tables) >= 3:
        out.append(f"Show a breakdown (count) by a category column in {tables[2]}.")
    return out[:3]


def _is_too_vague(question: str, schema: Dict[str, List[str]]) -> bool:
    qn = norm_question(question)
    if len(qn.split()) >= 4:
        return False
    tokens = set(qn.split())
    for t, cols in schema.items():
        if t.lower() in tokens:
            return False
        for c in cols[:50]:
            if c.lower() in tokens:
                return False
    return True


def prepare_schema_node(state: State) -> State:
    db_fp = db_fingerprint(DB_PATH)
    schema_text, relationships, _schema = _cached_schema_text(DB_PATH, db_fp)
    state["schema_text"] = schema_text
    state["relationships"] = relationships
    state["attempts"] = 0
    return state


def generate_sql_node(state: State) -> State:
    llm = _make_llm()
    msg = [
        SystemMessage(content=SQL_SYSTEM),
        SystemMessage(content=f"DATABASE SCHEMA:\n{state['schema_text']}"),
        HumanMessage(content=state["question"]),
    ]
    out = llm.invoke(msg)
    state["sql"] = _strip_markdown_fences(getattr(out, "content", str(out)))
    return state


def repair_sql_node(state: State) -> State:
    llm = _make_llm()
    prev_sql = state.get("sql", "")
    err = state.get("result", {}).get("error", "")
    cols = state.get("result", {}).get("columns", [])
    rows = state.get("result", {}).get("rows", [])

    msg = [
        SystemMessage(content=SQL_REPAIR_SYSTEM),
        SystemMessage(content=f"DATABASE SCHEMA:\n{state['schema_text']}"),
        HumanMessage(
            content=(
                f"User question:\n{state['question']}\n\n"
                f"Previous SQL (failed or unhelpful):\n{prev_sql}\n\n"
                f"Error (if any):\n{err}\n\n"
                f"Result preview (cols/rows):\n{cols}\n{rows[:5] if isinstance(rows, list) else rows}\n\n"
                "Return corrected SQL only."
            )
        ),
    ]
    out = llm.invoke(msg)
    state["sql"] = _strip_markdown_fences(getattr(out, "content", str(out)))
    return state


def run_sql_node(state: State) -> State:
    result = run_sql(DB_PATH, state.get("sql", ""), max_rows=SQL_MAX_ROWS, timeout_seconds=SQL_TIMEOUT_SECONDS)
    state["result"] = redact_result(result)
    state["attempts"] = int(state.get("attempts", 0)) + 1
    return state


def decide_retry(state: State) -> Literal["retry", "answer"]:
    result = state.get("result", {})
    if not result.get("ok"):
        return "retry" if state.get("attempts", 0) < 3 else "answer"
    if _expects_non_empty(state["question"]) and len(result.get("rows", []) or []) == 0:
        return "retry" if state.get("attempts", 0) < 3 else "answer"
    return "answer"


def decide_plot_node(state: State) -> State:
    state["want_plot"] = _wants_plot(state["question"])
    return state


def plot_sql_node(state: State) -> State:
    if not state.get("want_plot"):
        return state

    llm = _make_llm()
    msg = [
        SystemMessage(content=PLOT_SQL_SYSTEM),
        SystemMessage(content=f"DATABASE SCHEMA:\n{state['schema_text']}"),
        HumanMessage(content=state["question"]),
    ]
    out = llm.invoke(msg)
    state["plot_sql"] = _strip_markdown_fences(getattr(out, "content", str(out)))
    state["plot_result"] = redact_result(
        run_sql(DB_PATH, state["plot_sql"], max_rows=200, timeout_seconds=SQL_TIMEOUT_SECONDS)
    )
    state["image_b64"] = _build_plot_image(state["plot_result"], state["question"])
    return state


def answer_node(state: State) -> State:
    llm = _make_llm()
    result = state.get("result", {})

    msg = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"QUESTION:\n{state['question']}\n\n"
                f"SQL:\n{state.get('sql','')}\n\n"
                f"RESULT:\ncolumns={result.get('columns', [])}\nrows={result.get('rows', [])}\n"
            )
        ),
    ]
    out = llm.invoke(msg)
    state["answer"] = redact_text(getattr(out, "content", str(out)))
    return state


def build_graph():
    g = StateGraph(State)
    g.add_node("prepare_schema", prepare_schema_node)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("run_sql", run_sql_node)
    g.add_node("repair_sql", repair_sql_node)
    g.add_node("decide_plot", decide_plot_node)
    g.add_node("plot_sql", plot_sql_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "prepare_schema")
    g.add_edge("prepare_schema", "generate_sql")
    g.add_edge("generate_sql", "run_sql")
    g.add_conditional_edges("run_sql", decide_retry, {"retry": "repair_sql", "answer": "decide_plot"})
    g.add_edge("repair_sql", "run_sql")
    g.add_edge("decide_plot", "plot_sql")
    g.add_edge("plot_sql", "answer")
    g.add_edge("answer", END)
    return g.compile()


_GRAPH = None


def ask(question: str) -> Dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    model_name = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    db_fp = db_fingerprint(DB_PATH)

    # ✅ ALWAYS define suggestions first (fixes NameError)
    _schema_text, _rels, schema_map = _cached_schema_text(DB_PATH, db_fp)
    suggestions = _make_suggestions(schema_map)

    want_explain = _wants_explain(question)

    # Vague question guard (no LLM call)
    if _is_too_vague(question, schema_map):
        return {
            "sql": "",
            "answer": "Can you add a little more detail (which table or metric)? Try asking about totals, counts, top items, or include a table name.",
            "image_b64": None,
            "cache_hit": False,
            "cache_source": "none",
            "cache_age_seconds": 0.0,
            "suggestions": suggestions,
            "explain": {
                "note": "No SQL executed because the question was too vague.",
                "limits": {"max_rows": SQL_MAX_ROWS, "timeout_seconds": SQL_TIMEOUT_SECONDS, "read_only": True},
            }
            if want_explain
            else None,
        }

    # 1) Final answer cache
    key = cache_key(question, db_fp, model_name)
    hit = get_cached_answer(CACHE_DB, key, max_age_seconds=CACHE_TTL_SECONDS)
    if hit:
        answer_text = hit.get("answer", "")
        answer_text = "⚡ *This answer was served from cache.*\n\n" + answer_text

        resp = {
            "sql": hit.get("sql", ""),
            "answer": answer_text,
            "image_b64": hit.get("image_b64"),
            "cache_hit": True,
            "cache_source": "qa",
            "cache_age_seconds": float(hit.get("age_seconds", 0.0) or 0.0),
            "suggestions": suggestions,
        }
        if want_explain:
            resp["explain"] = {
                "source": "qa_cache",
                "normalized_question": norm_question(question),
                "limits": {"max_rows": SQL_MAX_ROWS, "timeout_seconds": SQL_TIMEOUT_SECONDS, "read_only": True},
            }
        return resp

    # 2) Question -> SQL cache
    qsql_k = q2sql_key(question, db_fp, model_name)
    qsql_hit = get_cached_sql(CACHE_DB, qsql_k, max_age_seconds=CACHE_TTL_SECONDS)
    if qsql_hit:
        sql = qsql_hit["sql"]

        # 3) SQL -> Result cache
        sk = sql_key(sql, db_fp)
        r_hit = get_cached_sql_result(CACHE_DB, sk, max_age_seconds=CACHE_TTL_SECONDS)
        if r_hit:
            result = r_hit
            cache_source = "sql_result"
            cache_age = float(r_hit.get("age_seconds", 0.0) or 0.0)
        else:
            result = redact_result(run_sql(DB_PATH, sql, max_rows=SQL_MAX_ROWS, timeout_seconds=SQL_TIMEOUT_SECONDS))
            set_cached_sql_result(CACHE_DB, sk, sql=sql, result=result, db_fp=db_fp)
            cache_source = "q2sql"
            cache_age = float(qsql_hit.get("age_seconds", 0.0) or 0.0)

        # Generate answer (1 LLM call)
        llm = _make_llm()
        msg = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"QUESTION:\n{question}\n\n"
                    f"SQL:\n{sql}\n\n"
                    f"RESULT:\ncolumns={result.get('columns', [])}\nrows={result.get('rows', [])}\n"
                )
            ),
        ]
        out = llm.invoke(msg)
        answer = redact_text(getattr(out, "content", str(out)))

        answer = "⚡ *This answer was served from cache.*\n\n" + answer

        resp = {
            "sql": sql,
            "answer": answer,
            "image_b64": None,
            "cache_hit": True,
            "cache_source": cache_source,
            "cache_age_seconds": cache_age,
            "suggestions": suggestions,
        }

        if want_explain:
            resp["explain"] = {
                "source": cache_source,
                "normalized_question": norm_question(question),
                "sql_from_cache": True,
                "limits": {"max_rows": SQL_MAX_ROWS, "timeout_seconds": SQL_TIMEOUT_SECONDS, "read_only": True},
            }

        # Save final answer too
        if answer:
            set_cached_answer(
                CACHE_DB,
                key=key,
                question=question,
                answer=answer.replace("⚡ *This answer was served from cache.*\n\n", "", 1),
                sql=sql,
                image_b64=None,
                db_fp=db_fp,
                model_name=model_name,
            )

        return resp

    # 4) Normal graph execution
    out_state: State = _GRAPH.invoke(
        {"question": question},
        config={"run_name": "delivery-cadet-ask", "tags": ["delivery-cadet", "langgraph", "sql-agent"]},
    )

    resp = {
        "sql": out_state.get("sql", ""),
        "answer": out_state.get("answer", ""),
        "image_b64": out_state.get("image_b64"),
        "cache_hit": False,
        "cache_source": "none",
        "cache_age_seconds": 0.0,
        "suggestions": suggestions,
    }

    # Save caches
    sql_out = resp["sql"]
    if sql_out:
        set_cached_sql(CACHE_DB, qsql_k, question=question, sql=sql_out, db_fp=db_fp, model_name=model_name)

        if isinstance(out_state.get("result"), dict) and out_state["result"].get("ok"):
            sk = sql_key(sql_out, db_fp)
            set_cached_sql_result(CACHE_DB, sk, sql=sql_out, result=out_state["result"], db_fp=db_fp)

    if resp["answer"]:
        set_cached_answer(
            CACHE_DB,
            key=key,
            question=question,
            answer=resp["answer"],
            sql=resp["sql"],
            image_b64=resp["image_b64"],
            db_fp=db_fp,
            model_name=model_name,
        )

    if want_explain:
        resp["explain"] = {
            "source": "live_run",
            "normalized_question": norm_question(question),
            "limits": {"max_rows": SQL_MAX_ROWS, "timeout_seconds": SQL_TIMEOUT_SECONDS, "read_only": True},
        }

    return resp
