"""Request-local measurements. Logs deliberately exclude prompts, SQL and rows."""
from contextvars import ContextVar
from functools import wraps
import json
import logging
import time
import uuid

_metrics = ContextVar("agent_metrics", default=None)
logger = logging.getLogger("agent.metrics")


def measured(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        metrics = _metrics.get()
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if metrics is not None:
                stages = metrics.setdefault("stages_ms", {})
                stages[fn.__name__] = stages.get(fn.__name__, 0) + round(
                    (time.perf_counter() - started) * 1000, 3
                )
    return wrapped


def invoke_llm(llm, messages):
    metrics = _metrics.get()
    if metrics is not None:
        metrics["llm_calls"] += 1
    result = llm.invoke(messages)
    usage = getattr(result, "usage_metadata", None) or {}
    if metrics is not None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            metrics[name] += int(usage.get(name, 0) or 0)
    return result


def observed(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        metrics = dict(llm_calls=0, input_tokens=0, output_tokens=0, total_tokens=0)
        token = _metrics.set(metrics)
        try:
            result = fn(*args, **kwargs)
            metrics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["metrics"] = metrics
            result["request_id"] = request_id
            logger.info(json.dumps({"event": "agent_request", "request_id": request_id, "cache_source": result.get("cache_source"), "ok": result.get("ok", True), **metrics}))
            return result
        except Exception as exc:
            logger.warning(json.dumps({"event": "agent_failure", "request_id": request_id, "error_type": type(exc).__name__, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}))
            raise
        finally:
            _metrics.reset(token)
    return wrapped
