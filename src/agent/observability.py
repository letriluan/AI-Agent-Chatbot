"""Request-local measurements. Logs deliberately exclude prompts, SQL and rows."""
from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
import time
import uuid

from groq import APIConnectionError

_metrics = ContextVar("agent_metrics", default=None)
logger = logging.getLogger("agent.metrics")


def connection_retry_limit():
    """A request-wide cap, with no SDK retries or alternate provider."""
    limit = int(os.getenv('LLM_CONNECTION_RETRIES', '2'))
    if not 0 <= limit <= 2:
        raise ValueError('LLM_CONNECTION_RETRIES must be between 0 and 2')
    return limit


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
    retries = 0
    limit = connection_retry_limit()
    while True:
        if metrics is not None:
            metrics['llm_calls'] += 1
        try:
            result = llm.invoke(messages)
            break
        except Exception as exc:
            if metrics is not None:
                metrics['llm_failed_calls'] = metrics.get('llm_failed_calls', 0) + 1
                # The server may have processed a request whose response was lost.
                metrics['token_usage_complete'] = False
            used = metrics.get('connection_retries', 0) if metrics is not None else retries
            if not isinstance(exc, APIConnectionError) or used >= limit:
                raise
            # Rate limits and HTTP errors are intentionally excluded. A shared
            # budget prevents every graph node from opening its own retry loop.
            if metrics is not None:
                metrics['connection_retries'] = used + 1
            retries += 1
            delay = 2 ** used
            if metrics is not None:
                metrics['retry_backoff_ms'] = metrics.get('retry_backoff_ms', 0) + delay * 1000
            time.sleep(delay)
    usage = getattr(result, "usage_metadata", None) or {}
    if metrics is not None:
        complete = all(isinstance(usage.get(name), int) and not isinstance(usage[name], bool)
                       and usage[name] >= 0 for name in ("input_tokens", "output_tokens"))
        metrics["token_usage_complete"] = metrics.get("token_usage_complete", True) and complete
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            metrics[name] += int(usage.get(name, 0) or 0)
    return result


def observed(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        metrics = dict(llm_calls=0, input_tokens=0, output_tokens=0, total_tokens=0,
                       token_usage_complete=True, llm_failed_calls=0,
                       connection_retries=0, retry_backoff_ms=0)
        token = _metrics.set(metrics)
        try:
            result = fn(*args, **kwargs)
            metrics["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["metrics"] = metrics
            result["request_id"] = request_id
            logger.info(json.dumps({"event": "agent_request", "request_id": request_id, "cache_source": result.get("cache_source"), "ok": result.get("ok", True), **metrics}))
            return result
        except Exception as exc:
            metrics['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
            # Retain partial counters without exposing exception text, inputs,
            # provider headers, credentials, SQL or result rows.
            try:
                exc.agent_metrics = dict(metrics)
                exc.agent_request_id = request_id
            except (AttributeError, TypeError):
                pass
            logger.warning(json.dumps({"event": "agent_failure", "request_id": request_id,
                                       "error_type": type(exc).__name__, **metrics}))
            raise
        finally:
            _metrics.reset(token)
    return wrapped
