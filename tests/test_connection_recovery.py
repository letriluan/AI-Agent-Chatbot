"""Fault injection tests; no live service or credentials are used."""
from types import SimpleNamespace

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, APIStatusError

from src.agent import observability as obs


def connection_error():
    return APIConnectionError(request=httpx.Request('POST', 'https://example.test'))


def client(outcomes, calls):
    outputs = iter(outcomes)
    def invoke(messages):
        calls.append(messages)
        output = next(outputs)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(content=output, usage_metadata={'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8})
    return SimpleNamespace(invoke=invoke)


@pytest.mark.parametrize('error', [connection_error(), APITimeoutError(request=httpx.Request('POST', 'https://example.test'))])
def test_connection_and_timeout_recover_once_with_visible_usage_gap(monkeypatch, error):
    monkeypatch.setenv('LLM_CONNECTION_RETRIES', '2')
    delays, calls = [], []
    monkeypatch.setattr(obs.time, 'sleep', delays.append)
    llm = client([error, 'success'], calls)
    @obs.observed
    def ask():
        return {'answer': obs.invoke_llm(llm, ['unchanged question']).content}
    result = ask()
    assert result['answer'] == 'success'
    assert calls == [['unchanged question'], ['unchanged question']]
    assert delays == [1]
    assert result['metrics']['llm_calls'] == 2
    assert result['metrics']['llm_failed_calls'] == result['metrics']['connection_retries'] == 1
    assert result['metrics']['total_tokens'] == 8
    assert not result['metrics']['token_usage_complete']


def test_retry_budget_is_shared_across_graph_nodes_and_partial_metrics_survive(monkeypatch):
    monkeypatch.setenv('LLM_CONNECTION_RETRIES', '2')
    delays, calls = [], []
    monkeypatch.setattr(obs.time, 'sleep', delays.append)
    first = client([connection_error(), 'sql'], calls)
    second = client([connection_error(), connection_error()], calls)
    @obs.observed
    def ask():
        obs.invoke_llm(first, ['sql question'])
        obs.invoke_llm(second, ['answer question'])
        return {}
    with pytest.raises(APIConnectionError) as captured:
        ask()
    metrics = captured.value.agent_metrics
    assert delays == [1, 2] and len(calls) == 4
    assert metrics['connection_retries'] == 2 and metrics['llm_failed_calls'] == 3
    assert metrics['total_tokens'] == 8 and not metrics['token_usage_complete']


@pytest.mark.parametrize('status', [400,401,402,403,404,429,500,503])
def test_http_errors_never_retry_or_fall_back(monkeypatch, status):
    delays, calls = [], []
    monkeypatch.setattr(obs.time, 'sleep', delays.append)
    request = httpx.Request('POST', 'https://example.test')
    error = APIStatusError('private-error-do-not-log', response=httpx.Response(status, request=request), body=None)
    llm = client([error], calls)
    @obs.observed
    def ask():
        obs.invoke_llm(llm, ['question'])
        return {}
    with pytest.raises(APIStatusError) as captured:
        ask()
    assert len(calls) == 1 and delays == []
    assert captured.value.agent_metrics['llm_failed_calls'] == 1


def test_failures_and_retries_do_not_log_sensitive_text(monkeypatch, caplog):
    monkeypatch.setenv('LLM_CONNECTION_RETRIES', '0')
    calls = []
    llm = client([connection_error()], calls)
    @obs.observed
    def ask():
        obs.invoke_llm(llm, ['private-question-123'])
        return {}
    with caplog.at_level('WARNING', logger='agent.metrics'), pytest.raises(APIConnectionError):
        ask()
    assert 'private-question-123' not in caplog.text
    assert 'example.test' not in caplog.text
    assert 'llm_failed_calls' in caplog.text


@pytest.mark.parametrize('value', ['-1','3','invalid'])
def test_invalid_retry_settings_fail_before_calling_provider(monkeypatch, value):
    monkeypatch.setenv('LLM_CONNECTION_RETRIES', value)
    calls = []
    with pytest.raises(ValueError):
        obs.invoke_llm(client(['unused'],calls), [])
    assert calls == []


def test_exhausted_connection_failure_returns_sanitised_service_unavailable(monkeypatch):
    from fastapi.testclient import TestClient
    from src.server.app import app
    error = connection_error()
    error.agent_request_id = 'safe-correlation-id'
    def fail(_):
        raise error
    monkeypatch.setattr('src.server.app.ask', fail)
    with TestClient(app) as client:
        response = client.post('/ask', json={'question':'How many orders are there?'})
    assert response.status_code == 503
    assert response.headers['Retry-After'] == '5'
    assert response.json()['detail']['request_id'] == 'safe-correlation-id'
    assert 'example.test' not in response.text
