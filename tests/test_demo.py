"""Demo limits must reject before provider work and survive guard recreation."""
import importlib
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from src.server.demo import DemoGuard


@pytest.fixture
def guard(tmp_path):
    return DemoGuard(tmp_path / 'usage.sqlite', per_client=2, clock=lambda: 1800000000)


def reserve(guard, peer='peer'):
    with guard.admit(peer):
        return True


def test_client_limit_and_no_refund_for_failure(guard):
    reserve(guard)
    with pytest.raises(RuntimeError):
        with guard.admit('peer'):
            raise RuntimeError('provider failed')
    with pytest.raises(HTTPException) as error:
        reserve(guard)
    assert error.value.status_code == 429
    assert error.value.headers['Retry-After'] == '60'
    assert reserve(guard, 'other')


def test_no_global_total_across_visitors(guard):
    for index in range(100):
        assert reserve(guard, f'visitor-{index}')


def test_no_hourly_or_daily_total(tmp_path):
    now = [1800000000.0]
    guard = DemoGuard(tmp_path / 'usage.sqlite', 5, clock=lambda: now[0])
    for _ in range(100):
        assert reserve(guard)
        now[0] += 61



def test_quota_survives_recreation(guard):
    reserve(guard)
    reserve(guard)
    restarted = DemoGuard(guard.path, 2, clock=guard.clock)
    with pytest.raises(HTTPException):
        reserve(restarted)


def test_concurrent_request_rejected_and_slot_released(guard):
    entered, release = threading.Event(), threading.Event()
    def first():
        with guard.admit('a'):
            entered.set()
            release.wait(5)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(first)
        assert entered.wait(2)
        try:
            with pytest.raises(HTTPException) as error:
                reserve(guard, 'b')
            assert error.value.headers['Retry-After'] == '15'
        finally:
            release.set()
        task.result()
    assert reserve(guard, 'b')


def test_api_blocks_before_agent_and_ignores_spoofed_forwarding(guard, monkeypatch):
    server = importlib.import_module('src.server.app')
    monkeypatch.setattr(server, 'get_guard', lambda: guard)
    calls = []
    def answer(question):
        calls.append(question)
        return {'question': question, 'answer': '6', 'sql': 'SELECT 6', 'ok': True}
    monkeypatch.setattr(server, 'ask', answer)
    with TestClient(server.app) as client:
        for peer in ['1.2.3.4', '5.6.7.8']:
            assert client.post('/ask', json={'question': 'Count orders'}, headers={'X-Forwarded-For': peer}).status_code == 200
        response = client.post('/ask', json={'question': 'Count orders'}, headers={'X-Forwarded-For': '9.9.9.9'})
        assert response.status_code == 429 and len(calls) == 2
        assert client.get('/health').status_code == 200
        assert client.get('/').status_code == 200


def test_large_body_rejected_before_provider(monkeypatch):
    server = importlib.import_module('src.server.app')
    def forbidden(_):
        pytest.fail('Provider should not run')
    monkeypatch.setattr(server, 'ask', forbidden)
    with TestClient(server.app) as client:
        response = client.post('/ask', content=b' ' * 9000, headers={'Content-Type': 'application/json'})
        assert response.status_code == 413


@pytest.mark.parametrize('limit', [0, -1, -5])
def test_invalid_limits_fail_closed(tmp_path, limit):
    with pytest.raises(ValueError):
        DemoGuard(tmp_path / 'usage.sqlite', limit)


def test_llm_has_output_cap(monkeypatch):
    from src.agent import graph
    monkeypatch.setenv('LLM_MAX_TOKENS', '1024')
    monkeypatch.setenv('GROQ_API_KEY', 'test-placeholder')
    assert graph._make_llm().max_tokens == 1024
    monkeypatch.setenv('DEMO_MODE', 'true')
    assert graph._make_llm().max_retries == 0

def test_client_window_expires(tmp_path):
    now = [1800000000.0]
    guard = DemoGuard(tmp_path / 'usage.sqlite', 1, clock=lambda: now[0])
    reserve(guard)
    with pytest.raises(HTTPException):
        reserve(guard)
    now[0] += 60
    assert reserve(guard)


def test_chunked_body_limit():
    import asyncio
    from src.server.demo import BodyLimitMiddleware
    called, sent = [], []
    async def app(scope, receive, send):
        called.append(True)
    chunks = iter([{'type': 'http.request', 'body': b'a' * 5000, 'more_body': True},
                   {'type': 'http.request', 'body': b'b' * 5000, 'more_body': False}])
    async def receive():
        return next(chunks)
    async def send(message):
        sent.append(message)
    asyncio.run(BodyLimitMiddleware(app)({'type': 'http', 'path': '/ask'}, receive, send))
    assert not called
    assert sent[0]['status'] == 413


def test_five_questions_then_sixth_blocked_and_other_browser_allowed(tmp_path, monkeypatch):
    server = importlib.import_module('src.server.app')
    now = [1800000000.0]
    guard = DemoGuard(tmp_path / 'usage.sqlite', clock=lambda: now[0])
    monkeypatch.setattr(server, 'get_guard', lambda: guard)
    calls = []
    def answer(question):
        calls.append(question)
        return {'question': question, 'answer': '6', 'sql': 'SELECT 6', 'ok': True}
    monkeypatch.setattr(server, 'ask', answer)
    with TestClient(server.app) as first, TestClient(server.app) as second:
        for _ in range(5):
            response = first.post('/ask', json={'question': 'Count orders'})
            assert response.status_code == 200
        assert 'demo_session' in first.cookies
        rejected = first.post('/ask', json={'question': 'Count orders'})
        assert rejected.status_code == 429 and len(calls) == 5
        assert rejected.headers['Retry-After'] == '60'
        assert second.post('/ask', json={'question': 'Count orders'}).status_code == 200
        now[0] += 60
        assert first.post('/ask', json={'question': 'Count orders'}).status_code == 200


def test_session_cookie_signature_and_security(monkeypatch):
    from starlette.requests import Request
    from starlette.responses import Response
    from src.server.demo import browser_session
    monkeypatch.setenv('DEMO_SECURE_COOKIE', 'true')
    request = Request({'type': 'http', 'headers': []})
    response = Response()
    identity = browser_session(request, response)
    header = response.headers['set-cookie']
    assert 'HttpOnly' in header and 'Secure' in header and 'SameSite=lax' in header
    cookie = header.split(';')[0]
    valid = Request({'type': 'http', 'headers': [(b'cookie', cookie.encode())]})
    assert browser_session(valid, Response()) == identity
    forged = Request({'type': 'http', 'headers': [(b'cookie', b'demo_session=' + b'a'*32 + b'.' + b'b'*64)]})
    assert browser_session(forged, Response()) != 'a'*32


def test_failed_first_request_still_sets_session_cookie(tmp_path, monkeypatch):
    server = importlib.import_module('src.server.app')
    guard = DemoGuard(tmp_path / 'usage.sqlite', 1)
    monkeypatch.setattr(server, 'get_guard', lambda: guard)
    def broken(_):
        raise RuntimeError('provider error')
    monkeypatch.setattr(server, 'ask', broken)
    with TestClient(server.app) as client:
        first = client.post('/ask', json={'question': 'Count orders'})
        assert first.status_code == 502 and 'demo_session' in client.cookies
        assert client.post('/ask', json={'question': 'Count orders'}).status_code == 429
