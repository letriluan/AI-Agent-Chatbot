"""Probe behaviour is tested with local fakes, not requests to a live service."""
import io
import sys
from urllib.error import HTTPError, URLError

import pytest

from evals import availability


class Response(io.BytesIO):
    status = 200


def test_probe_requires_expected_json_not_a_render_loading_page(monkeypatch):
    monkeypatch.setattr(availability, 'urlopen', lambda *a, **k: Response(b'<html>Starting</html>'))
    assert not availability.probe('https://example.test', '/health', 'ok')['available']
    monkeypatch.setattr(availability, 'urlopen', lambda *a, **k: Response(b'{"status":"ready"}'))
    assert not availability.probe('https://example.test', '/health', 'ok')['available']
    assert availability.probe('https://example.test', '/ready', 'ready')['available']


@pytest.mark.parametrize('error', [HTTPError('url', 503, 'secret-body', {}, None), URLError('secret'), TimeoutError('secret')])
def test_probe_preserves_failure_without_exposing_error_body(monkeypatch, error):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(availability, 'urlopen', fail)
    result = availability.probe('https://example.test', '/health', 'ok')
    assert not result['available']
    assert result['error_type'] == type(error).__name__
    assert 'secret' not in str(result)


def test_sample_uses_health_then_ready_without_model_calls(monkeypatch):
    endpoints = []
    def respond(request, **kwargs):
        endpoints.append(request.full_url)
        return Response(b'{"status":"ok"}' if request.full_url.endswith('/health') else b'{"status":"ready"}')
    monkeypatch.setattr(availability, 'urlopen', respond)
    result = availability.sample('https://example.test/', declared_quiet_seconds=1200)
    assert endpoints == ['https://example.test/health', 'https://example.test/ready']
    assert result['available'] and not result['model_called']
    assert not result['cold_start_confirmed']
    down = dict(result, available=False)
    summary = availability.summarise([result, down])
    assert summary['sampled_availability'] == 0.5
    assert not summary['continuous_uptime_measured']


@pytest.mark.parametrize('url', ['file:///tmp/file', 'https://user:password@example.test', 'https://example.test/?token=secret'])
def test_probe_rejects_non_origins_and_embedded_credentials(url):
    with pytest.raises(ValueError):
        availability.sample(url)


def test_existing_observation_is_preserved_without_another_request(tmp_path, monkeypatch):
    output = tmp_path / 'saved.json'
    output.write_text('old observation')
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(output)])
    monkeypatch.setattr(availability, 'sample', lambda *a: pytest.fail('No new request expected'))
    with pytest.raises(ValueError, match='preserve prior observations'):
        availability.main()
    assert output.read_text() == 'old observation'
