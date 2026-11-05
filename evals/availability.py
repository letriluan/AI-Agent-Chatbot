"""Sample process/database availability without calling the model or /ask."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def probe(base_url, endpoint, expected_status, timeout=120):
    started = time.perf_counter()
    result = dict(endpoint=endpoint, checked_at=datetime.now(timezone.utc).isoformat(),
                  available=False, http_status=None)
    try:
        request = Request(base_url + endpoint, headers={'Accept': 'application/json', 'User-Agent': 'sql-agent-availability/1'})
        with urlopen(request, timeout=timeout) as response:
            result['http_status'] = response.status
            payload = json.loads(response.read(4096))
            result['available'] = (response.status == 200 and isinstance(payload, dict)
                                   and payload.get('status') == expected_status)
    except HTTPError as exc:
        result.update(http_status=exc.code, error_type='HTTPError')
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        result['error_type'] = type(exc).__name__
    result['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
    return result


def sample(base_url, timeout=120, declared_quiet_seconds=None):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {'', '/'}):
        raise ValueError('Use a plain HTTP(S) service origin without credentials, query or path')
    if timeout <= 0 or (declared_quiet_seconds is not None and declared_quiet_seconds < 0):
        raise ValueError('Use a positive timeout and nonnegative quiet period')
    base_url = base_url.rstrip('/')
    # The first request can wake a sleeping instance; keep its time separate.
    health = probe(base_url, '/health', 'ok', timeout)
    ready = probe(base_url, '/ready', 'ready', timeout)
    return dict(base_url=base_url, checked_at=health['checked_at'], health=health, ready=ready,
                available=health['available'] and ready['available'],
                declared_quiet_seconds=declared_quiet_seconds,
                cold_start_confirmed=False, model_called=False,
                note='First health request may include wake-up. Quiet time and actual platform sleep are not independently verified.')


def summarise(samples):
    if not samples or len({s['base_url'] for s in samples}) != 1:
        raise ValueError('Provide samples from one service')
    return dict(sample_count=len(samples), available_samples=sum(s['available'] for s in samples),
                sampled_availability=sum(s['available'] for s in samples) / len(samples),
                continuous_uptime_measured=False, model_called=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://ai-agent-chatbot-demo.onrender.com')
    parser.add_argument('--timeout-seconds', type=float, default=120)
    parser.add_argument('--declared-quiet-seconds', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Choose a new output file to preserve prior observations')
    result = sample(args.url, args.timeout_seconds, args.declared_quiet_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['available']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
