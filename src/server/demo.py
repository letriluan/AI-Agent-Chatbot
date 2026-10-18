"""Single-instance demo admission control; never trusts client forwarding headers."""
from contextlib import contextmanager, closing
import hashlib
import hmac
import secrets
import math
import re
import os
from pathlib import Path
import sqlite3
import threading
import time

from fastapi import HTTPException


class DemoGuard:
    def __init__(self, path, per_client=5, clock=time.time):
        self.path = Path(path)
        self.per_client = per_client
        if per_client < 1:
            raise ValueError('Demo limits must be positive')
        self.clock = clock
        self.slot = threading.BoundedSemaphore(1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS admissions (at REAL NOT NULL, client TEXT NOT NULL)')
            conn.execute('CREATE INDEX IF NOT EXISTS admissions_at ON admissions(at)')
            conn.commit()

    @contextmanager
    def admit(self, client):
        if not self.slot.acquire(blocking=False):
            raise HTTPException(429, detail={'message': 'The demo is answering another question. Please try again shortly.'}, headers={'Retry-After': '15'})
        try:
            now = self.clock()
            # Hash the server-signed browser session identifier.
            identity = hashlib.sha256(client.encode()).hexdigest()
            with closing(sqlite3.connect(self.path, timeout=5)) as conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('DELETE FROM admissions WHERE at <= ?', (now - 60,))
                count, oldest = conn.execute('SELECT COUNT(*),MIN(at) FROM admissions WHERE client=? AND at>?', (identity, now - 60)).fetchone()
                if count >= self.per_client:
                    raise HTTPException(429, detail={'message': f'You can ask up to {self.per_client} questions per minute. Please wait a moment and try again.'}, headers={'Retry-After': str(max(1, math.ceil(oldest + 60 - now)))})
                conn.execute('INSERT INTO admissions VALUES (?,?)', (now, identity))
                conn.commit()
            yield
        finally:
            self.slot.release()


_guard = None
_guard_lock = threading.Lock()


def get_guard():
    global _guard
    if os.getenv('DEMO_MODE', 'false').lower() != 'true':
        return None
    with _guard_lock:
        if _guard is None:
            _guard = DemoGuard(os.getenv('DEMO_USAGE_DB', 'db/demo-usage.sqlite'),
                int(os.getenv('DEMO_CLIENT_PER_MINUTE', '5')))
    return _guard


_SESSION_SECRET = os.getenv('DEMO_SESSION_SECRET') or secrets.token_hex(32)


def browser_session(request, response):
    """Signed browser identity; no authentication or per-human identity claim."""
    cookie = request.cookies.get('demo_session', '')
    parts = cookie.split('.')
    if len(parts) == 2 and re.fullmatch(r'[0-9a-f]{32}', parts[0]) and re.fullmatch(r'[0-9a-f]{64}', parts[1]):
        expected = hmac.new(_SESSION_SECRET.encode(), parts[0].encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, parts[1]):
            return parts[0]
    identity = secrets.token_hex(16)
    signature = hmac.new(_SESSION_SECRET.encode(), identity.encode(), hashlib.sha256).hexdigest()
    response.set_cookie('demo_session', identity + '.' + signature, max_age=86400,
                        httponly=True, secure=os.getenv('DEMO_SECURE_COOKIE', 'false').lower() == 'true', samesite='lax')
    return identity


class BodyLimitMiddleware:
    """Bound /ask JSON before parsing, including chunked requests."""
    def __init__(self, app, maximum=8192):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/ask':
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > self.maximum:
                from starlette.responses import JSONResponse
                return await JSONResponse({'detail': {'message': 'Question payload is too large.'}}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get('more_body', False):
                break
        async def bounded_receive():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope, bounded_receive, send)
