"""Use one worker and raw transport peers; untrusted forwarding headers are ignored."""
import os

port = int(os.getenv('PORT', '8000'))
if not 1 <= port <= 65535:
    raise ValueError('Invalid PORT')
os.execvp('python', ['python', '-m', 'uvicorn', 'src.server.app:app', '--host', '0.0.0.0', '--port', str(port), '--workers', '1', '--no-proxy-headers', '--no-access-log'])
