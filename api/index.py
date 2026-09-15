import sys
import os

# Set up candidate search paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cwd = os.getcwd()

candidate_paths = [
    parent_dir,
    os.path.join(parent_dir, "src"),
    current_dir,
    os.path.join(current_dir, "src"),
    cwd,
    os.path.join(cwd, "src"),
    "/var/task",
    "/var/task/src",
    "/var/task/api"
]

for p in candidate_paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Top-level ASGI app and handlers for Vercel static AST parser
from app import app

import urllib.parse

class VercelPathFixMiddleware:
    """Pure ASGI middleware to correct scope['path'] rewritten by Vercel serverless rewrites."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            q_bytes = scope.get("query_string", b"")
            if q_bytes and b"__path__=" in q_bytes:
                params = urllib.parse.parse_qs(q_bytes.decode("latin1"), keep_blank_values=True)
                if "__path__" in params:
                    raw_p = params.pop("__path__")[0]
                    clean_p = raw_p if raw_p.startswith("/") else ("/" + raw_p)
                    # Normalize root / empty
                    clean_p = clean_p.rstrip("/") if len(clean_p) > 1 else clean_p
                    scope["path"] = clean_p or "/"
                    scope["raw_path"] = (clean_p or "/").encode("latin1")
                    scope["query_string"] = urllib.parse.urlencode(params, doseq=True).encode("latin1")
            elif scope.get("path") in ("/api/index.py", "/api/index", "/index.py", "/index"):
                scope["path"] = "/"
                scope["raw_path"] = b"/"
        await self.app(scope, receive, send)

handler = VercelPathFixMiddleware(app)
application = handler

