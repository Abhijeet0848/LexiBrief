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

class VercelPathFixMiddleware:
    """Pure ASGI middleware to correct scope['path'] rewritten by Vercel serverless rewrites."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            matched_path = (
                headers.get(b"x-matched-path", b"").decode("latin1")
                or headers.get(b"x-vercel-matched-path", b"").decode("latin1")
                or headers.get(b"x-forwarded-uri", b"").decode("latin1")
            )
            if matched_path:
                clean_path = matched_path.split("?")[0]
                scope["path"] = clean_path
                scope["raw_path"] = clean_path.encode("latin1")
            elif scope.get("path") in ("/api/index.py", "/api/index", "/index.py", "/index"):
                scope["path"] = "/"
                scope["raw_path"] = b"/"
        await self.app(scope, receive, send)

handler = VercelPathFixMiddleware(app)
application = handler

