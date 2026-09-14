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

handler = app
application = app
