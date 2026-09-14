import sys
import os

# Robustly find project root and src directory in any environment (local, Vercel /var/task, container)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cwd = os.getcwd()

candidate_paths = [
    cwd,
    os.path.join(cwd, "src"),
    current_dir,
    os.path.join(current_dir, "src"),
    parent_dir,
    os.path.join(parent_dir, "src"),
    "/var/task",
    "/var/task/src",
    "/var/task/api"
]

for p in candidate_paths:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from app import app
except ImportError:
    # In case app is in current or parent directory
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    from app import app
