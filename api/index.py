import sys
import os

# Resolve paths for Vercel serverless functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

for path in [BASE_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app import app
