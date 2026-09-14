import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
handlers = [logging.StreamHandler(sys.stdout)]

# Support file logging when filesystem is writable, gracefully fallback on serverless (Vercel/Lambda)
try:
    log_dir = "logs"
    log_filepath = os.path.join(log_dir, "running_logs.log")
    os.makedirs(log_dir, exist_ok=True)
    handlers.append(logging.FileHandler(log_filepath))
except (OSError, PermissionError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=handlers
)

logger = logging.getLogger("textSummarizerLogger")