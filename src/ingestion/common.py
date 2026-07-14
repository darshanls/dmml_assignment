"""
Common utilities shared by all ingestion scripts:
- Logging configuration (console + rotating file handler)
- Retry decorator wrapper (via tenacity) for network / IO calls
- Helper to compute partitioned raw storage paths: raw/<source>/<type>/<YYYY-MM-DD>/
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to console and to logs/ingestion.log (rotating)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "ingestion.log"), maxBytes=2_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def partitioned_path(source: str, data_type: str, run_date: datetime = None) -> str:
    """
    Build a partitioned raw-storage directory path:
        data/raw/<source>/<data_type>/<YYYY-MM-DD>/
    and ensure it exists.
    """
    run_date = run_date or datetime.utcnow()
    partition = run_date.strftime("%Y-%m-%d")
    path = os.path.join(RAW_DATA_DIR, source, data_type, partition)
    os.makedirs(path, exist_ok=True)
    return path


def network_retry(func):
    """Decorator: retry up to 4 times with exponential backoff on network/IO errors."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, OSError, requests.exceptions.RequestException)
        ),
    )(func)
