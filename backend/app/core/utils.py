import os
import re
import hashlib
import datetime
from typing import Optional


def get_utc_now() -> datetime.datetime:
    """
    Returns timezone-naive UTC datetime compatible with SQLAlchemy, SQLite, and PostgreSQL.
    Replaces deprecated datetime.datetime.utcnow().
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file for integrity verification and deduplication."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes uploaded filename against path traversal attacks and illegal characters.
    """
    base_name = os.path.basename(filename)
    # Remove any non-alphanumeric characters except dot, dash, and underscore
    clean_name = re.sub(r"[^\w\.-]", "_", base_name)
    return clean_name or "scan.dcm"
