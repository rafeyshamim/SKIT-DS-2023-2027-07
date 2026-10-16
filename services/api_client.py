"""Thin client for the FastAPI backend (with a mock mode for UI development).

Endpoint contract proposed for the backend (Member 2):
  POST /api/v1/scans                     -> {"scan_id": str}
  POST /api/v1/scans/{id}/files          (multipart field "file")
  POST /api/v1/scans/{id}/preprocess     -> {"status": "ok"}
"""
import time
import uuid

import requests

import config


class BackendError(Exception):
    """Plain-language error raised when the backend is unreachable or rejects a request."""


def _post(path, **kwargs):
    try:
        resp = requests.post(f"{config.BACKEND_URL}{path}", timeout=config.REQUEST_TIMEOUT_S, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.ConnectionError as exc:
        raise BackendError("Can't reach the analysis server. Please check that it is running and try again.") from exc
    except requests.Timeout as exc:
        raise BackendError("The server took too long to respond. Please try again.") from exc
    except (requests.RequestException, ValueError) as exc:
        raise BackendError(f"The server rejected the request ({exc}).") from exc


def upload_scan(files, on_progress):
    """Upload files one by one. on_progress(done, total) is called after each file. Returns scan_id."""
    total = len(files)
    if config.USE_MOCK_BACKEND:
        scan_id = uuid.uuid4().hex[:12]
        for i in range(total):
            time.sleep(min(0.6, 3.0 / total))
            on_progress(i + 1, total)
        return scan_id

    scan_id = _post("/api/v1/scans").get("scan_id")
    if not scan_id:
        raise BackendError("The server did not return a scan ID.")
    for i, (name, data) in enumerate(files):
        _post(f"/api/v1/scans/{scan_id}/files", files={"file": (name, data)})
        on_progress(i + 1, total)
    return scan_id


def run_preprocessing(scan_id):
    """Trigger backend preprocessing (resize + normalize + reconstruction prep)."""
    if config.USE_MOCK_BACKEND:
        time.sleep(2)
        return
    _post(f"/api/v1/scans/{scan_id}/preprocess")
