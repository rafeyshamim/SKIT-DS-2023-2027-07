"""Central settings for the Streamlit frontend (override with environment variables)."""
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Mock mode lets the UI run end-to-end before the FastAPI backend is ready.
# Set USE_MOCK_BACKEND=0 once the backend endpoints exist.
USE_MOCK_BACKEND = os.getenv("USE_MOCK_BACKEND", "1") == "1"
REQUEST_TIMEOUT_S = 120
UPLOADER_TYPES = ["dcm", "nii", "gz"]   # "gz" is validated as ".nii.gz" in scan_reader
