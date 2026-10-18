import os
import shutil
import uuid
import zipfile
import hashlib
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException, status
from app.core.config import settings
from app.core.logging import logger
from app.core.utils import sanitize_filename


class StorageService:
    """
    Manages secure streaming uploads, chunked I/O, hash verification,
    zip extraction safeguards, and path management for raw/processed medical scans.
    """
    ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".zip", ".nii", ".gz", ".npy"}

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.processed_dir = settings.PROCESSED_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def save_upload_file(self, upload_file: UploadFile) -> Tuple[str, str, int, str]:
        """
        Saves an uploaded file to storage using chunked streaming to prevent memory spikes.
        Calculates SHA-256 on the fly and prevents path traversal.
        Returns: (scan_uid, file_path, file_size_bytes, sha256_hash)
        """
        clean_name = sanitize_filename(upload_file.filename or "scan.dcm")
        file_ext = os.path.splitext(clean_name)[1].lower()
        if not file_ext:
            file_ext = ".dcm"

        if file_ext not in self.ALLOWED_EXTENSIONS and not clean_name.lower().endswith(".nii.gz"):
            logger.warning("Uncommon file extension '%s', proceeding with DICOM interpretation.", file_ext)

        scan_uid = f"SCAN_{uuid.uuid4().hex[:12].upper()}"
        dest_filename = f"{scan_uid}{file_ext}"
        dest_path = os.path.join(self.upload_dir, dest_filename)

        sha256 = hashlib.sha256()
        total_bytes = 0
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        try:
            with open(dest_path, "wb") as buffer:
                while chunk := upload_file.file.read(1024 * 1024):  # 1MB buffer
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE_MB}MB."
                        )
                    sha256.update(chunk)
                    buffer.write(chunk)
        except Exception as e:
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Failed to stream and save uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file: {str(e)}"
            )

        file_hash = sha256.hexdigest()
        logger.info(f"Saved {clean_name} -> {dest_path} ({total_bytes} bytes, SHA256: {file_hash[:12]}...)")
        return scan_uid, dest_path, total_bytes, file_hash

    def extract_zip_if_needed(self, file_path: str, scan_uid: str) -> str:
        """
        Safely extracts ZIP archives containing multi-slice DICOM series.
        Prevents Zip Slip directory traversal vulnerabilities.
        """
        if not file_path.lower().endswith(".zip"):
            return file_path

        extract_dir = os.path.join(self.upload_dir, f"{scan_uid}_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        resolved_base = os.path.abspath(extract_dir)

        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    target_path = os.path.abspath(os.path.join(extract_dir, member))
                    if not target_path.startswith(resolved_base):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="ZIP archive contains unsafe path traversal entries."
                        )
                zip_ref.extractall(extract_dir)
            logger.info(f"Safely extracted ZIP archive to {extract_dir}")
            return extract_dir
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file claims to be ZIP but is corrupted or invalid."
            )

    def get_processed_path(self, scan_uid: str) -> str:
        return os.path.join(self.processed_dir, f"{scan_uid}_volume.npy")

    def delete_scan_files(self, file_path: str, processed_path: Optional[str] = None):
        """Safely cleans up raw and preprocessed volumetric files."""
        try:
            if file_path and os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            if processed_path and os.path.exists(processed_path):
                os.remove(processed_path)
        except Exception as e:
            logger.warning(f"Error deleting files for scan: {e}")


storage_service = StorageService()
