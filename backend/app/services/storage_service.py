import os
import shutil
import uuid
import zipfile
from typing import Tuple, List, Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings
from app.core.logging import logger


class StorageService:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.processed_dir = settings.PROCESSED_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def save_upload_file(self, upload_file: UploadFile) -> Tuple[str, str, int]:
        """
        Saves an uploaded file to the local storage directory with a unique scan UID.
        Returns: (scan_uid, file_path, file_size_bytes)
        """
        scan_uid = f"SCAN_{uuid.uuid4().hex[:12].upper()}"
        file_ext = os.path.splitext(upload_file.filename or "")[1].lower()
        if not file_ext:
            file_ext = ".dcm"

        dest_filename = f"{scan_uid}{file_ext}"
        dest_path = os.path.join(self.upload_dir, dest_filename)

        total_bytes = 0
        try:
            with open(dest_path, "wb") as buffer:
                while chunk := upload_file.file.read(1024 * 1024):  # 1MB chunks
                    total_bytes += len(chunk)
                    if total_bytes > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB"
                        )
                    buffer.write(chunk)
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        logger.info(f"Saved upload file {upload_file.filename} as {dest_path} ({total_bytes} bytes)")
        return scan_uid, dest_path, total_bytes

    def extract_zip_if_needed(self, file_path: str, scan_uid: str) -> str:
        """
        If the file is a ZIP archive containing DICOM slices, extracts it to a subfolder.
        Returns directory path containing individual slice files.
        """
        if not file_path.lower().endswith(".zip"):
            return file_path

        extract_dir = os.path.join(self.upload_dir, f"{scan_uid}_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted ZIP archive to {extract_dir}")
            return extract_dir
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Uploaded file claims to be ZIP but is corrupted.")

    def get_processed_path(self, scan_uid: str) -> str:
        return os.path.join(self.processed_dir, f"{scan_uid}_volume.npy")

    def delete_scan_files(self, file_path: str, processed_path: Optional[str] = None):
        """Clean up raw and processed files when a scan is deleted."""
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
