"""Validate uploaded CT files and read series info with SimpleITK."""
import os
import tempfile
from dataclasses import dataclass


class ScanReadError(Exception):
    """Raised with a plain-language message that is safe to show to the user."""


@dataclass
class SeriesInfo:
    scan_id: str
    modality: str
    slice_count: int
    file_format: str          # "DICOM" or "NIfTI"
    dimensions: tuple         # (x, y, z)
    file_count: int


def classify_files(names):
    """Return 'DICOM' or 'NIfTI' for a list of file names, or raise ScanReadError."""
    lowered = [n.lower() for n in names]
    if not lowered:
        raise ScanReadError("No files selected.")
    dcm = [n for n in lowered if n.endswith(".dcm")]
    nii = [n for n in lowered if n.endswith(".nii") or n.endswith(".nii.gz")]
    if len(dcm) + len(nii) != len(lowered):
        raise ScanReadError("Unsupported file type. Please upload .dcm, .nii or .nii.gz files only.")
    if dcm and nii:
        raise ScanReadError("Please upload either a DICOM series or a single NIfTI file, not both.")
    if nii:
        if len(nii) > 1:
            raise ScanReadError("Please upload only one NIfTI file at a time.")
        return "NIfTI"
    return "DICOM"


def read_series_info(files):
    """files: list of (filename, bytes). Returns SeriesInfo or raises ScanReadError."""
    fmt = classify_files([name for name, _ in files])
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ScanReadError("SimpleITK is not installed on the server (pip install SimpleITK).") from exc

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (name, data) in enumerate(files):
            safe = f"{i:05d}_{os.path.basename(name)}"   # basename blocks path tricks
            path = os.path.join(tmp, safe)
            with open(path, "wb") as fh:
                fh.write(data)
            paths.append(path)
        try:
            if fmt == "DICOM":
                return _read_dicom(sitk, tmp, len(files))
            return _read_nifti(sitk, paths[0], files[0][0])
        except ScanReadError:
            raise
        except Exception as exc:  # SimpleITK raises RuntimeError and others
            raise ScanReadError(
                "We couldn't read this scan. The file may be corrupted or not a valid CT volume."
            ) from exc


def _read_dicom(sitk, folder, file_count):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise ScanReadError("No readable DICOM series found in the uploaded files.")
    if len(series_ids) > 1:
        raise ScanReadError(f"The upload contains {len(series_ids)} different series. Please upload one series.")
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_ids[0])
    first = sitk.ImageFileReader()
    first.SetFileName(file_names[0])
    first.ReadImageInformation()

    def tag(key, default):
        return first.GetMetaData(key).strip() if first.HasMetaDataKey(key) else default

    modality = tag("0008|0060", "Unknown")
    if modality != "CT":
        raise ScanReadError(f"This looks like a {modality} scan. Only CT scans are supported.")
    size = first.GetSize()
    return SeriesInfo(
        scan_id=tag("0010|0020", "") or tag("0020|000d", "")[-12:] or "UNKNOWN",
        modality=modality,
        slice_count=len(file_names),
        file_format="DICOM",
        dimensions=(int(size[0]), int(size[1]), len(file_names)),
        file_count=file_count,
    )


def _read_nifti(sitk, path, original_name):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    size = reader.GetSize()
    if len(size) < 3 or size[2] < 2:
        raise ScanReadError("This file is not a 3D volume (at least 2 slices are required).")
    stem = os.path.basename(original_name)
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return SeriesInfo(
        scan_id=stem or "UNKNOWN",
        modality="CT (assumed)",        # NIfTI has no modality tag
        slice_count=int(size[2]),
        file_format="NIfTI",
        dimensions=tuple(int(s) for s in size[:3]),
        file_count=1,
    )
