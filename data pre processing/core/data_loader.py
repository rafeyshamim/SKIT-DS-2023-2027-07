"""
Milestone 2: CT Scan & Medical Image Data Collection & Ingestion
================================================================
Handles loading of DICOM series, NIfTI volumes, BraTS slice metadata, and HDF5 slices
using SimpleITK and pydicom while strictly preserving coordinate frames.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class MedicalVolume:
    """
    Unified representation of a 3D medical volume with spatial metadata.
    """
    sitk_image: sitk.Image
    pixel_array: np.ndarray  # Shape: (Z, Y, X)
    spacing: Tuple[float, float, float]  # (sx, sy, sz) in mm
    origin: Tuple[float, float, float]   # (ox, oy, oz) in mm
    direction: Tuple[float, ...]         # 9-element direction cosines
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sitk(cls, sitk_image: sitk.Image, metadata: Optional[Dict[str, Any]] = None) -> "MedicalVolume":
        arr = sitk.GetArrayFromImage(sitk_image)  # converts to numpy (Z, Y, X)
        return cls(
            sitk_image=sitk_image,
            pixel_array=arr,
            spacing=sitk_image.GetSpacing(),
            origin=sitk_image.GetOrigin(),
            direction=sitk_image.GetDirection(),
            metadata=metadata or {}
        )

    def to_sitk(self) -> sitk.Image:
        """Syncs pixel_array changes back to a SimpleITK image with spatial metadata."""
        img = sitk.GetImageFromArray(self.pixel_array)
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        return img

    def save_nifti(self, output_path: str) -> None:
        """Saves volume to compressed NIfTI (.nii.gz)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = self.to_sitk()
        sitk.WriteImage(img, output_path)


def load_dicom_series(directory_path: str, series_id: Optional[str] = None) -> MedicalVolume:
    """
    Loads a multi-slice DICOM series from a folder using SimpleITK's GDCM reader.
    Ensures correct physical slice ordering along the scanner's Z-axis.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"DICOM directory not found: {directory_path}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory_path)
    if not series_ids:
        raise ValueError(f"No DICOM series detected in directory: {directory_path}")

    selected_series = series_id if series_id in series_ids else series_ids[0]
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path, selected_series)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    sitk_image = reader.Execute()
    
    # Extract metadata tags from the first slice
    metadata: Dict[str, Any] = {"series_id": selected_series, "slice_count": len(dicom_names)}
    if len(dicom_names) > 0 and reader.HasMetaDataKey(0, "0028|1052"): # Rescale Intercept
        metadata["rescale_intercept"] = reader.GetMetaData(0, "0028|1052")
    if len(dicom_names) > 0 and reader.HasMetaDataKey(0, "0028|1053"): # Rescale Slope
        metadata["rescale_slope"] = reader.GetMetaData(0, "0028|1053")

    return MedicalVolume.from_sitk(sitk_image, metadata=metadata)


def load_nifti(file_path: str) -> MedicalVolume:
    """
    Loads a 3D medical volume from a NIfTI (.nii or .nii.gz) file using SimpleITK.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    sitk_image = reader.Execute()
    
    metadata = {
        "source_file": file_path,
        "pixel_id_type": sitk_image.GetPixelIDTypeAsString()
    }
    return MedicalVolume.from_sitk(sitk_image, metadata=metadata)


def load_brats_metadata(csv_path: str) -> pd.DataFrame:
    """
    Loads and structures the BraTS20 Training Metadata CSV file.
    Provides analysis on slices, tumor presence, and volume distribution.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"BraTS metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def summarize_brats_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary metrics of the BraTS20 training metadata.
    """
    total_slices = len(df)
    unique_volumes = df['volume'].nunique() if 'volume' in df.columns else 0
    tumor_slices = (df['target'] == 1).sum() if 'target' in df.columns else 0
    bg_ratio_mean = df['background_ratio'].mean() if 'background_ratio' in df.columns else 0.0

    return {
        "total_slices": total_slices,
        "unique_volumes": unique_volumes,
        "tumor_positive_slices": int(tumor_slices),
        "tumor_slice_percentage": float((tumor_slices / total_slices) * 100) if total_slices > 0 else 0.0,
        "mean_background_ratio": float(bg_ratio_mean),
        "volumes_per_slice_avg": float(total_slices / unique_volumes) if unique_volumes > 0 else 0.0
    }


def load_h5_slice(h5_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Safely loads a BraTS preprocessed .h5 slice if available.
    """
    global h5py
    if h5py is None:
        try:
            import h5py as _h5py
            h5py = _h5py
        except ImportError:
            return None

    if not os.path.isfile(h5_path):
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            data = {}
            for k in f.keys():
                data[k] = np.array(f[k])
            return data
    except Exception:
        return None
