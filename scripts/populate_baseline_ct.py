"""
Utility script to populate initial baseline CT scans (NIfTI & DICOM)
in data/ct_scans/raw/ for immediate verification and pipeline readiness.
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

# Add data pre processing to sys.path
sys.path.insert(0, os.path.abspath("data pre processing"))
from phantom_generator import generate_anatomical_ct_phantom

def populate():
    base_dir = os.path.abspath("data/ct_scans")
    nifti_vol_dir = os.path.join(base_dir, "raw", "nifti", "volumes")
    nifti_mask_dir = os.path.join(base_dir, "raw", "nifti", "ground_truth_masks")
    dicom_patient_dir = os.path.join(base_dir, "raw", "dicom", "patient_001_abdominal_ct")
    
    os.makedirs(nifti_vol_dir, exist_ok=True)
    os.makedirs(nifti_mask_dir, exist_ok=True)
    os.makedirs(dicom_patient_dir, exist_ok=True)

    print("Generating baseline anatomical CT scan (NIfTI)...")
    ct_vol, gt_vol, organ_map = generate_anatomical_ct_phantom(
        dimensions=(64, 128, 128),
        spacing=(0.8, 0.8, 2.5),
        noise_sigma=12.0
    )

    nifti_vol_path = os.path.join(nifti_vol_dir, "ct_scan_sample_01.nii.gz")
    nifti_mask_path = os.path.join(nifti_mask_dir, "ct_scan_sample_01_seg.nii.gz")

    ct_vol.save_nifti(nifti_vol_path)
    gt_vol.save_nifti(nifti_mask_path)
    print(f"Saved NIfTI scan: {nifti_vol_path}")
    print(f"Saved Ground Truth Mask: {nifti_mask_path}")

    # Generate multi-slice DICOM series with pydicom
    print("Generating multi-slice clinical DICOM series with pydicom...")
    ct_array = np.clip(ct_vol.pixel_array, -1024, 3071).astype(np.int16)
    num_slices, height, width = ct_array.shape
    spacing = ct_vol.spacing
    origin = ct_vol.origin

    study_uid = generate_uid()
    series_uid = generate_uid()

    for i in range(num_slices):
        slice_z = origin[2] + i * spacing[2]
        filename = os.path.join(dicom_patient_dir, f"slice_{i+1:04d}.dcm")
        
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b'\0'*128)
        ds.Modality = "CT"
        ds.PatientID = "PATIENT_001"
        ds.PatientName = "SAMPLE^ABDOMINAL_001"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = CTImageStorage
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(slice_z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
        ds.SliceThickness = float(spacing[2])
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.Rows = height
        ds.Columns = width
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        slice_2d = ct_array[i, :, :]
        ds.PixelData = slice_2d.tobytes()
        ds.save_as(filename, enforce_file_format=False)

    print(f"Saved {num_slices} standard DICOM slices in {dicom_patient_dir}")
    print("Baseline CT collection populated successfully!")

if __name__ == "__main__":
    populate()
