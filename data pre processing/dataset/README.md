# CT Scan Datasets for Preprocessing & 3D Reconstruction Pipeline

The organized CT scan datasets (including clinical DICOM series, NIfTI volumes, and ground truth multi-organ segmentation masks) are managed under:

```
data/ct_scans/
├── raw/
│   ├── dicom/
│   └── nifti/
│       ├── volumes/
│       └── ground_truth_masks/
├── preprocessed/
├── validation_ground_truth/
├── reconstructed_3d/
└── metadata/
    ├── dataset_manifest.csv
    ├── download_sources.json
    └── DATASET_COLLECTION_GUIDE.md
```

See [data/ct_scans/README.md](../../data/ct_scans/README.md) and [DATASET_COLLECTION_GUIDE.md](../../data/ct_scans/metadata/DATASET_COLLECTION_GUIDE.md) for full instructions and download links.
