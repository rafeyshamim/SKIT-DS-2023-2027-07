# Data Directory

This directory stores raw and processed volumetric medical imaging datasets.

## Directory Structure

```
data/
├── raw/         # Raw volumetric data (.nii, .nii.gz, MedMNIST downloads)
└── processed/   # Preprocessed, normalized NumPy volumes (.npy, .npz)
```

## Supported Datasets

- **MedMNIST3D**: Standard volumetric biomedical benchmark (e.g. `organmnist3d`, `nodulemnist3d`, `fracturemnist3d`). Downloaded automatically via `medmnist` when `source: medmnist` is configured in `config/config.yaml`.
- **NIfTI (.nii, .nii.gz)**: Clinical CT and MRI scans. Placed in `data/raw/` when `source: nifti` is selected.
