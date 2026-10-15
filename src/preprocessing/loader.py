"""
Data loading utilities for volumetric medical imaging data.

Supports two sources:

1. **MedMNIST3D** (``source: medmnist`` in config.yaml)
   - Datasets: organmnist3d, nodulemnist3d, fracturemnist3d, …
   - Downloaded automatically via the ``medmnist`` package.

2. **NIfTI files** (``source: nifti`` in config.yaml)
   - Individual volumes: ``load_nifti_volume(filepath)``
   - Whole dataset:      ``load_nifti_dataset(data_dir)``
   - Expected directory structure::

         data/raw/
             train/
                 <class_A>/  *.nii.gz
                 <class_B>/  *.nii.gz
             val/  …
             test/ …
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# MedMNIST3D — dataset names -> python class names
# ---------------------------------------------------------------------------
_MEDMNIST3D_CLASSES: Dict[str, str] = {
    "organmnist3d":   "OrganMNIST3D",
    "nodulemnist3d":  "NoduleMNIST3D",
    "adrenalmnist3d": "AdrenalMNIST3D",
    "fracturemnist3d": "FractureMNIST3D",
    "vesselmnist3d":  "VesselMNIST3D",
    "synapsemnist3d": "SynapseMNIST3D",
}


def load_medmnist3d(
    dataset_name: str,
    data_dir: str = "data/",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a MedMNIST3D dataset into (volumes, labels) arrays for each split.

    The dataset is downloaded to *data_dir* on the first call and cached
    for subsequent calls.

    Args:
        dataset_name: Lowercase dataset key, e.g. ``'organmnist3d'``.
        data_dir:     Directory where MedMNIST caches its downloads.

    Returns:
        Dictionary with keys ``'train'``, ``'val'``, ``'test'``.
        Each value is a tuple ``(volumes, labels)`` where:

        - *volumes* — ``float32`` array of shape ``(N, D, H, W)``,
          raw voxel values in ``[0, 255]``.
        - *labels*  — ``int64``  array of shape ``(N,)``.

    Raises:
        ImportError: If the ``medmnist`` package is not installed.
        ValueError:  If *dataset_name* is not a known MedMNIST3D key.
    """
    try:
        import medmnist
    except ImportError as exc:
        raise ImportError(
            "The 'medmnist' package is required to load MedMNIST3D datasets.\n"
            "Install it with:  pip install medmnist"
        ) from exc

    dataset_name = dataset_name.lower()
    if dataset_name not in _MEDMNIST3D_CLASSES:
        raise ValueError(
            f"Unknown MedMNIST3D dataset: '{dataset_name}'.\n"
            f"Supported options: {list(_MEDMNIST3D_CLASSES.keys())}"
        )

    class_name = _MEDMNIST3D_CLASSES[dataset_name]
    DataClass = getattr(medmnist, class_name, None)
    if DataClass is None:
        raise ImportError(
            f"Could not find class '{class_name}' in the installed medmnist "
            f"package.  Please upgrade:  pip install --upgrade medmnist"
        )

    logger.info("Loading MedMNIST3D dataset: '%s' ...", dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for split in ("train", "val", "test"):
        dataset = DataClass(split=split, download=True, root=data_dir)

        # Access the underlying numpy arrays directly — avoids PIL overhead
        # for 3D datasets and is more memory-efficient.
        volumes: np.ndarray = dataset.imgs.astype(np.float32)   # (N, D, H, W)
        labels: np.ndarray  = dataset.labels.reshape(-1).astype(np.int64)  # (N,)

        # Squeeze trailing channel dim if present (some MedMNIST versions add it)
        if volumes.ndim == 5:
            volumes = volumes[..., 0]

        splits[split] = (volumes, labels)
        logger.info(
            "  %-6s : volumes=%s  labels=%s", split, volumes.shape, labels.shape
        )

    return splits


# ---------------------------------------------------------------------------
# NIfTI — individual file / directory loaders
# ---------------------------------------------------------------------------

def load_nifti_volume(filepath: str) -> np.ndarray:
    """
    Load a single NIfTI volumetric file (``.nii`` or ``.nii.gz``).

    The nibabel convention ``(W, H, D)`` is transposed to ``(D, H, W)``
    so that the depth axis is first — consistent with the rest of the pipeline.

    Args:
        filepath: Path to the NIfTI file.

    Returns:
        ``float32`` array of shape ``(D, H, W)`` in the original voxel units.

    Raises:
        ImportError:     If ``nibabel`` is not installed.
        FileNotFoundError: If *filepath* does not exist.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "The 'nibabel' package is required to load NIfTI files.\n"
            "Install it with:  pip install nibabel"
        ) from exc

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI file not found: '{filepath}'")

    logger.debug("Loading NIfTI: %s", filepath)
    img = nib.load(filepath)
    data: np.ndarray = img.get_fdata(dtype=np.float32)

    # nibabel returns axes in (W, H, D) order — transpose to (D, H, W)
    if data.ndim == 3:
        data = np.transpose(data, axes=(2, 1, 0))
    elif data.ndim == 4:
        # 4-D volumes (e.g. multi-echo / time-series) — take the first volume
        data = np.transpose(data[..., 0], axes=(2, 1, 0))
        logger.warning(
            "4-D NIfTI detected at '%s'; using the first 3-D volume.", filepath
        )
    else:
        raise ValueError(
            f"Unexpected NIfTI dimensionality {data.ndim} in: '{filepath}'."
        )

    logger.debug("Loaded NIfTI shape: %s", data.shape)
    return data


def load_nifti_dataset(
    data_dir: str,
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[list, np.ndarray]:
    """
    Load all NIfTI volumes from a class-structured directory.

    Expected layout::

        data_dir/
            <class_A>/
                volume1.nii.gz
                volume2.nii.gz
            <class_B>/
                …

    Args:
        data_dir:  Root directory containing per-class subdirectories.
        label_map: Mapping ``{class_folder_name: integer_label}``.
                   If *None*, labels are assigned alphabetically (0-indexed).

    Returns:
        Tuple ``(volumes, labels)`` where

        - *volumes* — Python list of ``float32`` arrays, each ``(D, H, W)``.
        - *labels*  — ``int64`` array of shape ``(N,)``.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError:        If no NIfTI files are found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: '{data_dir}'.\n"
            "Please provide a valid directory containing per-class NIfTI files."
        )

    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found in: '{data_dir}'.\n"
            "Expected structure:  data_dir/<class_name>/*.nii.gz"
        )

    if label_map is None:
        label_map = {cls: idx for idx, cls in enumerate(class_dirs)}

    logger.info("Loading NIfTI dataset from '%s' | classes: %s", data_dir, label_map)

    volumes = []
    labels  = []
    supported_extensions = (".nii", ".nii.gz")

    for class_name in class_dirs:
        label = label_map.get(class_name)
        if label is None:
            logger.warning("Skipping unlabelled directory: '%s'", class_name)
            continue

        class_dir = os.path.join(data_dir, class_name)
        files = [
            f for f in os.listdir(class_dir)
            if any(f.endswith(ext) for ext in supported_extensions)
        ]

        logger.info(
            "  Class '%s' (label=%d): %d files", class_name, label, len(files)
        )

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                vol = load_nifti_volume(fpath)
                volumes.append(vol)
                labels.append(label)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load '%s': %s", fpath, exc)

    if not volumes:
        raise ValueError(
            f"No valid NIfTI volumes were loaded from: '{data_dir}'."
        )

    labels_array = np.array(labels, dtype=np.int64)
    logger.info(
        "Total loaded: %d volumes | label distribution: %s",
        len(volumes),
        {lbl: int((labels_array == lbl).sum()) for lbl in np.unique(labels_array)},
    )
    return volumes, labels_array
