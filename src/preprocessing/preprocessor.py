"""
Volumetric medical image preprocessing utilities.

Sprint 1 — CNN Input Data Specification.

Preprocessing steps applied to every volume before it enters the 3D CNN:

    1. ``resize_volume``     — trilinear interpolation to target spatial size
    2. ``normalize_volume``  — intensity normalization (min-max or z-score)
    3. ``augment_volume``    — random geometric augmentations (training only)
    4. ``add_channel_dim``   — expand (D, H, W) -> (D, H, W, 1)

Final tensor specification
--------------------------
- Shape : ``(D, H, W, 1)``
- Dtype : ``float32``
- Range : ``[0.0, 1.0]``  (min-max)  or approximately ``N(0, 1)`` (z-score)
"""
from typing import List, Optional

import numpy as np
from scipy.ndimage import rotate, zoom

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_volume(volume: np.ndarray, target_size: List[int]) -> np.ndarray:
    """
    Resize a 3-D volume to *target_size* using trilinear interpolation
    (``scipy.ndimage.zoom`` with ``order=1``).

    Args:
        volume:      Input volume of shape ``(D, H, W)``.
        target_size: Target dimensions ``[D_out, H_out, W_out]``.

    Returns:
        Resized ``float32`` array of shape ``(D_out, H_out, W_out)``.

    Raises:
        ValueError: If *volume* is not 3-D or *target_size* has ≠ 3 elements.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"resize_volume expects a 3-D volume (D, H, W), "
            f"got shape: {volume.shape}"
        )
    if len(target_size) != 3:
        raise ValueError(
            f"target_size must have exactly 3 elements [D, H, W], "
            f"got: {target_size}"
        )

    d, h, w = volume.shape
    td, th, tw = target_size

    if (d, h, w) == (td, th, tw):
        return volume.astype(np.float32)

    zoom_factors = (td / d, th / h, tw / w)
    resized = zoom(volume.astype(np.float32), zoom_factors, order=1)
    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_min_max(
    volume: np.ndarray,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Scale all voxel intensities to ``[0, 1]`` using min-max normalization.

    Optionally clips the dynamic range before scaling — useful for CT HU values.

    Args:
        volume:   Input volume (any range).
        clip_min: Lower clipping bound.  ``None`` disables lower clipping.
        clip_max: Upper clipping bound.  ``None`` disables upper clipping.
        eps:      Small constant added to the denominator to avoid ÷0.

    Returns:
        Normalized ``float32`` volume in ``[0, 1]``.
    """
    vol = volume.astype(np.float32)

    if clip_min is not None or clip_max is not None:
        vol = np.clip(vol, a_min=clip_min, a_max=clip_max)

    v_min = float(vol.min())
    v_max = float(vol.max())
    vol = (vol - v_min) / (v_max - v_min + eps)
    return vol.astype(np.float32)


def normalize_z_score(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize voxel intensities to zero mean and unit variance.

    Args:
        volume: Input volume (any range).
        eps:    Small constant added to the standard deviation to avoid ÷0.

    Returns:
        Standardized ``float32`` volume (mean ≈ 0, std ≈ 1).
    """
    vol = volume.astype(np.float32)
    mean = float(vol.mean())
    std  = float(vol.std())
    return ((vol - mean) / (std + eps)).astype(np.float32)


def normalize_volume(
    volume: np.ndarray,
    method: str = "min_max",
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """
    Dispatch to the appropriate normalization function.

    Args:
        volume:   Input volume.
        method:   ``'min_max'`` or ``'z_score'``.
        clip_min: Only applied for ``'min_max'``.
        clip_max: Only applied for ``'min_max'``.

    Returns:
        Normalized ``float32`` volume.

    Raises:
        ValueError: If *method* is not ``'min_max'`` or ``'z_score'``.
    """
    if method == "min_max":
        return normalize_min_max(volume, clip_min=clip_min, clip_max=clip_max)
    if method == "z_score":
        return normalize_z_score(volume)
    raise ValueError(
        f"Unknown normalization method: '{method}'. "
        "Use 'min_max' or 'z_score'."
    )


# ---------------------------------------------------------------------------
# Channel expansion
# ---------------------------------------------------------------------------

def add_channel_dim(volume: np.ndarray) -> np.ndarray:
    """
    Add a trailing channel dimension.

    ``(D, H, W)`` → ``(D, H, W, 1)``

    Args:
        volume: 3-D volume of shape ``(D, H, W)``.

    Returns:
        4-D tensor of shape ``(D, H, W, 1)``.
    """
    return np.expand_dims(volume, axis=-1)


# ---------------------------------------------------------------------------
# Data augmentation (training only)
# ---------------------------------------------------------------------------

def random_flip(volume: np.ndarray, axes: List[int]) -> np.ndarray:
    """
    Randomly flip a 3-D volume along the specified axes (50 % probability each).

    Args:
        volume: 3-D volume ``(D, H, W)``.
        axes:   List of axes to consider for flipping (e.g. ``[0, 1, 2]``).

    Returns:
        Possibly-flipped volume (same shape as input).
    """
    vol = volume.copy()
    for axis in axes:
        if np.random.random() > 0.5:
            vol = np.flip(vol, axis=axis)
    return vol


def random_rotate(volume: np.ndarray, rotation_range: float) -> np.ndarray:
    """
    Rotate a 3-D volume by a random angle around the depth axis.

    Rotation is applied in the ``(H, W)`` plane (axes 1 and 2) and
    uses bilinear interpolation (``order=1``).

    Args:
        volume:         3-D volume ``(D, H, W)``.
        rotation_range: Maximum rotation angle in degrees.

    Returns:
        Rotated ``float32`` volume of the same shape.
    """
    angle = np.random.uniform(-rotation_range, rotation_range)
    rotated = rotate(
        volume.astype(np.float32),
        angle,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="nearest",
    )
    return rotated.astype(np.float32)


def random_zoom(volume: np.ndarray, zoom_range: float) -> np.ndarray:
    """
    Apply a uniform random zoom and restore the original spatial extent.

    After zooming, the result is centrally cropped or zero-padded back to the
    input shape so that the tensor shape is preserved.

    Args:
        volume:     3-D volume ``(D, H, W)``.
        zoom_range: Maximum fractional zoom change (e.g. ``0.1`` for ±10 %).

    Returns:
        Zoomed ``float32`` volume with the same shape as *volume*.
    """
    factor = 1.0 + np.random.uniform(-zoom_range, zoom_range)
    zoomed = zoom(volume.astype(np.float32), factor, order=1)

    original_shape = np.array(volume.shape)
    zoomed_shape   = np.array(zoomed.shape)

    result = np.zeros_like(volume, dtype=np.float32)
    slices_orig: List = []
    slices_zoom: List = []

    for orig_len, zoom_len in zip(original_shape, zoomed_shape):
        if zoom_len >= orig_len:
            start = (zoom_len - orig_len) // 2
            slices_zoom.append(slice(start, start + orig_len))
            slices_orig.append(slice(None))
        else:
            start = (orig_len - zoom_len) // 2
            slices_orig.append(slice(start, start + zoom_len))
            slices_zoom.append(slice(None))

    result[tuple(slices_orig)] = zoomed[tuple(slices_zoom)]
    return result


def augment_volume(
    volume: np.ndarray,
    flip_axes: Optional[List[int]] = None,
    rotation_range: float = 0.0,
    zoom_range: float = 0.0,
) -> np.ndarray:
    """
    Apply a configurable chain of augmentations to a single 3-D volume.

    This is called **during training only**; pass ``augment=False`` for
    validation and inference.

    Args:
        volume:         3-D volume ``(D, H, W)``.
        flip_axes:      Axes to randomly flip.  ``None`` disables flipping.
        rotation_range: Max rotation degrees.  ``0`` disables rotation.
        zoom_range:     Max zoom fraction.      ``0`` disables zoom.

    Returns:
        Augmented volume of the same shape as *volume*.
    """
    vol = volume.copy()

    if flip_axes:
        vol = random_flip(vol, flip_axes)
    if rotation_range > 0:
        vol = random_rotate(vol, rotation_range)
    if zoom_range > 0:
        vol = random_zoom(vol, zoom_range)

    return vol
