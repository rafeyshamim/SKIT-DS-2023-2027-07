"""
3D Reconstruction and visualization utilities.

Generates visual representations of preprocessed medical volumes:

    1. **Orthogonal slices** — axial, coronal, sagittal (matplotlib PNG)
    2. **Maximum Intensity Projection (MIP)** — along all 3 axes (matplotlib PNG)
    3. **3D isosurface** — marching cubes mesh via scikit-image,
       rendered as an interactive Plotly HTML file.

Input format expected:
    ``np.ndarray`` of shape ``(D, H, W)`` or ``(D, H, W, 1)``
    dtype ``float32``, values in ``[0, 1]``.
"""
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VolumeReconstructor:
    """
    Generates 2-D and 3-D reconstructions from a preprocessed medical volume.

    Usage::

        recon = VolumeReconstructor("results/reconstruction/")
        recon.generate_all_visualizations(volume, prefix="patient_001")
    """

    def __init__(self, output_dir: str = "results/reconstruction/") -> None:
        """
        Args:
            output_dir: Directory where all output files are saved.
                        Created automatically if it does not exist.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _squeeze(volume: np.ndarray) -> np.ndarray:
        """Strip a trailing channel dimension if present."""
        if volume.ndim == 4 and volume.shape[-1] == 1:
            return volume[..., 0]
        if volume.ndim == 3:
            return volume
        raise ValueError(
            f"Unexpected volume shape: {volume.shape}. "
            "Expected (D, H, W) or (D, H, W, 1)."
        )

    # ------------------------------------------------------------------
    # Orthogonal slices
    # ------------------------------------------------------------------

    def plot_orthogonal_slices(
        self,
        volume: np.ndarray,
        title: str = "Orthogonal Slices",
        filename: str = "orthogonal_slices.png",
    ) -> str:
        """
        Plot the three central orthogonal slices: axial, coronal, sagittal.

        Args:
            volume:   Volume of shape ``(D, H, W)`` or ``(D, H, W, 1)``.
            title:    Figure super-title.
            filename: Output PNG filename within ``output_dir``.

        Returns:
            Absolute path of the saved image.
        """
        vol = self._squeeze(volume)
        d, h, w = vol.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(vol[mid_d, :, :], cmap="gray", origin="lower")
        axes[0].set_title(f"Axial  (z={mid_d})")
        axes[0].axis("off")

        axes[1].imshow(vol[:, mid_h, :], cmap="gray", origin="lower")
        axes[1].set_title(f"Coronal (y={mid_h})")
        axes[1].axis("off")

        axes[2].imshow(vol[:, :, mid_w], cmap="gray", origin="lower")
        axes[2].set_title(f"Sagittal (x={mid_w})")
        axes[2].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Orthogonal slices → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Maximum Intensity Projection
    # ------------------------------------------------------------------

    def plot_mip(
        self,
        volume: np.ndarray,
        title: str = "Maximum Intensity Projection",
        filename: str = "mip.png",
    ) -> str:
        """
        Compute and plot Maximum Intensity Projections along all 3 axes.

        Args:
            volume:   Volume of shape ``(D, H, W)`` or ``(D, H, W, 1)``.
            title:    Figure super-title.
            filename: Output PNG filename.

        Returns:
            Absolute path of the saved image.
        """
        vol = self._squeeze(volume)

        labels      = ["MIP — Axial (Z-axis)", "MIP — Coronal (Y-axis)", "MIP — Sagittal (X-axis)"]
        proj_axes   = [0, 1, 2]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, axis, label in zip(axes, proj_axes, labels):
            mip = np.max(vol, axis=axis)
            ax.imshow(mip, cmap="hot", origin="lower")
            ax.set_title(label)
            ax.axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("MIP → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # 3D Isosurface (marching cubes + Plotly)
    # ------------------------------------------------------------------

    def extract_isosurface(
        self,
        volume: np.ndarray,
        level: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract an isosurface mesh using the marching cubes algorithm.

        Args:
            volume: Volume of shape ``(D, H, W)`` or ``(D, H, W, 1)``.
            level:  Isosurface threshold.  Defaults to 50 % of max intensity.

        Returns:
            Tuple ``(vertices, faces, normals)``.

        Raises:
            ImportError: If ``scikit-image`` is not installed.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError(
                "scikit-image is required for isosurface extraction.\n"
                "Install with:  pip install scikit-image"
            ) from exc

        vol = self._squeeze(volume)
        if level is None:
            level = float(vol.max() * 0.5)

        logger.info(
            "Marching cubes | level=%.4f | volume=%s", level, vol.shape
        )
        vertices, faces, normals, _ = marching_cubes(vol, level=level)
        logger.info(
            "Isosurface: %d vertices, %d faces", len(vertices), len(faces)
        )
        return vertices, faces, normals

    def plot_3d_isosurface(
        self,
        volume: np.ndarray,
        level: Optional[float] = None,
        title: str = "3D Isosurface Reconstruction",
        filename: str = "isosurface_3d.html",
    ) -> str:
        """
        Generate an interactive 3-D isosurface using Plotly and save as HTML.

        Args:
            volume:   Volume of shape ``(D, H, W)`` or ``(D, H, W, 1)``.
            level:    Isosurface threshold.  Defaults to 50 % of max.
            title:    Plot title.
            filename: Output HTML filename.

        Returns:
            Absolute path of the saved HTML file.

        Raises:
            ImportError: If ``plotly`` is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(
                "Plotly is required for 3D visualization.\n"
                "Install with:  pip install plotly"
            ) from exc

        vertices, faces, _ = self.extract_isosurface(volume, level=level)
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0],   faces[:, 1],   faces[:, 2]

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=0.75,
                    colorscale="Viridis",
                    intensity=z,
                    showscale=True,
                    colorbar_title="Depth",
                )
            ]
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title="Depth",
                yaxis_title="Height",
                zaxis_title="Width",
            ),
        )

        save_path = os.path.join(self.output_dir, filename)
        fig.write_html(save_path)
        logger.info("3D isosurface → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def generate_all_visualizations(
        self,
        volume: np.ndarray,
        prefix: str = "sample",
    ) -> Dict[str, str]:
        """
        Generate all available visualizations for *volume*.

        Args:
            volume: Preprocessed volume ``(D, H, W)`` or ``(D, H, W, 1)``.
            prefix: Filename prefix for all output files.

        Returns:
            Dictionary mapping visualization type → saved file path.
        """
        results: Dict[str, str] = {}

        results["orthogonal_slices"] = self.plot_orthogonal_slices(
            volume,
            title=f"Orthogonal Slices — {prefix}",
            filename=f"{prefix}_orthogonal_slices.png",
        )

        results["mip"] = self.plot_mip(
            volume,
            title=f"MIP — {prefix}",
            filename=f"{prefix}_mip.png",
        )

        try:
            results["isosurface_3d"] = self.plot_3d_isosurface(
                volume,
                title=f"3D Isosurface — {prefix}",
                filename=f"{prefix}_isosurface_3d.html",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("3D isosurface skipped: %s", exc)

        return results
