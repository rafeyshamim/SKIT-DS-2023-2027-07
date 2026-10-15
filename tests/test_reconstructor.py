"""
Unit tests for VolumeReconstructor module.
"""
import os
import numpy as np
import pytest

from src.reconstruction.reconstructor import VolumeReconstructor


class TestVolumeReconstructor:
    """Test suite for 2D/3D visualization and reconstruction."""

    def test_squeeze_dimensions(self):
        recon = VolumeReconstructor()
        vol3d = np.zeros((10, 10, 10), dtype=np.float32)
        assert recon._squeeze(vol3d).shape == (10, 10, 10)

        vol4d = np.zeros((10, 10, 10, 1), dtype=np.float32)
        assert recon._squeeze(vol4d).shape == (10, 10, 10)

        with pytest.raises(ValueError):
            recon._squeeze(np.zeros((10, 10)))

    def test_plot_orthogonal_slices(self, random_volume_3d, tmp_path):
        recon = VolumeReconstructor(output_dir=str(tmp_path))
        path = recon.plot_orthogonal_slices(random_volume_3d, filename="slices.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_plot_mip(self, random_volume_3d, tmp_path):
        recon = VolumeReconstructor(output_dir=str(tmp_path))
        path = recon.plot_mip(random_volume_3d, filename="mip.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_extract_isosurface(self, tmp_path):
        recon = VolumeReconstructor(output_dir=str(tmp_path))
        # Create a sphere in a 3D grid
        grid = np.zeros((20, 20, 20), dtype=np.float32)
        z, y, x = np.ogrid[:20, :20, :20]
        mask = (x - 10)**2 + (y - 10)**2 + (z - 10)**2 <= 5**2
        grid[mask] = 1.0

        vertices, faces, normals = recon.extract_isosurface(grid, level=0.5)
        assert len(vertices) > 0
        assert len(faces) > 0
        assert len(normals) > 0
