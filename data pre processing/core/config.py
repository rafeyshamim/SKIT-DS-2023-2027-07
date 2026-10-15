"""
Configuration and CT Window Presets for Medical Image Preprocessing
"""

from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass(frozen=True)
class WindowPreset:
    name: str
    window_center: float  # Window Level (WL)
    window_width: float   # Window Width (WW)

    @property
    def window_min(self) -> float:
        return self.window_center - (self.window_width / 2.0)

    @property
    def window_max(self) -> float:
        return self.window_center + (self.window_width / 2.0)


class CTWindowPresets:
    """Standard clinical CT window presets in Hounsfield Units (HU)"""
    BRAIN = WindowPreset(name="Brain", window_center=40.0, window_width=80.0)
    SUBDURAL = WindowPreset(name="Subdural", window_center=75.0, window_width=150.0)
    STROKE = WindowPreset(name="Stroke", window_center=32.0, window_width=8.0)
    BONE = WindowPreset(name="Bone", window_center=400.0, window_width=1800.0)
    SOFT_TISSUE = WindowPreset(name="Soft Tissue", window_center=40.0, window_width=400.0)
    LIVER = WindowPreset(name="Liver", window_center=60.0, window_width=160.0)
    LUNG = WindowPreset(name="Lung", window_center=-600.0, window_width=1500.0)
    MEDIASTINUM = WindowPreset(name="Mediastinum", window_center=50.0, window_width=350.0)

    _REGISTRY: Dict[str, WindowPreset] = {
        "brain": BRAIN,
        "subdural": SUBDURAL,
        "stroke": STROKE,
        "bone": BONE,
        "soft_tissue": SOFT_TISSUE,
        "liver": LIVER,
        "lung": LUNG,
        "mediastinum": MEDIASTINUM
    }

    @classmethod
    def get(cls, name: str) -> WindowPreset:
        key = name.lower().replace(" ", "_")
        if key not in cls._REGISTRY:
            raise KeyError(f"Window preset '{name}' not found. Available: {list(cls._REGISTRY.keys())}")
        return cls._REGISTRY[key]


# Reconstruction configuration
DEFAULT_ISOTROPIC_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (x, y, z) in mm
DEFAULT_SMOOTHING_ITERATIONS: int = 15
DEFAULT_SMOOTHING_RELAXATION: float = 0.2
DEFAULT_DECIMATION_TARGET_REDUCTION: float = 0.5  # 50% face reduction
