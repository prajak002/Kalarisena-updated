"""eta(s): the normalized feature vector SCVC's manifold distance and critic
both operate on (paper Sec 3.2-3.3). Built from quantities every other stage
in this repo already computes - no new physics primitives.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "joint_mse", "upright", "base_height", "phase",
    "com_margin", "cp_margin", "support_area", "com_vx", "com_vy",
]


def eta(info: dict, feats: dict, phase: float, base_height: float, com_vel_xy: tuple[float, float]) -> np.ndarray:
    return np.array([
        info.get("joint_mse", 0.0),
        info.get("upright", 1.0),
        base_height,
        phase,
        feats.get("com_margin", -999.0),
        feats.get("cp_margin", -999.0),
        feats.get("support_area", 0.0),
        com_vel_xy[0],
        com_vel_xy[1],
    ], dtype=np.float64)


def feature_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) fit on training features, std floored to avoid div/0."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-3)
    return mean, std
