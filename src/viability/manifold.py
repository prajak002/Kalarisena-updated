"""Successor-entry manifold Omega(g): "the intended continuation g+" is
instantiated as "reach the final phase of this same motion safely", not a
distinct next skill."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class SuccessorManifold:
    def __init__(self, k: int = 20, weights: np.ndarray | None = None):
        self.k = k
        self.weights = weights
        self._tree: cKDTree | None = None
        self._points: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "SuccessorManifold":
        pts = features if self.weights is None else features * self.weights[None, :]
        self._points = pts
        self._tree = cKDTree(pts)
        return self

    def distance(self, feature: np.ndarray) -> float:
        assert self._tree is not None, "call fit() first"
        pt = feature if self.weights is None else feature * self.weights
        k = min(self.k, self._points.shape[0])
        d, _ = self._tree.query(pt, k=k)
        return float(np.mean(d)) if k > 1 else float(d)

    def distances(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.distance(f) for f in features])
