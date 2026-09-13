"""Counterfactual perturbation sampler for SCVC training: external impulse
(direction, magnitude, timing, via `rt.pelvis_body` / `xfrc_applied`) and
observation noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Perturbation:
    t_start: float
    duration: float
    force_x: float
    force_y: float
    obs_noise_std: float

    def active(self, t: float) -> bool:
        return self.t_start <= t < self.t_start + self.duration


def sample_perturbation(
    rng: np.random.Generator,
    episode_duration: float,
    force_range: tuple[float, float] = (0.0, 140.0),
    duration_range: tuple[float, float] = (0.05, 0.15),
    obs_noise_range: tuple[float, float] = (0.0, 0.02),
) -> Perturbation:
    """Sample one counterfactual perturbation, full 360 degree horizontal direction."""
    angle = rng.uniform(0, 2 * np.pi)
    magnitude = rng.uniform(*force_range)
    duration = rng.uniform(*duration_range)
    t_start = rng.uniform(0.05 * episode_duration, 0.85 * episode_duration)
    obs_noise_std = rng.uniform(*obs_noise_range)
    return Perturbation(
        t_start=t_start,
        duration=duration,
        force_x=magnitude * np.cos(angle),
        force_y=magnitude * np.sin(angle),
        obs_noise_std=obs_noise_std,
    )


def null_perturbation() -> Perturbation:
    """No disturbance - used to collect the nominal successor-entry manifold."""
    return Perturbation(t_start=1e9, duration=0.0, force_x=0.0, force_y=0.0, obs_noise_std=0.0)


@dataclass
class PushSegment:
    t_start: float
    duration: float
    force_x: float
    force_y: float

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.force_x, self.force_y))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.force_y, self.force_x)))


class PushPattern:
    """Sequence of push segments; duck-types Perturbation's
    (`.active(t)`, `.force_x`, `.force_y`, `.obs_noise_std`) interface."""

    def __init__(self, segments: list[PushSegment], obs_noise_std: float = 0.0):
        self.segments = segments
        self.obs_noise_std = obs_noise_std
        self._current: PushSegment | None = None

    def active(self, t: float) -> bool:
        self._current = next((s for s in self.segments if s.t_start <= t < s.t_start + s.duration), None)
        return self._current is not None

    @property
    def force_x(self) -> float:
        return self._current.force_x if self._current else 0.0

    @property
    def force_y(self) -> float:
        return self._current.force_y if self._current else 0.0

    @classmethod
    def from_polar(cls, pushes: list[tuple[float, float, float, float]]) -> "PushPattern":
        """pushes: list of (t_start, angle_deg, magnitude_N, duration_s)."""
        segs = [
            PushSegment(
                t_start=float(t), duration=float(dur),
                force_x=float(mag) * np.cos(np.radians(angle)),
                force_y=float(mag) * np.sin(np.radians(angle)),
            )
            for t, angle, mag, dur in pushes
        ]
        return cls(segs)
