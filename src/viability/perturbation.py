"""Counterfactual perturbation sampler for SCVC training (paper Sec 3.3, D_pert).

The paper's D_pert covers external impulse, friction, mass/inertia, latency,
observation noise, contact timing, and phase. This implementation covers the
two dimensions that don't require reloading the MuJoCo model per trial:
external impulse (direction, magnitude, timing - reusing the exact
`rt.pelvis_body` / `xfrc_applied` mechanism scripts/sim_push_sweep.py already
uses) and observation noise. Friction/mass/inertia/latency randomization would
need per-episode model reconstruction, which G1MujocoRuntime does not support
live; that is a real, stated scope limit, not something quietly skipped.
"""

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
