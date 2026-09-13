from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.multi_motion_env import MultiMotionTrackEnv
from src.rewards.reward_builder import RewardBuilder


class FallEnv(MultiMotionTrackEnv):
    """G1's 29-DoF MJCF has no head body/geom - only torso_link. The
    'head' impact term is therefore always 0.0, reported as such rather than
    approximated from a nearby geom; 'torso' is the real collision-force
    signal from torso_link's collidable geom."""

    def __init__(self, npz_paths: list[str], reward_cfg: dict, seed: int | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        torso_body = self.rt.mujoco.mj_name2id(self.rt.model, self.rt.mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.torso_geoms = self.rt._collision_geoms_of_body(torso_body)
        self.reward_builder = RewardBuilder(reward_cfg)
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 2,), np.float64)

    def _switch_extra(self) -> np.ndarray:
        torso_force = self.rt.geom_group_force(self.torso_geoms)
        head_contact = False  # no head geom on this model
        return np.array([torso_force, float(head_contact)], dtype=np.float64)

    def _augmented_obs(self, base_obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, extra])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        return self._augmented_obs(base_obs, self._switch_extra()), info

    def step(self, action):
        base_obs, _, terminated, truncated, base_info = super().step(action)
        extra = self._switch_extra()
        torso_force = float(extra[0])

        reward, breakdown = self.reward_builder.compute(
            impact_forces={"torso": torso_force, "head": 0.0},
            head_contact=False,
        )

        info = dict(base_info)
        info.update({"torso_impact_force": torso_force, "reward_breakdown": breakdown})
        return self._augmented_obs(base_obs, extra), float(reward), terminated, truncated, info
