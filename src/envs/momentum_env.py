from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.envs.multi_motion_env import MultiMotionTrackEnv
from src.rewards.reward_builder import RewardBuilder

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"


class MomentumEnv(MultiMotionTrackEnv):
    def __init__(self, npz_paths: list[str], reward_cfg: dict, phase_weight: float = 1.0,
                 seed: int | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.pin = PinocchioWrapper(URDF)
        self.reward_builder = RewardBuilder(reward_cfg)
        self.phase_weight = phase_weight
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 6,), np.float64)

    def _hg(self) -> np.ndarray:
        d = self.rt.data
        q = np.concatenate([d.qpos[0:3], d.qpos[3:7][[1, 2, 3, 0]], d.qpos[self.rt.act_qadr]])
        dq = np.concatenate([d.qvel[0:3], d.qvel[3:6], d.qvel[self.rt.act_vadr]])
        return self.pin.compute_centroidal_momentum(q, dq)

    def _augmented_obs(self, base_obs: np.ndarray, hg: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, hg])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        return self._augmented_obs(base_obs, self._hg()), info

    def step(self, action):
        prev_action = self._prev_action.copy()
        base_obs, _, terminated, truncated, base_info = super().step(action)
        hg = self._hg()
        q_now = self.rt.data.qpos[self.rt.act_qadr]
        q_ref_now = self._ref_joints(self._frame)

        reward, breakdown = self.reward_builder.compute(
            q=q_now, q_ref=q_ref_now,
            action=np.asarray(action, dtype=np.float64), prev_action=prev_action,
            hg_angular=hg[3:], phase_weight=self.phase_weight,
        )

        info = dict(base_info)
        info.update({"hg_angular_norm": float(np.linalg.norm(hg[3:])),
                     "reward_breakdown": breakdown})
        return self._augmented_obs(base_obs, hg), float(reward), terminated, truncated, info
