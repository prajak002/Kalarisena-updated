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


class ComRefineEnv(MultiMotionTrackEnv):
    def __init__(self, npz_paths: list[str], reward_cfg: dict, seed: int | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.pin = PinocchioWrapper(URDF)
        self.reward_builder = RewardBuilder(reward_cfg)
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 6,), np.float64)

    def _physics_features(self) -> dict:
        d = self.rt.data
        q = np.concatenate([d.qpos[0:3], d.qpos[3:7][[1, 2, 3, 0]], d.qpos[self.rt.act_qadr]])
        dq = np.concatenate([d.qvel[0:3], d.qvel[3:6], d.qvel[self.rt.act_vadr]])
        left_c, right_c = self._foot_contacts()
        return self.pin.get_support_features(q, dq, np.array([left_c, right_c]))

    def _foot_contacts(self, z_thresh: float = 0.05) -> tuple[bool, bool]:
        rt = self.rt
        heights = {g: float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                   for g in rt.left_foot_geoms + rt.right_foot_geoms}
        left_h = min(heights[g] for g in rt.left_foot_geoms)
        right_h = min(heights[g] for g in rt.right_foot_geoms)
        return left_h < z_thresh, right_h < z_thresh

    _SUPPORT_MODE_CODE = {"double": 1.0, "single_left": 0.5, "single_right": -0.5, "no_contact": 0.0}

    def _augmented_obs(self, base_obs: np.ndarray, feats: dict) -> np.ndarray:
        has_support = feats["support_area"] > 0
        # get_support_features returns -999.0 sentinels for com_margin/cp_margin
        # when there's no contact (src/dynamics/pinocchio_wrapper.py) - the
        # reward path already guards against feeding that into training
        # (com_refine_env.step's `if has_support else 0.0`), but this
        # observation path didn't: the policy was seeing a raw -999 in its
        # input whenever contact was briefly lost, exactly when it most needs
        # a clean signal rather than an extreme out-of-distribution spike.
        com_margin = feats["com_margin"] if has_support else 0.0
        cp_margin = feats["cp_margin"] if has_support else 0.0
        support_center = feats["support_center"]
        support_center = np.nan_to_num(support_center, nan=0.0)
        mode_code = self._SUPPORT_MODE_CODE.get(feats.get("support_mode"), 0.0)
        extra = np.array([
            com_margin, cp_margin, feats["support_area"],
            *support_center, mode_code,
        ], dtype=np.float64)
        return np.concatenate([base_obs, extra])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        feats = self._physics_features()
        return self._augmented_obs(base_obs, feats), info

    def step(self, action):
        prev_action = self._prev_action.copy()
        base_obs, _, terminated, truncated, base_info = super().step(action)
        feats = self._physics_features()
        q_now = self.rt.data.qpos[self.rt.act_qadr]
        q_ref_now = self._ref_joints(self._frame)

        has_support = feats["support_area"] > 0
        com_margin = feats["com_margin"] if has_support else 0.0
        cp_margin = feats["cp_margin"] if has_support else 0.0
        reward, breakdown = self.reward_builder.compute(
            q=q_now, q_ref=q_ref_now,
            action=np.asarray(action, dtype=np.float64), prev_action=prev_action,
            com_margin=com_margin, cp_margin=cp_margin,
        )

        info = dict(base_info)
        info.update({"com_margin": com_margin, "cp_margin": cp_margin,
                     "has_support": bool(has_support), "reward_breakdown": breakdown})
        return self._augmented_obs(base_obs, feats), float(reward), terminated, truncated, info
