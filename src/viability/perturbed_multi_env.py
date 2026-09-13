from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.envs.kalari_track_env import ACTION_SCALE, FALL_DUP, FALL_DZ, W_JOINT, W_ROOTZ, W_SMOOTH, W_UPRIGHT, \
    JOINT_ERR_SCALE, ROOTZ_ERR_SCALE
from src.envs.multi_motion_env import MultiMotionTrackEnv
from src.viability.perturbation import Perturbation, null_perturbation

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"


class PerturbedMultiMotionTrackEnv(MultiMotionTrackEnv):
    def __init__(self, npz_paths: list[str], seed: int | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.pin = PinocchioWrapper(URDF)
        self._pert: Perturbation = null_perturbation()
        self._t = 0.0
        self._dt = 1.0 / self.rt.CONTROL_HZ
        self._xfrc = np.zeros_like(self.rt.data.xfrc_applied)

    def set_perturbation(self, pert: Perturbation) -> None:
        self._pert = pert

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._t = 0.0
        return self._noisy(obs), info

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        if self._pert.obs_noise_std > 0:
            obs = obs + self._rng.normal(0, self._pert.obs_noise_std, size=obs.shape)
        return obs

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_cmd = np.clip(self._ref_joints(self._frame) + ACTION_SCALE * action,
                         self.jnt_lo, self.jnt_hi)

        self._xfrc[:] = 0.0
        if self._pert.active(self._t):
            self._xfrc[self.rt.pelvis_body, 0] = self._pert.force_x
            self._xfrc[self.rt.pelvis_body, 1] = self._pert.force_y
        self.rt.control_step(q_cmd, xfrc=self._xfrc)
        self._t += self._dt
        self._frame += 1

        q = self.rt.data.qpos[self.rt.act_qadr]
        q_ref = self._ref_joints(self._frame)
        mse = float(np.mean((q - q_ref) ** 2))
        r_joint = np.exp(-JOINT_ERR_SCALE * mse)

        z = self.rt.base_height
        z_ref = float(self.ref["root_pos"][min(self._frame, self.n_frames - 1), 2])
        r_rootz = np.exp(-ROOTZ_ERR_SCALE * (z - z_ref) ** 2)

        upright = self.rt.torso_upright_cos()
        r_smooth = -float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action

        reward = (W_JOINT * r_joint + W_ROOTZ * r_rootz
                  + W_UPRIGHT * max(upright, 0.0) + W_SMOOTH * r_smooth)

        k = min(self._frame, self.n_frames - 1)
        fell = (z < self.ref_z[k] - FALL_DZ) or (upright < self.ref_up[k] - FALL_DUP)
        terminated = bool(fell)
        truncated = bool(self._frame >= self.n_frames - 1)
        info = {"joint_mse": mse, "upright": upright, "fell": fell,
                "r_joint": r_joint, "r_rootz": r_rootz, "frame": self._frame,
                "phase": k / max(self.n_frames - 1, 1), "motion_id": self.ref["motion_id"]}
        return self._noisy(self._obs()), float(reward), terminated, truncated, info

    def physics_features(self) -> dict:
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
