"""Stage A training environment: reference-motion tracking on the G1 (paper 6.9/6.10).

Gymnasium environment implementing the paper's Stage A contract:

  Observation (section 6.8 blocks A+B, Stage A uses proprioception + reference):
      joint angles (29), joint velocities (29, scaled),
      gravity vector in base frame (3), base angular velocity (3, scaled),
      reference joints current (29) and next frame (29), phase sin/cos (2)
      -> 124 dims

  Action (section 6.9, joint target residuals):
      a in [-1,1]^29,  q_cmd = q_ref(t) + s_a * a,  clipped to joint limits,
      then the existing 500 Hz PD layer (tau = kp e - kd qdot) executes it.

  Reward (section 6.12 Stage A tracking terms, via src/rewards/reward_builder
  term semantics): joint tracking, root-height tracking, upright bonus,
  action smoothness penalty. Termination on fall (base_z or upright below
  threshold), truncation at motion end.

References come from the ground-corrected NPZ library in
data/motions_retargeted/ (contract of scripts/annotate_motion_library.py).
Episodes start at a random reference frame, with the robot initialised ON the
reference state, so early training is not dominated by getting up.

This is deliberately the paper's minimal Stage A: no CoM/momentum blocks, no
fall/recovery routing. Those are Stages B-F.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium as gym
from gymnasium import spaces

from src.sim.conventions import quat_wxyz_to_matrix, quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime

VEL_SCALE = 0.1
ANGVEL_SCALE = 0.2
ACTION_SCALE = 0.25          # rad of residual authority (paper: s_a)
# Fall detection is RELATIVE to the reference: deep Kalari postures take the
# base to z~0.27 m and torso upright-cos to ~0.47 legitimately, so absolute
# thresholds terminate healthy tracking (a seated vadivu "fell" at step 1 with
# joint mse 4e-4 under absolute cutoffs).
FALL_DZ = 0.15               # base more than this below the reference height
FALL_DUP = 0.45              # upright-cos more than this below the reference
W_JOINT = 0.60               # reward weights, stage A tracking mix
W_ROOTZ = 0.20
W_UPRIGHT = 0.10
W_SMOOTH = 0.02
JOINT_ERR_SCALE = 4.0        # r_joint = exp(-scale * mean_sq_err)
ROOTZ_ERR_SCALE = 40.0


def load_reference(npz_path: str) -> dict:
    """Load a ground-corrected NPZ reference (annotate_motion_library contract)."""
    d = np.load(npz_path, allow_pickle=True)
    cols = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
            for c in np.asarray(d["joint_cols"]).reshape(-1)]
    return {
        "motion_id": str(np.asarray(d["motion_id"]).reshape(-1)[0].decode()
                         if "motion_id" in d.files else os.path.basename(npz_path)[:-4]),
        "joint_names": [c.removesuffix("_dof") for c in cols],
        "joint_pos": np.asarray(d["joint_pos"], dtype=np.float64),
        "root_pos": np.asarray(d["root_pos"], dtype=np.float64),
        "root_quat_xyzw": np.asarray(d["root_quat_xyzw"], dtype=np.float64),
        "fps": float(np.asarray(d["fps"]).reshape(-1)[0]),
    }


class KalariTrackEnv(gym.Env):
    """Single-motion tracking environment (Stage A)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, npz_path: str, seed: int | None = None,
                 max_start_frac: float = 0.7, render_mode: str | None = None):
        super().__init__()
        self.rt = G1MujocoRuntime()
        self.ref = load_reference(npz_path)
        self.render_mode = render_mode

        idx = {n: i for i, n in enumerate(self.ref["joint_names"])}
        self.col_for_act = np.array([idx.get(j, -1) for j in self.rt.act_joint_names])
        if (self.col_for_act < 0).any():
            missing = [j for j, c in zip(self.rt.act_joint_names, self.col_for_act) if c < 0]
            raise ValueError(f"reference missing joints: {missing}")

        self.n_frames = self.ref["joint_pos"].shape[0]
        self.ref_z = self.ref["root_pos"][:, 2]
        self.ref_up = np.array([
            quat_wxyz_to_matrix(quat_xyzw_to_wxyz(q))[2, 2]
            for q in self.ref["root_quat_xyzw"]])
        self.max_start = int(self.n_frames * max_start_frac)
        self.nu = self.rt.model.nu

        self.jnt_lo = np.zeros(self.nu)
        self.jnt_hi = np.zeros(self.nu)
        for a in range(self.nu):
            jid = int(self.rt.model.actuator_trnid[a, 0])
            self.jnt_lo[a], self.jnt_hi[a] = self.rt.model.jnt_range[jid]

        obs_dim = 29 + 29 + 3 + 3 + 29 + 29 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)

        self._rng = np.random.default_rng(seed)
        self._frame = 0
        self._prev_action = np.zeros(self.nu)

    # ------------------------------------------------------------------ ref --
    def _ref_joints(self, frame: int) -> np.ndarray:
        frame = min(frame, self.n_frames - 1)
        return self.ref["joint_pos"][frame][self.col_for_act]

    def _set_state_to_reference(self, frame: int) -> None:
        qpos = self.rt.default_qpos()
        qpos[0:3] = self.ref["root_pos"][frame]
        qpos[3:7] = quat_xyzw_to_wxyz(self.ref["root_quat_xyzw"][frame])
        qpos[self.rt.act_qadr] = self._ref_joints(frame)
        self.rt.data.qpos[:] = qpos
        self.rt.data.qvel[:] = 0.0
        self.rt.data.qacc[:] = 0.0
        self.rt.data.xfrc_applied[:] = 0.0
        self.rt.mujoco.mj_forward(self.rt.model, self.rt.data)

    # ------------------------------------------------------------------ obs --
    def _obs(self) -> np.ndarray:
        d = self.rt.data
        q = d.qpos[self.rt.act_qadr]
        dq = d.qvel[self.rt.act_vadr] * VEL_SCALE
        rot = quat_wxyz_to_matrix(d.qpos[3:7])
        gravity_b = rot.T @ np.array([0.0, 0.0, -1.0])
        angvel_b = d.qvel[3:6] * ANGVEL_SCALE
        phase = self._frame / max(self.n_frames - 1, 1)
        return np.concatenate([
            q, dq, gravity_b, angvel_b,
            self._ref_joints(self._frame),
            self._ref_joints(self._frame + 1),
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ])

    # ------------------------------------------------------------------ api --
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._frame = int(self._rng.integers(0, self.max_start + 1))
        self._set_state_to_reference(self._frame)
        self._prev_action = np.zeros(self.nu)
        return self._obs(), {"motion_id": self.ref["motion_id"], "start_frame": self._frame}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_cmd = np.clip(self._ref_joints(self._frame) + ACTION_SCALE * action,
                        self.jnt_lo, self.jnt_hi)
        self.rt.control_step(q_cmd)
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
                "r_joint": r_joint, "r_rootz": r_rootz}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.rt.render_frame(camera="track")
        return None

    def close(self):
        self.rt.close()
