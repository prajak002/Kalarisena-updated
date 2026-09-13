from __future__ import annotations

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rewards.reward_builder import RewardBuilder
from src.sim.conventions import quat_wxyz_to_matrix
from src.sim.mujoco_runtime import G1MujocoRuntime

ACTION_SCALE = 0.5
MAX_STEPS = 300
UPRIGHT_SUCCESS = 0.75
UPRIGHT_DWELL_STEPS = 30


class RecoveryEnv(gym.Env):
    """Recovery-to-standing from a randomized fallen pose; no reference
    motion, so this does not extend KalariTrackEnv."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, reward_cfg: dict, seed: int | None = None, render_mode: str | None = None):
        super().__init__()
        self.rt = G1MujocoRuntime()
        self.reward_builder = RewardBuilder(reward_cfg)
        self.render_mode = render_mode
        self.nu = self.rt.model.nu
        self.target = self.rt.default_joint_targets()

        self.jnt_lo = np.zeros(self.nu)
        self.jnt_hi = np.zeros(self.nu)
        for a in range(self.nu):
            jid = int(self.rt.model.actuator_trnid[a, 0])
            self.jnt_lo[a], self.jnt_hi[a] = self.rt.model.jnt_range[jid]

        obs_dim = 29 + 29 + 3 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)

        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._upright_dwell = 0
        self._prev_action = np.zeros(self.nu)

    def _fallen_start(self) -> None:
        qpos = self.rt.default_qpos()
        angle = self._rng.uniform(np.pi / 2, np.pi)
        axis = self._rng.normal(size=3)
        axis[2] = 0.0
        axis /= max(np.linalg.norm(axis), 1e-6)
        half = angle / 2.0
        tip_quat = np.array([np.cos(half), *(np.sin(half) * axis)])

        w0, x0, y0, z0 = qpos[3:7]
        w1, x1, y1, z1 = tip_quat
        qpos[3] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        qpos[4] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        qpos[5] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        qpos[6] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        qpos[2] = 0.9

        self.rt.data.qpos[:] = qpos
        self.rt.data.qvel[:] = 0.0
        self.rt.data.qacc[:] = 0.0
        self.rt.data.xfrc_applied[:] = 0.0
        self.rt.mujoco.mj_forward(self.rt.model, self.rt.data)
        self.rt.settle(0.6, q_cmd=self.target)

    def _obs(self) -> np.ndarray:
        d = self.rt.data
        q = d.qpos[self.rt.act_qadr]
        dq = d.qvel[self.rt.act_vadr] * 0.1
        rot = quat_wxyz_to_matrix(d.qpos[3:7])
        gravity_b = rot.T @ np.array([0.0, 0.0, -1.0])
        angvel_b = d.qvel[3:6] * 0.2
        upright = np.array([self.rt.torso_upright_cos()])
        return np.concatenate([q, dq, gravity_b, angvel_b, upright])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._fallen_start()
        self._step = 0
        self._upright_dwell = 0
        self._prev_action = np.zeros(self.nu)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_cmd = np.clip(self.target + ACTION_SCALE * action, self.jnt_lo, self.jnt_hi)
        self.rt.control_step(q_cmd)
        self._step += 1

        upright = self.rt.torso_upright_cos()
        is_upright = upright > UPRIGHT_SUCCESS
        self._upright_dwell = self._upright_dwell + 1 if is_upright else 0

        success = self._upright_dwell >= UPRIGHT_DWELL_STEPS
        timeout = self._step >= MAX_STEPS
        terminated = bool(success)
        truncated = bool(timeout)

        reward, breakdown = self.reward_builder.compute(
            is_upright=is_upright, step=self._step,
            upright=upright, base_height=float(self.rt.base_height))

        info = {"upright": float(upright), "is_upright": bool(is_upright),
                 "success": bool(success), "step": self._step,
                 "base_height": float(self.rt.base_height), "reward_breakdown": breakdown}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.rt.render_frame(camera="track")
        return None

    def close(self):
        self.rt.close()
