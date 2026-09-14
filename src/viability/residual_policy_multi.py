from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.viability.features import eta
from src.viability.perturbed_multi_env import PerturbedMultiMotionTrackEnv
from src.viability.residual_policy import (
    RESIDUAL_ACTION_SCALE, TAU_H, TAU_L, W_BALANCE, W_CONTACT, W_DELTA, W_SKILL_DEV, W_SUCC,
    W_TRACK, W_VIA, ViabilityGate, gate,
)


class IntentPreservingResidualMultiEnv(PerturbedMultiMotionTrackEnv):
    def __init__(self, npz_paths: list[str], tracker_policy, critic_path: str,
                 seed: int | None = None, tau_l: float | list[float] = TAU_L,
                 tau_h: float | list[float] = TAU_H,
                 force_gate: float | None = None, **kwargs):
        super().__init__(npz_paths, seed=seed, **kwargs)
        self.tracker_policy = tracker_policy
        self.gate_fn = ViabilityGate(critic_path)
        # Per-motion thresholds when a list is given (len == len(npz_paths)) -
        # required in practice: the critic's raw output scale varies by up to
        # 10 orders of magnitude across motions (see
        # scripts/calibrate_residual_gate.py's per-motion output), so one
        # global (tau_l, tau_h) pair pins the gate stuck open or closed for
        # most motions rather than actually gating.
        self.tau_l_by_motion = list(tau_l) if isinstance(tau_l, (list, tuple)) else [tau_l] * len(npz_paths)
        self.tau_h_by_motion = list(tau_h) if isinstance(tau_h, (list, tuple)) else [tau_h] * len(npz_paths)
        self.force_gate = force_gate
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_dim + 5,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        self._last_extra = np.zeros(5)
        self._last_base_obs = None

    def _augmented_obs(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, self._last_extra])

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        self.gate_fn.reset()
        self._last_base_obs = base_obs
        self._last_extra = np.zeros(5)
        return self._augmented_obs(base_obs), info

    def _viability_step(self, info_prev: dict):
        feats = self.physics_features()
        phase = self._frame / max(self.n_frames - 1, 1)
        eta_vec = eta(info_prev, feats, phase, self.rt.base_height, (0.0, 0.0))
        v_raw, v_bar = self.gate_fn.update(eta_vec)
        tau_l = self.tau_l_by_motion[self._active_motion_idx]
        tau_h = self.tau_h_by_motion[self._active_motion_idx]
        g = gate(v_bar, tau_l, tau_h) if self.force_gate is None else self.force_gate
        return v_raw, v_bar, g, feats

    def step(self, delta_a):
        delta_a = np.clip(np.asarray(delta_a, dtype=np.float64), -1.0, 1.0)

        info_prev = {"joint_mse": float(np.mean(
            (self.rt.data.qpos[self.rt.act_qadr] - self._ref_joints(self._frame)) ** 2)),
            "upright": self.rt.torso_upright_cos()}
        v_raw, v_bar, g, feats_prev = self._viability_step(info_prev)

        a0, _ = self.tracker_policy.predict(self._last_base_obs, deterministic=True)
        a0 = np.asarray(a0, dtype=np.float64)
        residual = RESIDUAL_ACTION_SCALE * delta_a
        a_deploy = a0 + g * residual

        base_obs, _, terminated, truncated, base_info = super().step(a_deploy)
        self._last_base_obs = base_obs

        contact_l, contact_r = self._foot_contacts()
        r_contact = float(contact_l or contact_r)

        cp_margin = float(feats_prev.get("cp_margin", -1.0))
        r_balance = -max(0.0, 0.05 - cp_margin) ** 2

        r_succ = float(truncated and not base_info["fell"])
        r_via = v_raw

        skill_dev = base_info["joint_mse"]
        r_track = base_info["r_joint"]

        reward = (W_TRACK * r_track + W_CONTACT * r_contact + W_BALANCE * r_balance
                  + W_SUCC * r_succ + W_VIA * r_via
                  - W_SKILL_DEV * skill_dev - W_DELTA * float(np.sum(residual ** 2)))

        self._last_extra = np.array([v_raw, v_bar, g, cp_margin,
                                      feats_prev.get("com_margin", -1.0)])
        info = dict(base_info)
        info.update({"v_raw": v_raw, "v_bar": v_bar, "gate": g,
                     "r_contact": r_contact, "r_balance": r_balance,
                     "r_succ": r_succ, "r_via": r_via})
        return self._augmented_obs(base_obs), float(reward), terminated, truncated, info
