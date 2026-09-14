"""Intent-preserving residual controller: pi_theta(s_t, z_t, g_t+, V_t) -> delta_a_t.

a_t^deploy = a_t^0 + g(V_bar_t) * delta_a_t, g(V) = clip((tau_h-V)/(tau_h-tau_l), 0, 1)
V_bar_t = beta * V_bar_{t-1} + (1-beta) * V_psi(eta(s_t))

No separate safe-recovery controller exists yet, so a_t^safe collapses to
the frozen tracker's own action (g -> 0).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gymnasium import spaces

from src.envs.kalari_track_env import ACTION_SCALE
from src.viability.critic import ViabilityCritic
from src.viability.features import eta
from src.viability.perturbation import Perturbation, null_perturbation
from src.viability.perturbed_env import PerturbedTrackEnv

RESIDUAL_ACTION_SCALE = 0.15
TAU_L = 0.25
TAU_H = 0.70
EMA_BETA = 0.90

W_TRACK = 0.35
W_CONTACT = 0.15
W_BALANCE = 0.15
W_SUCC = 0.10
W_VIA = 0.15
W_SKILL_DEV = 0.05
W_DELTA = 0.05


def gate(v_bar: float, tau_l: float = TAU_L, tau_h: float = TAU_H) -> float:
    if tau_h <= tau_l:
        return 0.0
    return float(np.clip((tau_h - v_bar) / (tau_h - tau_l), 0.0, 1.0))


class ViabilityGate:
    def __init__(self, critic_path: str, beta: float = EMA_BETA):
        ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
        self.critic = ViabilityCritic(in_dim=ckpt["in_dim"])
        self.critic.load_state_dict(ckpt["state_dict"])
        self.critic.eval()
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.beta = beta
        self.v_bar: float | None = None

    def reset(self) -> None:
        # None (not 1.0): this critic's raw output is often 10+ orders of
        # magnitude below 1.0 (see scripts/calibrate_residual_gate.py's
        # per-motion output), so bootstrapping the EMA at a fixed prior of
        # 1.0 meant v_bar took far longer to decay to the relevant scale
        # than a typical (~40-step) episode lasts - the gate never opened
        # at all. Seed with the first real reading instead.
        self.v_bar = None

    def update(self, feat_vec: np.ndarray) -> tuple[float, float]:
        x = (feat_vec - self.mean) / self.std
        with torch.no_grad():
            v_raw = float(self.critic(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item())
        self.v_bar = v_raw if self.v_bar is None else self.beta * self.v_bar + (1 - self.beta) * v_raw
        return v_raw, self.v_bar


class IntentPreservingResidualEnv(PerturbedTrackEnv):
    def __init__(self, npz_path: str, tracker_policy, critic_path: str,
                 seed: int | None = None, tau_l: float = TAU_L, tau_h: float = TAU_H,
                 force_gate: float | None = None, **kwargs):
        super().__init__(npz_path, seed=seed, **kwargs)
        self.tracker_policy = tracker_policy
        self.gate_fn = ViabilityGate(critic_path)
        self.tau_l, self.tau_h = tau_l, tau_h
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

    def _viability_step(self, info_prev: dict) -> tuple[float, float, float, dict]:
        feats = self.physics_features()
        phase = self._frame / max(self.n_frames - 1, 1)
        eta_vec = eta(info_prev, feats, phase, self.rt.base_height, (0.0, 0.0))
        v_raw, v_bar = self.gate_fn.update(eta_vec)
        g = gate(v_bar, self.tau_l, self.tau_h) if self.force_gate is None else self.force_gate
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
