#!/usr/bin/env python3
"""Compute the empirical distribution of the viability critic's raw output
V_raw over nominal (no-residual) rollouts of the frozen tracker, so the
residual gate's tau_l/tau_h can be set from real quantiles of this
critic's own output instead of fixed literal constants (0.25/0.70) that
may not match its actual calibration.

IMPORTANT: pass every motion the resulting gate will actually be used
against (e.g. the full MOTION_SET for a multi-motion residual policy).
Calibrating against a single motion whose critic-output scale is not
representative of the others will mis-set the gate for the rest - this
happened in practice: calibrating against kt_vadivu_lowseat alone (an
outlier with a ~30x lower V_raw scale than kw_long_stance) pinned the
gate fully closed (mean_gate=0.0) once applied across all 12 motions.

Usage
  python3 scripts/calibrate_residual_gate.py \
      --tracker logs/stageA_kw_long_stance/tracking_best.zip \
      --critic logs/scvc_kw_long_stance/scvc_critic.pt \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --episodes 20

  # pool across every motion a multi-motion residual policy will see
  python3 scripts/calibrate_residual_gate.py \
      --tracker logs/stageA_multi12/tracking_multi_best.zip \
      --critic logs/scvc_multi12/scvc_critic.pt \
      --npz data/motions_retargeted/kw_long_stance.npz data/motions_retargeted/kt_vadivu_lowseat.npz ... \
      --episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viability.perturbed_env import PerturbedTrackEnv
from src.viability.residual_policy import ViabilityGate
from src.viability.features import eta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--npz", required=True, nargs="+")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    tracker = PPO.load(args.tracker, device="cpu")

    v_raws = []
    per_motion_mean = {}
    for npz_path in args.npz:
        gate_fn = ViabilityGate(args.critic)
        env = PerturbedTrackEnv(npz_path, seed=0)
        env.max_start = 0
        motion_v_raws = []
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep)
            gate_fn.reset()
            done = trunc = False
            while not (done or trunc):
                info_prev = {"joint_mse": float(np.mean(
                    (env.rt.data.qpos[env.rt.act_qadr] - env._ref_joints(env._frame)) ** 2)),
                    "upright": env.rt.torso_upright_cos()}
                feats = env.physics_features()
                phase = env._frame / max(env.n_frames - 1, 1)
                eta_vec = eta(info_prev, feats, phase, env.rt.base_height, (0.0, 0.0))
                v_raw, v_bar = gate_fn.update(eta_vec)
                motion_v_raws.append(v_raw)
                action, _ = tracker.predict(obs, deterministic=True)
                obs, r, done, trunc, si = env.step(action)
        env.close()
        mv = np.array(motion_v_raws)
        per_motion_mean[os.path.basename(npz_path)[:-4]] = {
            "mean": float(mv.mean()),
            "tau_l": float(np.percentile(mv, 20)),
            "tau_h": float(np.percentile(mv, 70)),
        }
        v_raws.extend(motion_v_raws)
        print(f"  {npz_path}: mean V_raw = {mv.mean():.6g} ({len(motion_v_raws)} steps)")

    v_raws = np.array(v_raws)
    percentiles = {p: float(np.percentile(v_raws, p)) for p in [5, 10, 20, 30, 50, 70, 80, 90, 95]}
    result = {
        "tracker": args.tracker, "critic": args.critic, "npz": args.npz,
        "per_motion_mean_v_raw": per_motion_mean,
        "n_episodes": args.episodes, "n_steps": len(v_raws),
        "mean": float(v_raws.mean()), "std": float(v_raws.std()),
        "min": float(v_raws.min()), "max": float(v_raws.max()),
        "percentiles": percentiles,
        "suggested_tau_l": percentiles[20], "suggested_tau_h": percentiles[70],
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
