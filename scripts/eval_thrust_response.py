#!/usr/bin/env python3
"""Real post-training thrust-response rollout: the TRAINED Stage A policy
under a lateral push, via PerturbedTrackEnv (src/viability/perturbed_env.py) -
same xfrc_applied mechanism as scripts/sim_push_sweep.py, applied to the
learned policy instead of the scripted controller.

Usage
  python3 scripts/eval_thrust_response.py \
      --policy logs/stageA_kw_long_stance/tracking_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --forces 20 60 100 --out logs/stageA_kw_long_stance
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viability.perturbation import Perturbation
from src.viability.perturbed_env import PerturbedTrackEnv


def run_one(env: PerturbedTrackEnv, policy, force: float, record: bool):
    pert = Perturbation(t_start=0.6, duration=0.1, force_x=0.0, force_y=force, obs_noise_std=0.0)
    env.set_perturbation(pert)
    obs, _ = env.reset()
    frames = []
    fell = False
    steps = 0
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if record:
            frames.append(env.render())
        fell = info["fell"]
        steps += 1
        done = terminated or truncated
    return {"force": force, "fell": bool(fell), "steps": steps}, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--forces", type=float, nargs="+", default=[20.0, 60.0, 100.0])
    ap.add_argument("--out", default="logs/thrust_eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO
    policy = PPO.load(args.policy)

    env = PerturbedTrackEnv(args.npz, seed=0, render_mode="rgb_array")

    results = []
    for force in args.forces:
        record = True
        res, frames = run_one(env, policy, force, record)
        results.append(res)
        if frames:
            path = os.path.join(args.out, f"thrust_trained_{int(force)}N.mp4")
            imageio.mimwrite(path, frames, fps=30, macro_block_size=None)
            print(f"wrote {path} ({len(frames)} frames)")
        print(res)

    with open(os.path.join(args.out, "thrust_eval_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
