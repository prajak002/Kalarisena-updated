#!/usr/bin/env python3
"""Stage E: recovery-to-standing, via src/envs/recovery_env.py.

The one stage in this ladder that couldn't reuse the tracking-episode
structure every other stage does: there's no reference motion to recover
into, so the environment starts from a randomized, physically-settled
fallen pose and the policy has to reach and hold an upright stance with no
tracking target at all.

Usage
  python3 scripts/train_recovery.py --steps 2000000 --n-envs 8 --out logs/stageE_recovery
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env(reward_cfg: dict, rank: int):
    def _f():
        from src.envs.recovery_env import RecoveryEnv

        return RecoveryEnv(reward_cfg, seed=13000 + rank)
    return _f


def evaluate(model, reward_cfg: dict, n_episodes: int = 20) -> dict:
    from src.envs.recovery_env import RecoveryEnv

    env = RecoveryEnv(reward_cfg, seed=14000)
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=14000 + ep)
        done = trunc = False
        max_upright, steps = -1.0, 0
        success = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, si = env.step(action)
            max_upright = max(max_upright, si["upright"])
            steps += 1
            success = success or si["success"]
        rows.append({"success": int(success), "steps": steps, "max_upright": float(max_upright)})
    env.close()
    return {
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_max_upright": float(np.mean([r["max_upright"] for r in rows])),
        "n_episodes": n_episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recovery.yaml")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageE_recovery")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config)) or {}
    reward_cfg = cfg["rewards"]

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "recovery_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, reward_cfg)
        print(json.dumps(summary, indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "stageE_recovery", "stage": "E (recovery-to-standing)",
            "algo": "PPO (stable-baselines3)", "config": args.config,
            "steps": args.steps, "n_envs": args.n_envs,
            "note": "starts from a randomized settled fallen pose, no reference motion"}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(reward_cfg, i) for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=49, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, reward_cfg)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
