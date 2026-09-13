#!/usr/bin/env python3
"""Stage C: momentum regulation, via src/envs/momentum_env.py.

configs/momentum.yaml's curriculum names a "rotational" family that does not
exist in configs/motion_families.yaml (only stable_stance / translational /
explosive_strike are tagged) - the same taxonomy mismatch flagged earlier in
this repo's own notes. Using explosive_strike (the config's other named
family) with its configured phase_weight instead.

Usage
  python3 scripts/train_momentum.py --steps 2000000 --n-envs 8 --out logs/stageC_momentum
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPLOSIVE_MOTIONS = [
    "data/motions_retargeted/ks_side_kick.npz",
    "data/motions_retargeted/kw_highkick_right.npz",
    "data/motions_retargeted/kw_highkick_left.npz",
    "data/motions_retargeted/ky_kick_seq.npz",
    "data/motions_retargeted/pk_kick_lunge.npz",
    "data/motions_retargeted/nk_squat_highkick.npz",
]


def make_env(reward_cfg: dict, phase_weight: float, rank: int):
    def _f():
        from src.envs.momentum_env import MomentumEnv

        return MomentumEnv(EXPLOSIVE_MOTIONS, reward_cfg, phase_weight=phase_weight, seed=5000 + rank)
    return _f


def evaluate(model, reward_cfg: dict, phase_weight: float, n_episodes_per_motion: int = 3) -> dict:
    from src.envs.momentum_env import MomentumEnv

    per_motion = {}
    for npz in EXPLOSIVE_MOTIONS:
        mid = os.path.basename(npz)[:-4]
        rows = []
        for ep in range(n_episodes_per_motion):
            env = MomentumEnv([npz], reward_cfg, phase_weight=phase_weight, seed=9000 + ep)
            env.max_start = 0
            obs, _ = env.reset(seed=9000 + ep)
            done = trunc = False
            hg_norms, rewards, fell = [], [], False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, si = env.step(action)
                hg_norms.append(si["hg_angular_norm"])
                rewards.append(r)
                fell = fell or si["fell"]
            rows.append({"mean_hg_angular_norm": float(np.mean(hg_norms)),
                         "mean_reward": float(np.mean(rewards)), "fell": int(fell)})
            env.close()
        per_motion[mid] = {
            "mean_hg_angular_norm": float(np.mean([r["mean_hg_angular_norm"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
        }
    overall = {
        "mean_hg_angular_norm": float(np.mean([v["mean_hg_angular_norm"] for v in per_motion.values()])),
        "fall_rate": float(np.mean([v["fall_rate"] for v in per_motion.values()])),
    }
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/momentum.yaml")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageC_momentum")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config)) or {}
    reward_cfg = cfg["rewards"]
    phase_weight = float(cfg.get("curriculum", {}).get(
        "phase_weight_by_family", {}).get("explosive_strike", 1.0))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "momentum_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, reward_cfg, phase_weight)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "stageC_momentum", "stage": "C (momentum regulation)",
            "algo": "PPO (stable-baselines3)", "config": args.config,
            "motions": EXPLOSIVE_MOTIONS, "phase_weight": phase_weight,
            "steps": args.steps, "n_envs": args.n_envs,
            "note": "trained from scratch; see docstring re: rotational/explosive_strike mismatch"}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(reward_cfg, phase_weight, i) for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=46, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, reward_cfg, phase_weight)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
