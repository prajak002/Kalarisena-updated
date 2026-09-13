#!/usr/bin/env python3
"""Stage B: CoM/capture-point recoverability, via src/envs/com_refine_env.py.

Trains on the stable_stance family (configs/motion_families.yaml), reward
and obs blocks taken directly from configs/com.yaml (RewardBuilder /
ObservationBuilder terms already existed; this script is what was missing -
an actual env + PPO loop using them). Trained from scratch rather than
warm-started from Stage A: Stage A's checkpoint has a different observation
dimension (no com_support block), so a direct weight load isn't possible.

Usage
  python3 scripts/train_com.py --steps 2000000 --n-envs 8 --out logs/stageB_com
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STABLE_STANCE_MOTIONS = [
    "data/motions_retargeted/kw_long_stance.npz",
    "data/motions_retargeted/kw_long_stance_l.npz",
    "data/motions_retargeted/kt_vadivu_lowseat.npz",
    "data/motions_retargeted/kt_warrior_pose.npz",
    "data/motions_retargeted/ky_warrior_lunge.npz",
    "data/motions_retargeted/pk_warrior_oneleg.npz",
]


def make_env(reward_cfg: dict, rank: int):
    def _f():
        from src.envs.com_refine_env import ComRefineEnv

        return ComRefineEnv(STABLE_STANCE_MOTIONS, reward_cfg, seed=4000 + rank)
    return _f


def evaluate(model, reward_cfg: dict, out_dir: str, n_episodes_per_motion: int = 3) -> dict:
    from src.envs.com_refine_env import ComRefineEnv

    per_motion = {}
    for npz in STABLE_STANCE_MOTIONS:
        mid = os.path.basename(npz)[:-4]
        rows = []
        for ep in range(n_episodes_per_motion):
            env = ComRefineEnv([npz], reward_cfg, seed=8000 + ep)
            env.max_start = 0
            obs, _ = env.reset(seed=8000 + ep)
            done = trunc = False
            cp_margins, rewards, fell = [], [], False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, si = env.step(action)
                if si["has_support"]:
                    cp_margins.append(si["cp_margin"])
                rewards.append(r)
                fell = fell or si["fell"]
            rows.append({"mean_cp_margin": float(np.mean(cp_margins)) if cp_margins else 0.0,
                         "mean_reward": float(np.mean(rewards)), "fell": int(fell)})
            env.close()
        per_motion[mid] = {
            "mean_cp_margin": float(np.mean([r["mean_cp_margin"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
        }
    overall = {
        "mean_cp_margin": float(np.mean([v["mean_cp_margin"] for v in per_motion.values()])),
        "fall_rate": float(np.mean([v["fall_rate"] for v in per_motion.values()])),
    }
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/com.yaml")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageB_com")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config)) or {}
    reward_cfg = cfg["rewards"]

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "com_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, reward_cfg, args.out)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "stageB_com", "stage": "B (CoM/capture-point recoverability)",
            "algo": "PPO (stable-baselines3)", "config": args.config,
            "motions": STABLE_STANCE_MOTIONS, "steps": args.steps, "n_envs": args.n_envs,
            "note": "trained from scratch; Stage A checkpoint has a different obs dim"}
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
        seed=45, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, reward_cfg, args.out)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
