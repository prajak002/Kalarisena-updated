#!/usr/bin/env python3
"""Stage D: fall-impact minimization, via src/envs/fall_env.py.

configs/fall.yaml's vulnerable_weights names "head" and "torso"; the G1
29-DoF MJCF has no head body, so the head term is always 0.0 (see
fall_env.py's own docstring) - the torso term is the real signal.

Usage
  python3 scripts/train_fall.py --steps 2000000 --n-envs 8 --out logs/stageD_fall
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOTION_SET = [
    "data/motions_retargeted/kw_long_stance.npz",
    "data/motions_retargeted/kt_vadivu_lowseat.npz",
    "data/motions_retargeted/kt_warrior_pose.npz",
    "data/motions_retargeted/ky_warrior_lunge.npz",
    "data/motions_retargeted/kt_chuvadu_step.npz",
    "data/motions_retargeted/kw_deep_reach.npz",
    "data/motions_retargeted/ky_deep_lunge.npz",
    "data/motions_retargeted/pk_stance_transitions.npz",
    "data/motions_retargeted/ks_side_kick.npz",
    "data/motions_retargeted/kw_highkick_right.npz",
    "data/motions_retargeted/ky_kick_seq.npz",
    "data/motions_retargeted/pk_kick_lunge.npz",
]


def make_env(reward_cfg: dict, rank: int):
    def _f():
        from src.envs.fall_env import FallEnv

        return FallEnv(MOTION_SET, reward_cfg, seed=10000 + rank)
    return _f


def evaluate(model, reward_cfg: dict, n_episodes_per_motion: int = 3) -> dict:
    from src.envs.fall_env import FallEnv

    per_motion = {}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        rows = []
        for ep in range(n_episodes_per_motion):
            env = FallEnv([npz], reward_cfg, seed=11000 + ep)
            env.max_start = 0
            obs, _ = env.reset(seed=11000 + ep)
            done = trunc = False
            peak_force, rewards, fell = 0.0, [], False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, si = env.step(action)
                peak_force = max(peak_force, si["torso_impact_force"])
                rewards.append(r)
                fell = fell or si["fell"]
            rows.append({"peak_torso_force": float(peak_force),
                         "mean_reward": float(np.mean(rewards)), "fell": int(fell)})
            env.close()
        per_motion[mid] = {
            "peak_torso_force": float(np.mean([r["peak_torso_force"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
        }
    overall = {
        "peak_torso_force": float(np.mean([v["peak_torso_force"] for v in per_motion.values()])),
        "fall_rate": float(np.mean([v["fall_rate"] for v in per_motion.values()])),
    }
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fall.yaml")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageD_fall")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config)) or {}
    reward_cfg = cfg["rewards"]

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    ckpt = os.path.join(args.out, "fall_best.zip")
    norm_path = os.path.join(args.out, "vecnormalize.pkl")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, reward_cfg)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "stageD_fall", "stage": "D (fall-impact minimization)",
            "algo": "PPO (stable-baselines3)", "config": args.config,
            "motions": MOTION_SET, "steps": args.steps, "n_envs": args.n_envs,
            "note": "head term always 0.0, no head geom on this G1 MJCF; torso is real"}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(reward_cfg, i) for i in range(args.n_envs)]))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=48, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    venv.save(norm_path)
    print(f"saved {ckpt}")

    summary = evaluate(model, reward_cfg)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
