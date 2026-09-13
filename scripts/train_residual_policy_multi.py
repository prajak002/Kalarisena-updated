#!/usr/bin/env python3
"""Residual policy trained against the multi-motion tracker + multi-motion
SCVC critic (logs/stageA_multi12, logs/scvc_multi12), via
src/viability/residual_policy_multi.py.

Usage
  python3 scripts/train_residual_policy_multi.py \
      --tracker logs/stageA_multi12/tracking_multi_best.zip \
      --critic logs/scvc_multi12/scvc_critic.pt \
      --steps 1000000 --n-envs 8 --out logs/residual_multi12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

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


def make_env(tracker_path: str, critic_path: str, rank: int):
    def _f():
        from stable_baselines3 import PPO

        from src.viability.residual_policy_multi import IntentPreservingResidualMultiEnv

        tracker = PPO.load(tracker_path, device="cpu")
        return IntentPreservingResidualMultiEnv(MOTION_SET, tracker, critic_path, seed=6000 + rank)
    return _f


def evaluate(model, tracker_path: str, critic_path: str, n_episodes_per_motion: int = 2) -> dict:
    from stable_baselines3 import PPO

    from src.viability.residual_policy_multi import IntentPreservingResidualMultiEnv

    tracker = PPO.load(tracker_path, device="cpu")
    per_motion = {}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        rows = {"residual": [], "tracker_only": []}
        for arm in ("residual", "tracker_only"):
            for ep in range(n_episodes_per_motion):
                env = IntentPreservingResidualMultiEnv([npz], tracker, critic_path, seed=7500 + ep)
                env.max_start = 0
                obs, _ = env.reset(seed=7500 + ep)
                done = trunc = False
                mses, rewards, gates, fell = [], [], [], False
                while not (done or trunc):
                    if arm == "residual":
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        action = np.zeros(env.action_space.shape, dtype=np.float32)
                    obs, r, done, trunc, si = env.step(action)
                    mses.append(si["joint_mse"])
                    rewards.append(r)
                    gates.append(si["gate"])
                    fell = fell or si["fell"]
                rows[arm].append({"episode_len": len(rewards), "fell": int(fell),
                                   "mean_gate": float(np.mean(gates))})
                env.close()
        per_motion[mid] = {arm: {
            "fall_rate": float(np.mean([r["fell"] for r in rows[arm]])),
            "mean_episode_len": float(np.mean([r["episode_len"] for r in rows[arm]])),
        } for arm in ("residual", "tracker_only")}

    overall = {arm: {
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
        "mean_episode_len": float(np.mean([per_motion[m][arm]["mean_episode_len"] for m in per_motion])),
    } for arm in ("residual", "tracker_only")}
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/residual_multi12")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "residual_multi_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, args.tracker, args.critic)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "residual_multi12",
            "stage": "intent-preserving residual policy, multi-motion",
            "algo": "PPO (stable-baselines3)", "tracker": args.tracker, "critic": args.critic,
            "motions": MOTION_SET, "steps": args.steps, "n_envs": args.n_envs}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(args.tracker, args.critic, i) for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=47, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs, {len(MOTION_SET)} motions -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, args.tracker, args.critic)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
