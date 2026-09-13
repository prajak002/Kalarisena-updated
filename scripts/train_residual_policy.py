#!/usr/bin/env python3
"""Train the intent-preserving residual policy on top of a frozen tracker
and a frozen SCVC critic, via IntentPreservingResidualEnv.

Usage
  python3 scripts/train_residual_policy.py \
      --tracker logs/stageA_kw_long_stance/tracking_best.zip \
      --critic logs/scvc_kw_long_stance/scvc_critic.pt \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --steps 1000000 --n-envs 8 --out logs/residual_kw_long_stance
  python3 scripts/train_residual_policy.py --eval-only --out logs/residual_kw_long_stance \
      --tracker logs/stageA_kw_long_stance/tracking_best.zip \
      --critic logs/scvc_kw_long_stance/scvc_critic.pt \
      --npz data/motions_retargeted/kw_long_stance.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env(npz_path: str, tracker_path: str, critic_path: str, rank: int):
    def _f():
        from stable_baselines3 import PPO

        from src.viability.residual_policy import IntentPreservingResidualEnv

        tracker = PPO.load(tracker_path, device="cpu")
        return IntentPreservingResidualEnv(npz_path, tracker, critic_path, seed=2000 + rank)
    return _f


def evaluate(model, npz_path: str, tracker_path: str, critic_path: str, out_dir: str,
             n_episodes: int = 5, record_video: bool = True) -> dict:
    from stable_baselines3 import PPO

    from src.sim.rollout import write_video
    from src.viability.residual_policy import IntentPreservingResidualEnv

    tracker = PPO.load(tracker_path, device="cpu")
    results = {"residual": [], "tracker_only": []}
    video_frames = []

    for arm in ("residual", "tracker_only"):
        for ep in range(n_episodes):
            env = IntentPreservingResidualEnv(
                npz_path, tracker, critic_path, seed=6000 + ep,
                render_mode="rgb_array" if (record_video and ep == 0) else None)
            env.max_start = 0
            obs, info = env.reset(seed=6000 + ep)
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
                if record_video and ep == 0 and env.render_mode:
                    frame = env.render()
                    if frame is not None and len(video_frames) < 2000:
                        video_frames.append((arm, frame))
            results[arm].append({
                "episode": ep,
                "tracking_rmse_rad": float(np.sqrt(np.mean(mses))),
                "mean_reward": float(np.mean(rewards)),
                "mean_gate": float(np.mean(gates)),
                "episode_len": len(rewards),
                "fell": int(fell),
            })
            env.close()

    if record_video and video_frames:
        for arm in ("residual", "tracker_only"):
            fr = [f for a, f in video_frames if a == arm]
            if fr:
                write_video(os.path.join(out_dir, f"eval_{arm}.mp4"), fr, fps=30)

    summary = {}
    for arm, rows in results.items():
        summary[arm] = {
            "tracking_rmse_mean": float(np.mean([r["tracking_rmse_rad"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
            "mean_episode_len": float(np.mean([r["episode_len"] for r in rows])),
            "mean_gate": float(np.mean([r["mean_gate"] for r in rows])),
            "n_episodes": len(rows),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/motions_retargeted/kw_long_stance.npz")
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    mid = os.path.basename(args.npz)[:-4]
    out_dir = args.out or f"logs/residual_{mid}"
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(out_dir, "residual_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, args.npz, args.tracker, args.critic, out_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {
        "experiment_name": f"residual_{mid}",
        "stage": "intent-preserving residual policy (paper Sec 3.3-3.4)",
        "algo": "PPO (stable-baselines3)",
        "tracker": args.tracker, "critic": args.critic, "npz": args.npz,
        "steps": args.steps, "n_envs": args.n_envs,
        "action": f"delta_a_t, authority-scaled and gated by g(V_bar_t)",
        "obs": "base tracking obs (124d) + [V_raw, V_bar, gate, cp_margin, com_margin]",
        "note": ("first trained instance of pi_theta; frozen tracker and frozen "
                 "critic both narrow-scope (single motion) per their own training runs"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(args.npz, args.tracker, args.critic, i)
                                for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(out_dir, "tb"),
        seed=43, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {out_dir}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, args.npz, args.tracker, args.critic, out_dir,
                        record_video=not args.smoke)
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
