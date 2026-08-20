#!/usr/bin/env python3
"""Stage A training: PPO tracking policy on one reference motion (paper 6.10).

Implements the paper's Stage A with the honest, minimal scope stated in
results_paper/PAPER_ASSETS.md: a single-motion residual tracking policy,
trained with PPO in MuJoCo. Not ExBody-scale (that needs massively parallel
GPU simulation); a genuine first rung that produces a real learned policy,
real training curves, and a real evaluation per the section 7 protocol.

Usage
  python3 scripts/train_tracking.py --npz data/motions_retargeted/kw_long_stance.npz \
      --steps 3000000 --n-envs 8 --out logs/stageA_kw_long_stance
  python3 scripts/train_tracking.py --eval-only --out logs/stageA_kw_long_stance

Artifacts (paper 7.2 training-run contract): config snapshot, meta.json,
checkpoints, train CSV via SB3 logger, eval CSV, rollout video.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env(npz_path: str, rank: int):
    def _f():
        from src.envs.kalari_track_env import KalariTrackEnv

        return KalariTrackEnv(npz_path, seed=1000 + rank)
    return _f


def evaluate(model, npz_path: str, out_dir: str, n_episodes: int = 5,
             record_video: bool = True) -> dict:
    """Deterministic evaluation + comparison against zero-action (pure PD) baseline."""
    from src.envs.kalari_track_env import KalariTrackEnv
    from src.sim.rollout import write_video

    results = {"policy": [], "pd_baseline": []}
    video_frames = []

    for arm in ("policy", "pd_baseline"):
        for ep in range(n_episodes):
            env = KalariTrackEnv(npz_path, seed=5000 + ep,
                                 render_mode="rgb_array" if (record_video and ep == 0) else None)
            # evaluation always starts at frame 0: full-motion execution
            env.max_start = 0
            obs, info = env.reset(seed=5000 + ep)
            done = trunc = False
            mses, rewards, fell = [], [], False
            while not (done or trunc):
                if arm == "policy":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, r, done, trunc, si = env.step(action)
                mses.append(si["joint_mse"])
                rewards.append(r)
                fell = fell or si["fell"]
                if record_video and ep == 0 and env.render_mode:
                    frame = env.render()
                    if frame is not None and len(video_frames) < 2000:
                        video_frames.append((arm, frame))
            results[arm].append({
                "episode": ep,
                "tracking_rmse_rad": float(np.sqrt(np.mean(mses))),
                "mean_reward": float(np.mean(rewards)),
                "episode_len": len(rewards),
                "fell": int(fell),
            })
            env.close()

    if record_video and video_frames:
        for arm in ("policy", "pd_baseline"):
            fr = [f for a, f in video_frames if a == arm]
            if fr:
                write_video(os.path.join(out_dir, f"eval_{arm}.mp4"), fr, fps=30)

    summary = {}
    for arm, rows in results.items():
        summary[arm] = {
            "tracking_rmse_mean": float(np.mean([r["tracking_rmse_rad"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
            "mean_episode_len": float(np.mean([r["episode_len"] for r in rows])),
            "n_episodes": len(rows),
        }
    with open(os.path.join(out_dir, "eval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm"] + list(results["policy"][0]))
        w.writeheader()
        for arm, rows in results.items():
            for r in rows:
                w.writerow({"arm": arm, **r})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/motions_retargeted/kw_long_stance.npz")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring")
    args = ap.parse_args()

    mid = os.path.basename(args.npz)[:-4]
    out_dir = args.out or f"logs/stageA_{mid}"
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(out_dir, "tracking_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, args.npz, out_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {
        "experiment_name": f"stageA_{mid}", "stage": "A (nominal tracking)",
        "algo": "PPO (stable-baselines3)", "npz": args.npz, "steps": args.steps,
        "n_envs": args.n_envs, "action": "joint target residuals, s_a=0.25 rad",
        "obs": "proprioception + reference + phase (124d)",
        "note": "minimal Stage A per PAPER_ASSETS.md; single motion",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(args.npz, i) for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(out_dir, "tb"),
        seed=42, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs -> {out_dir}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, args.npz, out_dir, record_video=not args.smoke)
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
