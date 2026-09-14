#!/usr/bin/env python3
"""Multi-motion Stage A training on the full retargeted corpus, using the
train/val/test split in data/splits/ instead of the fixed 12-motion subset
in scripts/train_tracking_multi.py. Evaluates separately on the held-out
val+test motions the policy never trains on.

Usage
  python3 scripts/train_tracking_multi_full.py --steps 20000000 --n-envs 48 \
      --out logs/stageA_multi_full
  python3 scripts/train_tracking_multi_full.py --eval-only --out logs/stageA_multi_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS_DIR = "data/splits"
MOTIONS_DIR = "data/motions_retargeted"


def _load_split(name: str) -> list[str]:
    path = os.path.join(SPLITS_DIR, f"{name}_ids.txt")
    with open(path) as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    paths = [os.path.join(MOTIONS_DIR, f"{i}.npz") for i in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: missing npz files: {missing}")
    return paths


TRAIN_SET = _load_split("train")
VAL_SET = _load_split("val")
TEST_SET = _load_split("test")
HELDOUT_SET = VAL_SET + TEST_SET


def make_env(rank: int):
    def _f():
        from src.envs.multi_motion_env import MultiMotionTrackEnv

        return MultiMotionTrackEnv(TRAIN_SET, seed=5000 + rank)
    return _f


def evaluate_split(model, motions: list[str], out_dir: str, tag: str,
                    n_episodes_per_motion: int = 2,
                    record_video: bool = False) -> dict:
    from src.envs.multi_motion_env import MultiMotionTrackEnv
    from src.sim.rollout import write_video

    per_motion = {}
    video_frames = []
    first_motion_done = False

    for npz in motions:
        mid = os.path.basename(npz)[:-4]
        rows = {"policy": [], "pd_baseline": []}
        for arm in ("policy", "pd_baseline"):
            for ep in range(n_episodes_per_motion):
                want_video = record_video and not first_motion_done and ep == 0
                env = MultiMotionTrackEnv([npz], seed=9000 + ep,
                                           render_mode="rgb_array" if want_video else None)
                env.max_start = 0
                obs, info = env.reset(seed=9000 + ep)
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
                    if want_video and env.render_mode:
                        frame = env.render()
                        if frame is not None and len(video_frames) < 2000:
                            video_frames.append((arm, frame))
                rows[arm].append({
                    "tracking_rmse_rad": float(np.sqrt(np.mean(mses))),
                    "mean_reward": float(np.mean(rewards)),
                    "episode_len": len(rewards),
                    "fell": int(fell),
                })
                env.close()
            if arm == "pd_baseline":
                first_motion_done = True
        per_motion[mid] = {
            arm: {
                "tracking_rmse_mean": float(np.mean([r["tracking_rmse_rad"] for r in rows[arm]])),
                "fall_rate": float(np.mean([r["fell"] for r in rows[arm]])),
                "mean_episode_len": float(np.mean([r["episode_len"] for r in rows[arm]])),
            } for arm in ("policy", "pd_baseline")
        }

    if record_video and video_frames:
        for arm in ("policy", "pd_baseline"):
            fr = [f for a, f in video_frames if a == arm]
            if fr:
                write_video(os.path.join(out_dir, f"eval_{tag}_{arm}.mp4"), fr, fps=30)

    overall = {arm: {
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
        "tracking_rmse_mean": float(np.mean([per_motion[m][arm]["tracking_rmse_mean"] for m in per_motion])),
        "mean_episode_len": float(np.mean([per_motion[m][arm]["mean_episode_len"] for m in per_motion])),
    } for arm in ("policy", "pd_baseline")}
    return {"overall": overall, "per_motion": per_motion, "n_motions": len(motions)}


def evaluate(model, out_dir: str, record_video: bool = True) -> dict:
    """Evaluate on the train split and the held-out val+test split separately."""
    train_summary = evaluate_split(model, TRAIN_SET, out_dir, "train",
                                    record_video=False)
    heldout_summary = evaluate_split(model, HELDOUT_SET, out_dir, "heldout",
                                      record_video=record_video)
    return {"train": train_summary, "heldout": heldout_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--out", default="logs/stageA_multi_full")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "tracking_multi_full_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt, device=args.device)
        summary = evaluate(model, args.out)
        print(json.dumps({k: v["overall"] for k, v in summary.items()}, indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {
        "experiment_name": "stageA_multi_full", "stage": "A (multi-motion tracking, full corpus)",
        "algo": "PPO (stable-baselines3)",
        "train_motions": TRAIN_SET, "heldout_motions": HELDOUT_SET,
        "n_train_motions": len(TRAIN_SET), "n_heldout_motions": len(HELDOUT_SET),
        "steps": args.steps, "n_envs": args.n_envs, "device": args.device,
        "action": "joint target residuals, s_a=0.25 rad",
        "obs": "proprioception + reference + phase (124d)",
        "note": "full 56-motion train split; held-out eval on val+test (14 motions)",
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(i) for i in range(args.n_envs)]))

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=args.seed, device=args.device,
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs, "
          f"{len(TRAIN_SET)} train motions ({len(HELDOUT_SET)} held out) -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, args.out, record_video=not args.smoke)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps({k: v["overall"] for k, v in summary.items()}, indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
