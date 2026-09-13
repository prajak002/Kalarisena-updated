#!/usr/bin/env python3
"""Multi-motion Stage A training via src/envs/multi_motion_env.py, on a
12-motion subset spanning all 3 motion families in
configs/motion_families.yaml.

Usage
  python3 scripts/train_tracking_multi.py --steps 3000000 --n-envs 8 \
      --out logs/stageA_multi12
  python3 scripts/train_tracking_multi.py --eval-only --out logs/stageA_multi12
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


def make_env(rank: int):
    def _f():
        from src.envs.multi_motion_env import MultiMotionTrackEnv

        return MultiMotionTrackEnv(MOTION_SET, seed=3000 + rank)
    return _f


def evaluate(model, out_dir: str, n_episodes_per_motion: int = 2,
             record_video: bool = True) -> dict:
    from src.envs.multi_motion_env import MultiMotionTrackEnv
    from src.sim.rollout import write_video

    per_motion = {}
    video_frames = []
    first_motion_done = False

    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        rows = {"policy": [], "pd_baseline": []}
        for arm in ("policy", "pd_baseline"):
            for ep in range(n_episodes_per_motion):
                want_video = record_video and not first_motion_done and ep == 0
                env = MultiMotionTrackEnv([npz], seed=7000 + ep,
                                           render_mode="rgb_array" if want_video else None)
                env.max_start = 0
                obs, info = env.reset(seed=7000 + ep)
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
                write_video(os.path.join(out_dir, f"eval_{arm}.mp4"), fr, fps=30)

    overall = {arm: {
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
        "tracking_rmse_mean": float(np.mean([per_motion[m][arm]["tracking_rmse_mean"] for m in per_motion])),
        "mean_episode_len": float(np.mean([per_motion[m][arm]["mean_episode_len"] for m in per_motion])),
    } for arm in ("policy", "pd_baseline")}
    return {"overall": overall, "per_motion": per_motion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="logs/stageA_multi12")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    ckpt = os.path.join(args.out, "tracking_multi_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt)
        summary = evaluate(model, args.out)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {
        "experiment_name": "stageA_multi12", "stage": "A (multi-motion tracking)",
        "algo": "PPO (stable-baselines3)", "motions": MOTION_SET,
        "steps": args.steps, "n_envs": args.n_envs,
        "action": "joint target residuals, s_a=0.25 rad",
        "obs": "proprioception + reference + phase (124d)",
        "note": ("widens single-motion Stage A (logs/stageA_kw_long_stance) to 12 "
                 "motions across all 3 taxonomy families; still far short of the "
                 "paper's full corpus"),
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
        seed=44, device="auto",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs, {len(MOTION_SET)} motions -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    print(f"saved {ckpt}")

    summary = evaluate(model, args.out, record_video=not args.smoke)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
