#!/usr/bin/env python3
"""Residual policy trained against the multi-motion tracker + multi-motion
SCVC critic (logs/stageA_multi12, logs/scvc_multi12), via
src/viability/residual_policy_multi.py.

Usage
  python3 scripts/calibrate_residual_gate.py --tracker ... --critic ... \
      --npz <every motion in MOTION_SET> --out logs/residual_multi12/gate_calibration.json
  python3 scripts/train_residual_policy_multi.py \
      --tracker logs/stageA_multi12/tracking_multi_best.zip \
      --critic logs/scvc_multi12/scvc_critic.pt \
      --gate-calibration logs/residual_multi12/gate_calibration.json \
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


def make_env(tracker_path: str, critic_path: str, rank: int, tau_map: dict[str, tuple[float, float]],
             seed_base: int, force_gate: float | None = None):
    def _f():
        from stable_baselines3 import PPO

        from src.viability.residual_policy_multi import IntentPreservingResidualMultiEnv

        tracker = PPO.load(tracker_path, device="cpu")
        ids = [os.path.basename(p)[:-4] for p in MOTION_SET]
        tau_l = [tau_map[i][0] for i in ids]
        tau_h = [tau_map[i][1] for i in ids]
        return IntentPreservingResidualMultiEnv(MOTION_SET, tracker, critic_path,
                                                 seed=seed_base + rank, tau_l=tau_l, tau_h=tau_h,
                                                 force_gate=force_gate)
    return _f


def evaluate(model, tracker_path: str, critic_path: str, tau_map: dict[str, tuple[float, float]],
             n_episodes_per_motion: int = 10, force_gate: float | None = None) -> dict:
    from stable_baselines3 import PPO

    from src.viability.residual_policy_multi import IntentPreservingResidualMultiEnv

    tracker = PPO.load(tracker_path, device="cpu")
    per_motion = {}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        tau_l, tau_h = tau_map[mid]
        rows = {"residual": [], "tracker_only": []}
        for arm in ("residual", "tracker_only"):
            for ep in range(n_episodes_per_motion):
                env = IntentPreservingResidualMultiEnv([npz], tracker, critic_path, seed=7500 + ep,
                                                         tau_l=[tau_l], tau_h=[tau_h], force_gate=force_gate)
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
            "mean_gate": float(np.mean([r["mean_gate"] for r in rows[arm]])),
        } for arm in ("residual", "tracker_only")}

    overall = {arm: {
        "fall_rate": float(np.mean([per_motion[m][arm]["fall_rate"] for m in per_motion])),
        "mean_episode_len": float(np.mean([per_motion[m][arm]["mean_episode_len"] for m in per_motion])),
        "mean_gate": float(np.mean([per_motion[m][arm]["mean_gate"] for m in per_motion])),
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
    ap.add_argument("--gate-calibration", required=True,
                     help="scripts/calibrate_residual_gate.py --out JSON, run with --npz covering "
                          "every motion in MOTION_SET; its per_motion_mean_v_raw[*].tau_l/tau_h "
                          "are used per-motion (a single global threshold pair does not work - "
                          "the critic's raw output scale varies by ~10 orders of magnitude "
                          "across motions)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--force-gate", type=float, default=None,
                     help="ablation: bypass the critic and fix the gate to this value "
                          "(e.g. 1.0 = residual always fully applied, no gating)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.gate_calibration) as fh:
        calib = json.load(fh)
    tau_map = {mid: (v["tau_l"], v["tau_h"]) for mid, v in calib["per_motion_mean_v_raw"].items()}
    missing = [os.path.basename(p)[:-4] for p in MOTION_SET if os.path.basename(p)[:-4] not in tau_map]
    if missing:
        raise ValueError(f"--gate-calibration is missing motions: {missing}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    ckpt = os.path.join(args.out, "residual_multi_best.zip")

    if args.eval_only:
        model = PPO.load(ckpt, device="cpu")
        summary = evaluate(model, args.tracker, args.critic, tau_map,
                            n_episodes_per_motion=args.eval_episodes, force_gate=args.force_gate)
        print(json.dumps(summary["overall"], indent=2))
        return 0

    if args.smoke:
        args.steps, args.n_envs = 4096, 2

    meta = {"experiment_name": "residual_multi12",
            "stage": "intent-preserving residual policy, multi-motion",
            "algo": "PPO (stable-baselines3)", "tracker": args.tracker, "critic": args.critic,
            "motions": MOTION_SET, "steps": args.steps, "n_envs": args.n_envs,
            "gate_calibration": args.gate_calibration, "tau_map": tau_map, "seed": args.seed,
            "force_gate": args.force_gate,
            "note": ("gate thresholds calibrated per tracker/critic pair via "
                     "scripts/calibrate_residual_gate.py rather than fixed constants; "
                     "r_balance uses a squared-shortfall penalty (gradient below the "
                     "safety margin) instead of a reward clipped to zero there; "
                     "RESIDUAL_ACTION_SCALE reduced from 0.4 to 0.15")}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecMonitor(vec_cls([make_env(args.tracker, args.critic, i, tau_map, 6000,
                                         force_gate=args.force_gate)
                                for i in range(args.n_envs)]))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=256, batch_size=1024, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.out, "tb"),
        seed=args.seed, device="cpu",
    )
    print(f"training {args.steps:,} steps on {args.n_envs} envs, {len(MOTION_SET)} motions -> {args.out}")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ckpt)
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"saved {ckpt}")

    summary = evaluate(model, args.tracker, args.critic, tau_map,
                        n_episodes_per_motion=args.eval_episodes, force_gate=args.force_gate)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    print(json.dumps(summary["overall"], indent=2))
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
