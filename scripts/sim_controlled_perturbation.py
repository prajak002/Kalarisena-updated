#!/usr/bin/env python3
"""Apply a push pattern (timing, direction, magnitude, one or many pushes
per episode) to a trained policy via PerturbedTrackEnv, optionally routed
through the real ModeSwitch (--fall enables NOMINAL/FALL/RECOVERY
switching), and render an annotated video: push arrow, capture-point
margin, angular momentum, active mode, fall status.

Usage
  # one push, tracker only
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi_full/tracking_multi_full_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --push 1.0 90 80 0.1 --out logs/controlled_demo/single_push

  # custom multi-push pattern, with real mode switching to the fall policy
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --push 0.8 90 60 0.1 --push 1.7 270 60 0.1 --push 2.6 90 90 0.1 \
      --out logs/controlled_demo/custom

  # a named preset instead of typing out --push repeatedly
  python3 scripts/sim_controlled_perturbation.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --recovery logs/stageE_recovery_v2/recovery_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --preset relentless --out logs/controlled_demo/relentless

Each --push is (t_start_seconds, angle_degrees, magnitude_newtons, duration_seconds).
angle 0=+x (forward, in the motion's frame), 90=+y (left), 180=-x, 270=-y.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.rollout import write_video
from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig
from src.viability.perturbation import PushPattern
from src.viability.perturbed_env import PerturbedTrackEnv

PRESETS = {
    # (t_start, angle_deg, magnitude_N, duration_s)
    "single": [(1.0, 90.0, 80.0, 0.1)],
    "double": [(1.0, 90.0, 70.0, 0.1), (2.5, 270.0, 70.0, 0.1)],
    "relentless": [(0.8 + 0.9 * i, 90.0 if i % 2 == 0 else 270.0, 60.0, 0.1) for i in range(6)],
    "growing": [(0.8 + 0.9 * i, 90.0, 30.0 + 20.0 * i, 0.1) for i in range(5)],
}

MODE_COLOR = {"nominal": (80, 220, 80), "fall": (60, 160, 255), "recovery": (60, 60, 255)}


def _draw_overlay(frame: np.ndarray, t: float, seg_mag: float, seg_angle: float,
                   feats: dict, mode: str | None, fell: bool, torso_force: float) -> np.ndarray:
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    y = [22]

    def line(text: str, color=(255, 255, 255)) -> None:
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y[0] += 20

    line(f"t = {t:5.2f}s")
    if mode is not None:
        line(f"MODE: {mode.upper()}", MODE_COLOR[mode])
    cp = feats["cp_margin"]
    line(f"cp_margin: {cp:+.3f}", (80, 220, 80) if cp > 0.05 else (60, 60, 255))
    line(f"momentum: {feats['momentum_norm']:.2f}")
    line(f"torso impact: {torso_force:.1f} N")
    if fell:
        cv2.putText(frame, "FELL", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

    if seg_mag > 0:
        cx, cy = w // 2, h - 40
        rad = np.radians(seg_angle)
        dx, dy = int(28 * np.cos(rad)), int(-28 * np.sin(rad))
        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 165, 255), 3, tipLength=0.4)
        cv2.putText(frame, f"{seg_mag:.0f} N", (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)
    return frame


def run_episode(env: PerturbedTrackEnv, nominal, fall, recovery,
                 switch: ModeSwitch | None, pattern: PushPattern,
                 out_path: str, fps: int = 30) -> dict:
    obs, info = env.reset()
    if switch is not None:
        switch.reset()
    env.set_perturbation(pattern)

    done = trunc = False
    step = 0
    frames = []
    fell_ever = False
    mode_counts: dict[str, int] = {}

    while not (done or trunc):
        feats = env.switch_features(step)
        mode = switch.step(feats) if switch is not None else Mode.NOMINAL
        mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        if mode == Mode.FALL and fall is not None:
            fall_obs = np.concatenate([obs, [env.torso_force(), 0.0]])
            action, _ = fall.predict(fall_obs, deterministic=True)
        elif mode == Mode.RECOVERY and recovery is not None:
            # RecoveryEnv's observation space differs from the tracker's;
            # fall back to the nominal tracker (see eval_integrated_switch.py).
            action, _ = nominal.predict(obs, deterministic=True)
        else:
            action, _ = nominal.predict(obs, deterministic=True)

        obs, r, done, trunc, si = env.step(action)
        fell_ever = fell_ever or si["fell"]

        seg = pattern._current
        seg_mag = seg.magnitude if seg else 0.0
        seg_angle = seg.angle_deg if seg else 0.0
        frame = _draw_overlay(env.render(), env.t, seg_mag, seg_angle, feats,
                               mode.value if switch is not None else None,
                               si["fell"], env.torso_force())
        frames.append(frame)
        step += 1

    write_video(out_path, frames, fps=fps)
    return {
        "fell": int(fell_ever), "steps": step, "mode_counts": mode_counts,
        "transitions": switch.transition_log if switch is not None else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--fall", default=None, help="Stage D policy; enables mode switching")
    ap.add_argument("--recovery", default=None, help="Stage E policy")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--push", type=float, nargs=4, action="append", metavar=("T", "ANGLE_DEG", "MAG_N", "DUR_S"),
                     help="One push segment; repeat for multiple pushes in one episode")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--out", default="logs/controlled_demo")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    pushes = args.push or (PRESETS[args.preset] if args.preset else PRESETS["single"])
    pattern = PushPattern.from_polar(pushes)

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO

    nominal = PPO.load(args.nominal, device="cpu")
    fall = PPO.load(args.fall, device="cpu") if args.fall else None
    recovery = PPO.load(args.recovery, device="cpu") if args.recovery else None
    switch = ModeSwitch(SwitchConfig(delta1=0.05, delta2=15.0, delta3=0.20, min_dwell_steps=30)) if fall else None

    env = PerturbedTrackEnv(args.npz, seed=0, render_mode="rgb_array")
    env.max_start = 0

    video_path = os.path.join(args.out, "annotated.mp4")
    result = run_episode(env, nominal, fall, recovery, switch, pattern, video_path, fps=args.fps)
    env.close()

    summary = {
        "nominal": args.nominal, "fall": args.fall, "recovery": args.recovery,
        "npz": args.npz, "pushes": pushes, "switching_enabled": switch is not None,
        "result": result,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
