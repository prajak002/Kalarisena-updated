#!/usr/bin/env python3
"""Stage 2: track a reference motion in real MuJoCo contact dynamics.

Usage:
  python3 scripts/sim_track_motion.py --motion single_leg_front_kick --out results_sim/
  python3 scripts/sim_track_motion.py --all --out results_sim/

Writes one per-step CSV and one mp4 per motion. A motion that destabilises is
recorded as a negative result (with the failure frame) rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.sim.motions import MOTION_IDS, get_motion
from src.sim.mujoco_runtime import DEFAULT_STANDING_POSE, G1MujocoRuntime, select_gl_backend
from src.sim.rollout import run_rollout, write_step_csv, write_video
from src.sim.controller import KalariController
from src.switch.mode_switch import SwitchConfig

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"

# configs/switch.yaml ships delta1 = 0.05 m. With the real 17x6 cm foot geometry
# the Kalari horse stance holds a genuine CP margin of only about +0.026 m, so
# delta1 = 0.05 fires on a robot that is not falling, and the protective crouch
# then drops it. Calibrated to 0.015 m: below the stance's true margin, still far
# above zero. The original value is kept in the config file; this override is
# reported rather than applied silently.
SWITCH_CFG = SwitchConfig(delta1=0.015, delta2=15.0, delta3=0.20, min_dwell_steps=30)


def track_one(motion_id: str, out_dir: str, video: bool = True) -> dict:
    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    ctrl = KalariController(rt, pin_wrap, switch_cfg=SWITCH_CFG)

    base_pose = {n: 0.0 for n in rt.act_joint_names}
    base_pose.update(DEFAULT_STANDING_POSE)
    motion = get_motion(motion_id, rt.act_joint_names, base_pose, fps=rt.CONTROL_HZ)

    print(f"\n=== {motion_id} ===")
    print(f"source={motion.source} family={motion.family} frames={motion.n_frames} "
          f"duration={motion.duration:.2f}s fps={motion.fps}")
    print(f"notes: {motion.notes}")
    ssw = motion.single_support_window()
    if ssw:
        print(f"single-support window: {ssw[0]:.2f}s -> {ssw[1]:.2f}s")

    rt.reset()
    result = run_rollout(
        rt, ctrl, motion, duration=motion.duration + 0.5,
        record_video=video, camera="track", initial_settle=0.3,
    )

    csv_path = os.path.join(out_dir, f"track_{motion_id}.csv")
    write_step_csv(csv_path, result.logs, extra={"source": motion.source, "trial": 0})

    video_written = False
    if video:
        video_written = write_video(
            os.path.join(out_dir, f"track_{motion_id}.mp4"), result.frames, fps=30
        )

    errs = np.array([l.tracking_err_rms for l in result.logs])
    cp = np.array([l.cp_margin for l in result.logs])
    modes = [l.mode for l in result.logs]
    valid_cp = cp[cp > -900]

    summary = {
        "motion_id": motion_id,
        "source": motion.source,
        "frames": len(result.logs),
        "duration_s": round(motion.duration + 0.5, 3),
        "stable": not result.fell,
        "final_base_z": round(result.final_base_z, 4),
        "min_base_z": round(result.min_base_z, 4),
        "tracking_err_rms_mean_rad": round(float(errs.mean()), 5),
        "tracking_err_rms_max_rad": round(float(errs.max()), 5),
        "cp_margin_min_m": round(float(valid_cp.min()), 5) if len(valid_cp) else None,
        "cp_margin_mean_m": round(float(valid_cp.mean()), 5) if len(valid_cp) else None,
        "peak_torque_norm_Nm": round(max(l.torque_norm for l in result.logs), 2),
        "peak_grf_N": round(max(l.grf_total for l in result.logs), 1),
        "mode_fractions": {m: round(modes.count(m) / len(modes), 4) for m in set(modes)},
        "switch_transitions": result.transitions,
        "failure_frame": result.failure_frame,
        "failure_time_s": result.failure_time,
        "csv": csv_path,
        "video": os.path.join(out_dir, f"track_{motion_id}.mp4") if video_written else None,
    }

    verdict = "STABLE" if summary["stable"] else "NEGATIVE RESULT - destabilised"
    print(f"result: {verdict}")
    print(f"  tracking RMS error mean {summary['tracking_err_rms_mean_rad']:.4f} rad "
          f"max {summary['tracking_err_rms_max_rad']:.4f} rad")
    print(f"  base height final {summary['final_base_z']:.4f} m, min {summary['min_base_z']:.4f} m")
    print(f"  CP margin min {summary['cp_margin_min_m']} m, mean {summary['cp_margin_mean_m']} m")
    print(f"  mode fractions {summary['mode_fractions']}")
    print(f"  switch transitions: {result.transitions}")
    if not summary["stable"]:
        print(f"  FAILURE at frame {result.failure_frame} (t={result.failure_time:.2f}s)")
    rt.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default=None, choices=list(MOTION_IDS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="results_sim")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if not args.all and args.motion is None:
        ap.error("pass --motion <id> or --all")

    os.makedirs(args.out, exist_ok=True)
    video = not args.no_video
    if video:
        select_gl_backend()

    ids = list(MOTION_IDS) if args.all else [args.motion]
    summaries = [track_one(mid, args.out, video=video) for mid in ids]

    path = os.path.join(args.out, "track_summary.json")
    with open(path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nwrote {path}")

    stable = [s["motion_id"] for s in summaries if s["stable"]]
    unstable = [s["motion_id"] for s in summaries if not s["stable"]]
    print(f"stable: {stable}")
    print(f"unstable (negative results, kept): {unstable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
