#!/usr/bin/env python3
"""Physics validation of retargeted NPZ motions (paper section 6 preprocessing checks).

For every NPZ in data/motions_retargeted/ this script:

  1. replays the motion kinematically in the validated MuJoCo G1 model
     (results_review/mjc_<id>.mp4) so the retarget can be compared against
     GEM-X's own render in an independent model;
  2. measures, per frame, with the same Pinocchio/MuJoCo stack the controller
     uses:
       - lowest-foot-sphere height  -> ground penetration / floating
       - joint-limit violations against the official MJCF ranges
       - peak joint velocity (finite-difference)
       - CoM horizontal offset from the foot-support centroid
  3. writes one diagnostics plot per motion (results_review/diag_<id>.png)
     and a machine-readable summary (results_review/diagnostics.json).

Every number is measured from the retarget itself; thresholds are stated in the
output. This is the honest answer to "which motions need correction": the page
shows the reviewer both what it looks like and what the physics says.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.render_g1_motion import load_npz_motion
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime, select_gl_backend
from src.sim.rollout import write_video

NPZ_DIR = "data/motions_retargeted"
OUT = "results_review"

PENETRATION_TOL = 0.02   # m below floor before flagged
FLOAT_TOL = 0.06         # m above floor before flagged (both feet)
VEL_LIMIT = 12.0         # rad/s plausibility bound (well above G1 joint specs)


def diagnose(npz_path: str, rt: G1MujocoRuntime, video_fps: int = 30,
             render: bool = True) -> dict:
    m = load_npz_motion(npz_path)
    mid = m["motion_id"]
    joint_pos = m["joint_pos"]
    n = joint_pos.shape[0]
    fps = float(m["fps"]) or 30.0

    # name-keyed joint mapping (never index order)
    idx = {nme: i for i, nme in enumerate(m["joint_names"] or [])}
    col_for_act = np.array([idx.get(j, -1) for j in rt.act_joint_names])

    # joint limits from the official model
    lo = np.zeros(rt.model.nu)
    hi = np.zeros(rt.model.nu)
    for a in range(rt.model.nu):
        jid = int(rt.model.actuator_trnid[a, 0])
        lo[a], hi[a] = rt.model.jnt_range[jid]

    foot_geoms = rt.left_foot_geoms + rt.right_foot_geoms
    default_q = rt.default_qpos()
    stride = max(1, int(round(fps / video_fps)))

    frames = []
    foot_z, com_off, viol_frames = [], [], []
    worst_viol = ("", 0.0)

    for k in range(n):
        qpos = default_q.copy()
        if m["root_pos"] is not None:
            qpos[0:3] = m["root_pos"][k]
        qpos[3:7] = quat_xyzw_to_wxyz(m["root_quat_xyzw"][k])
        vals = np.zeros(rt.model.nu)
        for a in range(rt.model.nu):
            if col_for_act[a] >= 0:
                vals[a] = joint_pos[k, col_for_act[a]]
                qpos[rt.act_qadr[a]] = vals[a]
        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = 0.0
        rt.mujoco.mj_forward(rt.model, rt.data)

        lowest = min(float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                     for g in foot_geoms)
        foot_z.append(lowest)

        over = np.maximum(vals - hi, 0) + np.maximum(lo - vals, 0)
        if over.max() > 1e-6:
            viol_frames.append(k)
            a = int(np.argmax(over))
            if over[a] > worst_viol[1]:
                worst_viol = (rt.act_joint_names[a], float(over[a]))

        feet_xy = np.mean([rt.data.geom_xpos[g][:2] for g in foot_geoms], axis=0)
        com = rt.mj_subtree_com()
        com_off.append(float(np.linalg.norm(com[:2] - feet_xy)))

        if render and k % stride == 0:
            frames.append(rt.render_frame(camera="track"))

    dq = np.diff(joint_pos, axis=0) * fps
    vel_peak = float(np.abs(dq).max()) if len(dq) else 0.0
    vel_joint = (m["joint_names"][int(np.unravel_index(np.argmax(np.abs(dq)), dq.shape)[1])]
                 if len(dq) and m["joint_names"] else "?")

    video = os.path.join(OUT, f"mjc_{mid}.mp4")
    if render:
        write_video(video, frames, fps=video_fps)
    elif not os.path.isfile(video):
        video = None

    foot_z = np.array(foot_z)
    com_off = np.array(com_off)
    t = np.arange(n) / fps

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    axes[0].plot(t, foot_z * 100, lw=1.4, color="#1d3557")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhspan(-100, -PENETRATION_TOL * 100, color="#c1121f", alpha=0.12)
    axes[0].axhspan(FLOAT_TOL * 100, 100, color="#e9c46a", alpha=0.18)
    axes[0].set_ylabel("lowest foot (cm)")
    axes[0].set_ylim(min(-4, foot_z.min() * 100 - 1), max(12, foot_z.max() * 100 + 1))
    axes[0].set_title(f"{mid}: retarget physics check", fontsize=10)
    axes[1].plot(t, com_off * 100, lw=1.4, color="#2a9d8f")
    axes[1].set_ylabel("CoM offset from\nfeet centroid (cm)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    plot = os.path.join(OUT, f"diag_{mid}.png")
    fig.savefig(plot, dpi=110)
    plt.close(fig)

    pen_frames = int((foot_z < -PENETRATION_TOL).sum())
    float_frames = int((foot_z > FLOAT_TOL).sum())
    flags = []
    if pen_frames > 0.05 * n:
        flags.append(f"ground penetration {pen_frames}/{n} frames "
                     f"(max {-foot_z.min() * 100:.1f} cm)")
    if float_frames > 0.05 * n:
        flags.append(f"feet floating {float_frames}/{n} frames "
                     f"(max {foot_z.max() * 100:.1f} cm)")
    if len(viol_frames) > 0.02 * n:
        flags.append(f"joint-limit violations {len(viol_frames)}/{n} frames "
                     f"(worst {worst_viol[0]} +{np.degrees(worst_viol[1]):.1f} deg)")
    if vel_peak > VEL_LIMIT:
        flags.append(f"implausible joint velocity {vel_peak:.1f} rad/s ({vel_joint})")

    rec = {
        "motion_id": mid,
        "mjc_video": video,
        "diag_plot": plot,
        "frames": n,
        "foot_penetration_max_cm": round(float(max(0, -foot_z.min())) * 100, 2),
        "foot_float_max_cm": round(float(max(0, foot_z.max())) * 100, 2),
        "frames_penetrating": pen_frames,
        "frames_floating": float_frames,
        "joint_limit_violation_frames": len(viol_frames),
        "worst_limit_joint": worst_viol[0],
        "worst_limit_overrun_deg": round(float(np.degrees(worst_viol[1])), 2),
        "peak_joint_velocity_rad_s": round(vel_peak, 2),
        "peak_velocity_joint": vel_joint,
        "com_offset_mean_cm": round(float(com_off.mean()) * 100, 2),
        # per-frame series for the interactive chart on the review page
        "series": {
            "fps": fps,
            "foot_cm": [round(float(v) * 100, 2) for v in foot_z],
            "com_cm": [round(float(v) * 100, 2) for v in com_off],
        },
        "physics_flags": flags,
    }
    print(f"  {mid:26s} pen {rec['foot_penetration_max_cm']:5.1f}cm  "
          f"float {rec['foot_float_max_cm']:5.1f}cm  "
          f"limit-viol {len(viol_frames):3d}f  vel {vel_peak:5.1f}rad/s  "
          f"{'FLAGS: ' + '; '.join(flags) if flags else 'clean'}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true",
                    help="recompute metrics/series without re-rendering replay videos")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.no_video:
        select_gl_backend()
    rt = G1MujocoRuntime()
    files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no NPZs in {NPZ_DIR}")
    print(f"{len(files)} motions; thresholds: penetration>{PENETRATION_TOL*100:.0f}cm, "
          f"float>{FLOAT_TOL*100:.0f}cm, vel>{VEL_LIMIT}rad/s\n")
    records = [diagnose(os.path.join(NPZ_DIR, f), rt, render=not args.no_video) for f in files]
    with open(os.path.join(OUT, "diagnostics.json"), "w") as fh:
        json.dump(records, fh, indent=2)
    clean = sum(1 for r in records if not r["physics_flags"])
    print(f"\n{clean}/{len(records)} clean; wrote {OUT}/diagnostics.json")
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
