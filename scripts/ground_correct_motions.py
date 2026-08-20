#!/usr/bin/env python3
"""Ground-contact correction of retargeted NPZs (preprocessing report sections 1-7).

Fixes the two defects motion_diagnostics.py measured on the raw GEM-X retargets:

  1. FOOT FLOATING (11/12 motions, 5-23 cm): the root track carries a smooth
     height bias. Correction: compute the lowest foot-sphere height per frame by
     forward kinematics in the official G1 model, low-pass it with a
     Savitzky-Golay filter (so genuine vertical dynamics such as kicks survive),
     and subtract that smoothed offset from the root height, leaving a small
     clearance. This is the report's "grounded height = -z_min" correction with
     smoothing in place of per-frame clamping.

  2. ARM VELOCITY SPIKES (12-18 rad/s at wrist/shoulder/elbow): retarget jitter.
     Correction: Savitzky-Golay smoothing (window 9, order 3 at 30 fps) on all
     joint columns - mild enough to preserve real kick speed (~10 rad/s).

Also recomputes per-foot CONTACT labels (threshold 4 cm, median-filtered), which
were all-false on the raw data because the feet never came near the floor.

Raw NPZs are preserved in data/motions_retargeted_raw/ - the correction is
re-runnable and reversible, and the page shows before vs after honestly.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from scipy.signal import medfilt, savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime

NPZ_DIR = "data/motions_retargeted"
RAW_DIR = "data/motions_retargeted_raw"
CLEARANCE = 0.002        # m left under the lowest sphere after correction
CONTACT_Z = 0.04         # m, contact threshold (preprocessing report value)
JOINT_SG_WIN = 9         # frames, order 3 (mild; keeps ~10 rad/s kicks)


def foot_heights(rt: G1MujocoRuntime, root_pos, root_quat_xyzw, joint_pos,
                 col_for_act) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame lowest-sphere height: overall, left foot, right foot."""
    n = joint_pos.shape[0]
    default_q = rt.default_qpos()
    lowest = np.zeros(n)
    left = np.zeros(n)
    right = np.zeros(n)
    for k in range(n):
        qpos = default_q.copy()
        qpos[0:3] = root_pos[k]
        qpos[3:7] = quat_xyzw_to_wxyz(root_quat_xyzw[k])
        for a in range(rt.model.nu):
            if col_for_act[a] >= 0:
                qpos[rt.act_qadr[a]] = joint_pos[k, col_for_act[a]]
        rt.data.qpos[:] = qpos
        rt.mujoco.mj_forward(rt.model, rt.data)
        heights = {g: float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                   for g in rt.left_foot_geoms + rt.right_foot_geoms}
        lowest[k] = min(heights.values())
        left[k] = min(heights[g] for g in rt.left_foot_geoms)
        right[k] = min(heights[g] for g in rt.right_foot_geoms)
    return lowest, left, right


def odd_window(n: int, target: int) -> int:
    w = min(target, n - 1 if n % 2 == 0 else n - 2)
    if w % 2 == 0:
        w -= 1
    return max(w, 5)


def correct_one(path: str, rt: G1MujocoRuntime) -> dict:
    data = dict(np.load(path, allow_pickle=True))
    mid = os.path.splitext(os.path.basename(path))[0]

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    quat = np.asarray(data["root_quat_xyzw"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    n = joint_pos.shape[0]

    cols = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
            for c in np.asarray(data["joint_cols"]).reshape(-1)]
    stripped = [c.removesuffix("_dof") for c in cols]
    idx = {nme: i for i, nme in enumerate(stripped)}
    col_for_act = np.array([idx.get(j, -1) for j in rt.act_joint_names])

    # --- 1) joint smoothing (before height correction, so FK uses final joints)
    win_j = odd_window(n, JOINT_SG_WIN)
    joint_s = savgol_filter(joint_pos, win_j, 3, axis=0)

    # --- 2) root-height correction from smoothed FK foot height --------------
    lowest, _, _ = foot_heights(rt, root_pos, quat, joint_s, col_for_act)
    win_h = odd_window(n, 31)
    offset = savgol_filter(lowest, win_h, 3)
    root_c = root_pos.copy()
    root_c[:, 2] -= offset - CLEARANCE

    # residual pass: clamp any remaining penetration (short transients the
    # smooth offset cannot follow), smoothed lightly to avoid steps
    lowest2, left2, right2 = foot_heights(rt, root_c, quat, joint_s, col_for_act)
    pen = np.minimum(lowest2, 0.0)
    if pen.min() < -0.005:
        root_c[:, 2] -= savgol_filter(pen, odd_window(n, 9), 2)
        lowest2, left2, right2 = foot_heights(rt, root_c, quat, joint_s, col_for_act)

    # --- 3) contacts + derivatives ------------------------------------------
    contacts = np.stack([left2 < CONTACT_Z, right2 < CONTACT_Z], axis=1)
    k_med = min(5, n if n % 2 else n - 1)
    contacts = np.stack([medfilt(contacts[:, 0].astype(float), k_med) > 0.5,
                         medfilt(contacts[:, 1].astype(float), k_med) > 0.5], axis=1)

    q = np.asarray(data["q"], dtype=np.float64)
    q[:, 0:3] = root_c
    q[:, 7:] = joint_s
    dq = np.gradient(q, 1.0 / fps, axis=0)

    data.update({
        "q": q, "dq": dq, "joint_pos": joint_s, "root_pos": root_c,
        "contacts": contacts.astype(bool),
    })
    np.savez(path, **data)

    rep = {
        "motion_id": mid,
        "float_before_cm": round(float(max(0, lowest.max())) * 100, 1),
        "float_after_cm": round(float(max(0, lowest2.max())) * 100, 1),
        "pen_after_cm": round(float(max(0, -lowest2.min())) * 100, 2),
        "contact_frac_after": round(float(contacts.any(axis=1).mean()), 2),
        "joint_sg_window": win_j,
    }
    print(f"  {mid:26s} float {rep['float_before_cm']:5.1f} -> {rep['float_after_cm']:4.1f} cm   "
          f"residual pen {rep['pen_after_cm']:4.2f} cm   "
          f"contact frames {rep['contact_frac_after']:.0%}")
    return rep


def main() -> int:
    files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no NPZs in {NPZ_DIR}")

    os.makedirs(RAW_DIR, exist_ok=True)
    for f in files:
        raw = os.path.join(RAW_DIR, f)
        if not os.path.exists(raw):
            shutil.copy2(os.path.join(NPZ_DIR, f), raw)
    print(f"raw NPZs preserved in {RAW_DIR}/\n")

    rt = G1MujocoRuntime()
    reports = [correct_one(os.path.join(NPZ_DIR, f), rt) for f in files]
    rt.close()

    import json
    with open(os.path.join(NPZ_DIR, "ground_correction_report.json"), "w") as fh:
        json.dump(reports, fh, indent=2)
    ok = sum(1 for r in reports if r["float_after_cm"] < 3 and r["pen_after_cm"] < 1)
    print(f"\n{ok}/{len(reports)} motions now well-grounded; "
          f"report: {NPZ_DIR}/ground_correction_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
