#!/usr/bin/env python3
"""Kinematic replay of the reference motions in MuJoCo (FALLBACK PATH).

THIS IS NOT A CONTROLLED SIMULATION. qpos is written directly from the reference
trajectory each frame and mj_forward is called for rendering only -- no torques,
no contact response, no physics integration. The robot cannot fall here because
nothing is simulated. Use these videos only to show what the reference motion is;
every dynamic claim must come from scripts/sim_track_motion.py, sim_push_sweep.py
or sim_fall_ab.py, which run the real closed loop.

The floating base is placed per frame so the lowest foot contact sphere rests on
the floor, which is itself a kinematic assumption, not a computed contact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.motions import MOTION_IDS, get_motion
from src.sim.mujoco_runtime import DEFAULT_STANDING_POSE, G1MujocoRuntime, select_gl_backend
from src.sim.rollout import write_video

LABEL = "KINEMATIC REPLAY - no physics"


def replay(motion_id: str, out_dir: str, fps: int = 30) -> dict:
    rt = G1MujocoRuntime()
    base_pose = {n: 0.0 for n in rt.act_joint_names}
    base_pose.update(DEFAULT_STANDING_POSE)
    motion = get_motion(motion_id, rt.act_joint_names, base_pose, fps=rt.CONTROL_HZ)

    rt.reset()
    foot_geoms = rt.left_foot_geoms + rt.right_foot_geoms
    frame_every = max(1, int(round(rt.CONTROL_HZ / fps)))
    frames = []

    for k in range(motion.n_frames):
        qpos = rt.default_qpos()
        qpos[rt.act_qadr] = motion.q_ref[k]
        qpos[2] = 1.0
        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = 0.0
        rt.mujoco.mj_forward(rt.model, rt.data)
        lowest = min(float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                     for g in foot_geoms)
        qpos[2] += 0.001 - lowest
        rt.data.qpos[:] = qpos
        rt.mujoco.mj_forward(rt.model, rt.data)
        if k % frame_every == 0:
            frames.append(rt.render_frame(camera="track"))

    path = os.path.join(out_dir, f"replay_{motion_id}.mp4")
    ok = write_video(path, frames, fps=fps)
    print(f"  {motion_id}: {motion.n_frames} frames, {LABEL}")
    rt.close()
    return {"motion_id": motion_id, "video": path if ok else None,
            "frames": motion.n_frames, "source": motion.source, "label": LABEL}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default=None, choices=list(MOTION_IDS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="results_sim")
    args = ap.parse_args()
    if not args.all and args.motion is None:
        ap.error("pass --motion <id> or --all")

    os.makedirs(args.out, exist_ok=True)
    select_gl_backend()
    print(f"*** {LABEL} *** these videos show the reference only, not controlled behaviour")

    ids = list(MOTION_IDS) if args.all else [args.motion]
    out = [replay(m, args.out) for m in ids]
    path = os.path.join(args.out, "replay_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
