#!/usr/bin/env python3
"""Paper Figure 2: physical challenges rendered from real simulation states.

Fills the arXiv draft's Figure 2 placeholder ("four representative motion frames
with overlays for CoM and support polygon, capture point during one-leg balance,
... a recovery-critical state near the boundary of the recoverable set").

Each panel is a REAL MuJoCo frame from the validated runtime, with the physics
quantities computed by the same PinocchioWrapper/StateConverter path the
controller uses (max cross-engine CoM error 1e-6 m, see tests/test_conventions.py).
Nothing is drawn from intuition: polygon vertices, CoM, and capture point are the
runtime's own numbers for the rendered state.

Panels:
  A  deep horse stance, double support      -- CoM + support polygon + R_cp
  B  push at the fall threshold (120 N)     -- capture point escaping support
  C  recovered push (100 N)                 -- capture point held inside
  D  post-fall state                        -- protective crouch, near V_rec boundary

Output: results_paper/fig2_physical_challenges.png (+ .pdf) and per-panel PNGs.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.sim.controller import KalariController
from src.sim.motions import get_motion
from src.sim.mujoco_runtime import DEFAULT_STANDING_POSE, G1MujocoRuntime, select_gl_backend
from src.sim.rollout import Push, run_rollout
from src.switch.mode_switch import SwitchConfig

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
SWITCH_CFG = SwitchConfig(delta1=0.015, delta2=15.0, delta3=0.20, min_dwell_steps=30)
OUT = "results_paper"

# World -> pixel mapping is done by projecting ground-plane points through the
# camera; simpler and sufficient here: draw the physics overlay in a ground-plane
# inset axis beside the render rather than warping into camera space.


def run_and_capture(push_n: float | None, capture_t: float, duration: float = 5.0):
    """Run the standard horse-stance rollout, return (frame, ground-truth physics)."""
    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    ctrl = KalariController(rt, pin_wrap, switch_cfg=SWITCH_CFG)
    base = {n: 0.0 for n in rt.act_joint_names}
    base.update(DEFAULT_STANDING_POSE)
    motion = get_motion("horse_stance_hold", rt.act_joint_names, base, fps=rt.CONTROL_HZ)
    ctrl.bind_motion(motion)
    ctrl.reset()
    rt.reset()
    rt.settle(0.3)

    push = Push(2.4, 0.1, push_n) if push_n else None
    n_ticks = int(duration * rt.CONTROL_HZ)
    cap_tick = int(capture_t * rt.CONTROL_HZ)
    frame, physics = None, None
    xfrc = np.zeros_like(rt.data.xfrc_applied)

    for k in range(n_ticks):
        t = k / rt.CONTROL_HZ
        xfrc[:] = 0.0
        if push and push.active(t):
            xfrc[rt.pelvis_body, 1] = push.force_y
        log = ctrl.measure(k, t, motion)
        q_cmd, mode = ctrl.targets(log, motion, t)
        log.mode = mode.value
        rt.control_step(q_cmd, xfrc=xfrc)

        if k == cap_tick:
            frame = rt.render_frame(camera="front", width=640, height=480)
            contacts = rt.foot_contacts()
            pts = rt.foot_support_points(contacts)
            q_pin = ctrl.conv.mj_to_pin_q(np.array(rt.data.qpos))
            v_pin = ctrl.conv.mj_to_pin_v(np.array(rt.data.qpos), np.array(rt.data.qvel))
            feats = pin_wrap.get_support_features(
                q_pin, v_pin, np.array([contacts.left_contact, contacts.right_contact]),
                support_points_world=pts if len(pts) else None)
            com, com_vel = pin_wrap.compute_com(q_pin, v_pin)
            physics = {
                "polygon": np.asarray(feats["support_polygon"], dtype=float),
                "com": com, "com_vel": com_vel,
                "cp": np.asarray(feats["capture_point"], dtype=float),
                "cp_margin": feats["cp_margin"],
                "mode": mode.value, "t": t,
                "base_z": rt.base_height,
            }
    rt.close()
    return frame, physics


def draw_panel(ax_img, ax_geo, frame, phys, title: str):
    ax_img.imshow(frame)
    ax_img.set_title(title, fontsize=10)
    ax_img.axis("off")

    poly = phys["polygon"]
    if len(poly) >= 3:
        ax_geo.add_patch(MplPolygon(poly, closed=True, facecolor="#2a9d8f33",
                                    edgecolor="#2a9d8f", lw=1.6,
                                    label="support polygon"))
    com = phys["com"]
    cp = phys["cp"]
    ax_geo.plot(com[0], com[1], "o", color="#1d3557", ms=9, label="CoM (xy)")
    ax_geo.plot(cp[0], cp[1], "X", color="#c1121f", ms=11,
                label=r"capture point $\xi$")
    ax_geo.annotate("", xy=(cp[0], cp[1]), xytext=(com[0], com[1]),
                    arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.4))
    m = phys["cp_margin"]
    ax_geo.set_title(f"$R_{{cp}}$ = {m:+.3f} m   mode: {phys['mode']}", fontsize=9)
    ax_geo.set_aspect("equal")
    ax_geo.grid(alpha=0.25, lw=0.5)
    ax_geo.tick_params(labelsize=7)

    allx = list(poly[:, 0]) + [com[0], cp[0]] if len(poly) else [com[0], cp[0]]
    ally = list(poly[:, 1]) + [com[1], cp[1]] if len(poly) else [com[1], cp[1]]
    pad = 0.08
    ax_geo.set_xlim(min(allx) - pad, max(allx) + pad)
    ax_geo.set_ylim(min(ally) - pad, max(ally) + pad)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    select_gl_backend()

    specs = [
        ("A  deep stance (double support)", None, 2.2),
        ("B  120 N push: CP escaping support", 120.0, 3.00),
        ("C  100 N push: CP held inside", 100.0, 2.62),
        ("D  post-fall protective state", 200.0, 4.40),
    ]

    panels = []
    for title, push_n, t_cap in specs:
        print(f"rendering: {title} (push={push_n}, t={t_cap}s)")
        frame, phys = run_and_capture(push_n, t_cap)
        if frame is None or phys is None:
            raise SystemExit(f"capture failed for panel '{title}'")
        print(f"  R_cp={phys['cp_margin']:+.4f} m  mode={phys['mode']}  base_z={phys['base_z']:.3f}")
        panels.append((title, frame, phys))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2),
                             gridspec_kw={"height_ratios": [2.2, 1.4]})
    handles = labels = None
    for i, (title, frame, phys) in enumerate(panels):
        draw_panel(axes[0][i], axes[1][i], frame, phys, title)
        if handles is None:
            handles, labels = axes[1][i].get_legend_handles_labels()
        # per-panel export for LaTeX flexibility
        sub, subax = plt.subplots(2, 1, figsize=(4.4, 6.6),
                                  gridspec_kw={"height_ratios": [2.2, 1.4]})
        draw_panel(subax[0], subax[1], frame, phys, title)
        sub.tight_layout()
        sub.savefig(os.path.join(OUT, f"fig2_panel_{chr(97 + i)}.png"), dpi=150)
        plt.close(sub)

    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Physical challenges in Kalaripayattu execution "
                 "(real MuJoCo states; physics from the validated Pinocchio runtime)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"fig2_physical_challenges.{ext}")
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
