#!/usr/bin/env python3
"""Stage 3A: lateral push-recovery sweep in real MuJoCo contact dynamics.

DEVIATION FROM THE ORIGINAL PLAN (stated up front, see DEMO_REPORT.md):
The plan was to push during the single-support phase of single_leg_front_kick.
That phase does not exist in this build: under joint-space PD tracking the robot
cannot balance on one leg at all (scripts/sim_track_motion.py records the fall,
and a 36-pose grid search over hip-roll / ankle-roll / waist-roll found no static
single-leg posture that survives 2.5 s). The documented fallback is to run the
sweep from a static stance instead. The deepest stance that IS stable is the
Kalaripayattu horse stance (double support), so the sweep is run from there.

Everything else is as specified: a lateral pelvis push via data.xfrc_applied for
0.1 s, forces [0, 20, 40, 60, 80, 100] N, 3 trials each with the push timing
jittered by +/-2 control frames, logging the CP-margin trace, switch events and
whether the robot fell.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.sim.controller import KalariController
from src.sim.motions import get_motion
from src.sim.mujoco_runtime import DEFAULT_STANDING_POSE, G1MujocoRuntime, select_gl_backend
from src.sim.rollout import Push, run_rollout, write_step_csv, write_video
from src.switch.mode_switch import SwitchConfig

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
# The brief specifies [0, 20, 40, 60, 80, 100] N. Run at that range the horse
# stance never falls (0/3 at every force), so the curve is degenerate and shows
# nothing. The specified points are all kept, and the sweep is extended past
# 100 N to locate the actual fall threshold. A verification run confirmed the
# push is real: 100 N for 0.1 s produces a measured peak CoM velocity of
# 0.259 m/s against 10 N s / 33.34 kg = 0.300 m/s predicted.
SPEC_FORCES = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
FORCES = SPEC_FORCES + [120.0, 140.0, 160.0, 180.0, 200.0, 240.0]
TRIALS = 3
PUSH_DURATION = 0.1
BASE_PUSH_TIME = 2.4          # inside the horse-stance hold window (1.6 s - 3.4 s)
FRAME_JITTER = [-2, 0, 2]     # control frames, per the brief

# ModeSwitch delta1 must sit below the CP margin the robot genuinely holds in a
# deep stance, otherwise it fires on every stance and the protective crouch drops
# a robot that was never falling. Measured stable horse-stance minimum is about
# +0.026 m (results_sim/track_horse_stance_hold.csv), so 0.015 m leaves headroom
# without disabling the trigger. Reported explicitly rather than silently tuned.
SWITCH_CFG = SwitchConfig(delta1=0.015, delta2=15.0, delta3=0.20, min_dwell_steps=30)


def build(protective: bool = True):
    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    ctrl = KalariController(rt, pin_wrap, switch_cfg=SWITCH_CFG,
                            protective_response=protective)
    base_pose = {n: 0.0 for n in rt.act_joint_names}
    base_pose.update(DEFAULT_STANDING_POSE)
    motion = get_motion("horse_stance_hold", rt.act_joint_names, base_pose, fps=rt.CONTROL_HZ)
    return rt, ctrl, motion


def one_trial(force: float, jitter_frames: int, record: bool = False):
    rt, ctrl, motion = build()
    push_t = BASE_PUSH_TIME + jitter_frames / rt.CONTROL_HZ
    push = Push(t_start=push_t, duration=PUSH_DURATION, force_y=force)
    rt.reset()
    res = run_rollout(rt, ctrl, motion, duration=5.0, push=push,
                      record_video=record, camera="front", initial_settle=0.3)
    rt.close()
    return res, push_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_sim")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.no_video:
        select_gl_backend()

    rows = []
    traces = {}
    recovered_example = fallen_example = None

    print(f"push sweep: forces {FORCES} N, {TRIALS} trials each, "
          f"{PUSH_DURATION}s lateral pelvis push at t={BASE_PUSH_TIME}s +/- {FRAME_JITTER} frames")
    print(f"ModeSwitch: delta1={SWITCH_CFG.delta1} delta2={SWITCH_CFG.delta2} "
          f"delta3={SWITCH_CFG.delta3} min_dwell={SWITCH_CFG.min_dwell_steps}\n")

    for force in FORCES:
        for trial_i, jitter in enumerate(FRAME_JITTER[:TRIALS]):
            res, push_t = one_trial(force, jitter)
            cp = np.array([l.cp_margin for l in res.logs])
            valid = cp[cp > -900]
            post = [l for l in res.logs if l.t >= push_t]
            cp_post = np.array([l.cp_margin for l in post])
            cp_post = cp_post[cp_post > -900]
            switched = [t for t in res.transitions]

            row = {
                "force_N": force,
                "trial": trial_i,
                "jitter_frames": jitter,
                "push_time_s": round(push_t, 4),
                "push_duration_s": PUSH_DURATION,
                "fell": int(res.fell),
                "final_base_z": round(res.final_base_z, 4),
                "min_base_z": round(res.min_base_z, 4),
                "cp_margin_min": round(float(valid.min()), 5) if len(valid) else None,
                "cp_margin_min_after_push": round(float(cp_post.min()), 5) if len(cp_post) else None,
                "cp_margin_mean": round(float(valid.mean()), 5) if len(valid) else None,
                "n_switch_events": len(switched),
                "switch_events": json.dumps(switched),
                "entered_fall_mode": int(any(t["to"] == "fall" for t in switched)),
                "peak_grf_N": round(max(l.grf_total for l in res.logs), 1),
                "peak_torque_norm_Nm": round(max(l.torque_norm for l in res.logs), 2),
                "failure_time_s": res.failure_time,
            }
            rows.append(row)
            traces[(force, trial_i)] = np.array([[l.t, l.cp_margin] for l in res.logs])

            print(f"  F={force:6.1f}N trial{trial_i} jitter{jitter:+d}  fell={res.fell}  "
                  f"min_z={res.min_base_z:.3f}  cp_min_after_push="
                  f"{row['cp_margin_min_after_push']}  switches={len(switched)}")

            if not args.no_video:
                if not res.fell and recovered_example is None and force >= 40:
                    recovered_example = (force, trial_i, jitter)
                if res.fell and fallen_example is None:
                    fallen_example = (force, trial_i, jitter)

            if force == FORCES[-1] and trial_i == 0:
                write_step_csv(os.path.join(args.out, "push_example_steps.csv"), res.logs,
                               extra={"force_N": force, "trial": trial_i})

    csv_path = os.path.join(args.out, "push_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- summary: fall rate vs force -------------------------------------
    print(f"\n{'force (N)':>10} {'fall rate':>10} {'mean cp_min after push (m)':>28}")
    summary = []
    for force in FORCES:
        sub = [r for r in rows if r["force_N"] == force]
        rate = sum(r["fell"] for r in sub) / len(sub)
        cps = [r["cp_margin_min_after_push"] for r in sub if r["cp_margin_min_after_push"] is not None]
        summary.append({"force_N": force, "fall_rate": rate,
                        "mean_cp_min_after_push": float(np.mean(cps)) if cps else float("nan"),
                        "n": len(sub)})
        print(f"{force:10.1f} {rate:10.2f} {np.mean(cps) if cps else float('nan'):28.5f}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot([s["force_N"] for s in summary], [s["fall_rate"] for s in summary],
            "o-", color="#c1121f", lw=2, ms=7)
    ax.set_xlabel("lateral pelvis push (N, 0.1 s)")
    ax.set_ylabel(f"fall rate ({TRIALS} trials per force)")
    ax.set_title("Push magnitude vs fall rate\nKalari horse stance, double support")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(FORCES)))
    for color, force in zip(cmap, FORCES):
        tr = traces[(force, 0)]
        m = tr[:, 1] > -900
        ax.plot(tr[m, 0], tr[m, 1], color=color, lw=1.6, label=f"{force:.0f} N")
    ax.axvline(BASE_PUSH_TIME, color="k", ls="--", lw=1, label="push")
    ax.axhline(SWITCH_CFG.delta1, color="#c1121f", ls=":", lw=1.2,
               label=f"switch $\\delta_1$={SWITCH_CFG.delta1}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("capture-point margin (m)")
    ax.set_title("CP margin trace (trial 0)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.out, "push_sweep.png")
    fig.savefig(plot_path, dpi=140)
    print(f"wrote {plot_path}")

    # ---- example videos --------------------------------------------------
    videos = {}
    if not args.no_video:
        for label, example in (("recovered", recovered_example), ("fallen", fallen_example)):
            if example is None:
                print(f"[video] no {label} trial to record")
                continue
            force, trial_i, jitter = example
            res, _ = one_trial(force, jitter, record=True)
            path = os.path.join(args.out, f"push_{label}_{int(force)}N.mp4")
            if write_video(path, res.frames, fps=30):
                videos[label] = {"path": path, "force_N": force, "trial": trial_i}

    with open(os.path.join(args.out, "push_sweep_summary.json"), "w") as fh:
        json.dump({"summary": summary, "videos": videos,
                   "switch_cfg": SWITCH_CFG.__dict__,
                   "push_duration_s": PUSH_DURATION,
                   "stance": "horse_stance_hold (double support)",
                   "deviation_note": (
                       "Push applied in the horse-stance hold, not the kick's single-support "
                       "phase: PD tracking cannot hold single support in this build.")},
                  fh, indent=2)
    print("wrote results_sim/push_sweep_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
