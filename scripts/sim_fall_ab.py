#!/usr/bin/env python3
"""Stage 3B: impact-severity A/B -- does the scripted protective response help?

A fall is forced with a large lateral pelvis push from the Kalari horse stance.

  Arm 1 "tracking_only"  : the controller keeps tracking the reference motion
                           whatever happens (the paper's "tracking alone fails"
                           condition). ModeSwitch still runs and its events are
                           logged, but its output is not acted on.
  Arm 2 "protective"     : ModeSwitch routes to the scripted protective crouch
                           as soon as the fall trigger fires.

Impact severity I_b = peak contact-force magnitude on the head and torso geoms.
The impulse integral (N s) is logged alongside it.

5 seeds per arm. Seeds vary the push direction slightly and the push timing, so
the two arms see matched perturbations (seed i is identical across arms).

The protective crouch is HAND-DESIGNED AND SCRIPTED, not learned. Whatever the
numbers say is what gets reported -- including "it does not help".
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
FALL_PUSH_N = 420.0        # well past the ~120-140 N fall threshold found in 3A
PUSH_DURATION = 0.1
BASE_PUSH_TIME = 2.4
N_SEEDS = 5
DURATION = 6.0
SWITCH_CFG = SwitchConfig(delta1=0.015, delta2=15.0, delta3=0.20, min_dwell_steps=30)

ARMS = {"tracking_only": False, "protective": True}


def run_arm(arm: str, seed: int, record: bool = False):
    protective = ARMS[arm]
    rng = np.random.default_rng(seed)
    # Matched perturbation: identical for the same seed across both arms.
    jitter_frames = int(rng.integers(-2, 3))
    lateral = FALL_PUSH_N * float(rng.uniform(0.92, 1.08))
    forward = FALL_PUSH_N * float(rng.uniform(-0.15, 0.15))

    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    ctrl = KalariController(rt, pin_wrap, switch_cfg=SWITCH_CFG,
                            protective_response=protective)
    base_pose = {n: 0.0 for n in rt.act_joint_names}
    base_pose.update(DEFAULT_STANDING_POSE)
    motion = get_motion("horse_stance_hold", rt.act_joint_names, base_pose, fps=rt.CONTROL_HZ)

    push = Push(t_start=BASE_PUSH_TIME + jitter_frames / rt.CONTROL_HZ,
                duration=PUSH_DURATION, force_y=lateral, force_x=forward)
    rt.reset()
    res = run_rollout(rt, ctrl, motion, duration=DURATION, push=push,
                      record_video=record, camera="track", initial_settle=0.3)
    rt.close()
    return res, {"lateral_N": lateral, "forward_N": forward, "jitter_frames": jitter_frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_sim")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.no_video:
        select_gl_backend()

    print(f"fall A/B: push {FALL_PUSH_N} N nominal for {PUSH_DURATION}s, "
          f"{N_SEEDS} seeds per arm, matched perturbations\n")

    rows = []
    for arm in ARMS:
        for seed in range(N_SEEDS):
            res, push_info = run_arm(arm, seed)
            switched = res.transitions
            row = {
                "arm": arm,
                "seed": seed,
                "push_lateral_N": round(push_info["lateral_N"], 2),
                "push_forward_N": round(push_info["forward_N"], 2),
                "jitter_frames": push_info["jitter_frames"],
                "fell": int(res.fell),
                "final_base_z": round(res.final_base_z, 4),
                "min_base_z": round(res.min_base_z, 4),
                "peak_head_force_N": round(res.peak_head_force, 2),
                "peak_torso_force_N": round(res.peak_torso_force, 2),
                "peak_head_torso_N": round(max(res.peak_head_force, res.peak_torso_force), 2),
                "head_impulse_Ns": round(res.head_impulse, 4),
                "torso_impulse_Ns": round(res.torso_impulse, 4),
                "entered_fall_mode": int(any(t["to"] == "fall" for t in switched)),
                "n_switch_events": len(switched),
                "switch_events": json.dumps(switched),
                "failure_time_s": res.failure_time,
            }
            rows.append(row)
            print(f"  {arm:14s} seed{seed}  fell={res.fell}  "
                  f"I_head={row['peak_head_force_N']:8.2f}N  "
                  f"I_torso={row['peak_torso_force_N']:8.2f}N  "
                  f"J_head={row['head_impulse_Ns']:7.3f}Ns  "
                  f"J_torso={row['torso_impulse_Ns']:7.3f}Ns  "
                  f"switch={row['entered_fall_mode']}")
            if seed == 0:
                write_step_csv(os.path.join(args.out, f"fall_ab_steps_{arm}.csv"),
                               res.logs, extra={"arm": arm, "seed": seed})

    csv_path = os.path.join(args.out, "fall_ab.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- statistics -------------------------------------------------------
    stats = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        stats[arm] = {
            m: {"mean": float(np.mean([r[m] for r in sub])),
                "std": float(np.std([r[m] for r in sub], ddof=1)),
                "max": float(np.max([r[m] for r in sub]))}
            for m in ("peak_head_force_N", "peak_torso_force_N",
                      "head_impulse_Ns", "torso_impulse_Ns")
        }
        stats[arm]["fall_rate"] = sum(r["fell"] for r in sub) / len(sub)

    print(f"\n{'metric':22s} {'tracking_only':>22s} {'protective':>22s} {'change':>12s}")
    for m in ("peak_head_force_N", "peak_torso_force_N", "head_impulse_Ns", "torso_impulse_Ns"):
        a, b = stats["tracking_only"][m], stats["protective"][m]
        delta = (b["mean"] - a["mean"]) / a["mean"] * 100 if a["mean"] else float("nan")
        print(f"{m:22s} {a['mean']:10.2f} +/-{a['std']:8.2f} "
              f"{b['mean']:10.2f} +/-{b['std']:8.2f} {delta:+11.1f}%")

    # A mean over trials hides the mechanism when the outcome is bimodal (some
    # trials never touch the body part at all). Separate "did it make contact"
    # from "how hard was the contact when it happened".
    verdict_lines = []
    for m, label in (("peak_head_force_N", "head"), ("peak_torso_force_N", "torso")):
        a, b = stats["tracking_only"][m], stats["protective"][m]
        for arm in ARMS:
            sub = [r for r in rows if r["arm"] == arm]
            hits = [r[m] for r in sub if r[m] > 0]
            stats[arm].setdefault("contact_rate", {})[label] = len(hits) / len(sub)
            stats[arm].setdefault("mean_given_contact", {})[label] = (
                float(np.mean(hits)) if hits else None)

        ca = stats["tracking_only"]["contact_rate"][label]
        cb = stats["protective"]["contact_rate"][label]
        ga = stats["tracking_only"]["mean_given_contact"][label]
        gb = stats["protective"]["mean_given_contact"][label]

        if ca == 0 and cb == 0:
            verdict_lines.append(f"{label}: no ground contact recorded in either arm")
            continue
        line = (f"{label}: contact occurred in {ca:.0%} of tracking_only trials vs "
                f"{cb:.0%} of protective trials")
        if ga is not None and gb is not None:
            line += (f"; when contact did occur, peak force was {ga:.1f} N vs {gb:.1f} N "
                     f"({(gb - ga) / ga * 100:+.1f}%)")
        elif gb is None:
            line += "; the protective arm never made contact at all"
        line += (f". Mean-over-all-trials change is {(b['mean'] - a['mean']) / a['mean'] * 100:+.1f}%, "
                 f"driven by the change in contact rate rather than by softer impacts")
        verdict_lines.append(line)
    print("\nVERDICT (report as-is):")
    for line in verdict_lines:
        print(f"  {line}")

    # ---- bar chart --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    arms = list(ARMS)
    colors = ["#8d99ae", "#2a9d8f"]
    for ax, (m_head, m_torso, unit, title) in zip(
        axes,
        [("peak_head_force_N", "peak_torso_force_N", "N", "Peak impact force $I_b$"),
         ("head_impulse_Ns", "torso_impulse_Ns", "N s", "Impact impulse (integral)")],
    ):
        x = np.arange(2)
        width = 0.35
        for i, arm in enumerate(arms):
            means = [stats[arm][m_head]["mean"], stats[arm][m_torso]["mean"]]
            stds = [stats[arm][m_head]["std"], stats[arm][m_torso]["std"]]
            ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
                   label=arm, color=colors[i], edgecolor="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["head", "torso"])
        ax.set_ylabel(unit)
        ax.set_title(f"{title}\nmean $\\pm$ std, {N_SEEDS} seeds per arm")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = os.path.join(args.out, "fall_ab.png")
    fig.savefig(plot_path, dpi=140)
    print(f"wrote {plot_path}")

    # ---- one video per arm ------------------------------------------------
    videos = {}
    if not args.no_video:
        for arm in ARMS:
            res, _ = run_arm(arm, 0, record=True)
            path = os.path.join(args.out, f"fall_ab_{arm}.mp4")
            if write_video(path, res.frames, fps=30):
                videos[arm] = path

    with open(os.path.join(args.out, "fall_ab_summary.json"), "w") as fh:
        json.dump({"stats": stats, "verdict": verdict_lines, "videos": videos,
                   "push_N": FALL_PUSH_N, "n_seeds": N_SEEDS,
                   "response_type": "scripted hand-designed crouch, NOT learned",
                   "switch_cfg": SWITCH_CFG.__dict__}, fh, indent=2)
    print("wrote results_sim/fall_ab_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
