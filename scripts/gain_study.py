#!/usr/bin/env python3
"""Sweep PD gain schemes for standing and record which ones actually work.

This exists because the brief's starting gains (kp 80/40/30, kd 2.5) do not
stand, and the reason is worth recording rather than quietly tuning away: with
no joint armature the light distal joints violate the explicit-damping stability
bound kd * dt / I < 2 and diverge. The sweep is run with and without armature so
the failure and the fix are both visible in the output CSV.

Writes results_sim/gain_study.csv. Every column is measured, none assumed.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.mujoco_runtime import G1MujocoRuntime, PDGains

STAND_SECONDS = 3.0
MIN_HEIGHT = 0.5


def trial(gains: PDGains, armature: float | None) -> dict:
    rt = G1MujocoRuntime(gains=gains, armature=armature)
    rt.reset()
    target = rt.default_joint_targets()
    heights, taus = [], []
    for _ in range(int(STAND_SECONDS * rt.CONTROL_HZ)):
        tau = rt.control_step(target)
        heights.append(rt.base_height)
        taus.append(float(np.linalg.norm(tau)))
    contacts = rt.foot_contacts()
    stands = (
        min(heights) > MIN_HEIGHT
        and heights[-1] > MIN_HEIGHT
        and contacts.left_contact
        and contacts.right_contact
    )
    worst_ratio = float(np.max(rt.kd * rt.dt / rt.joint_inertia)) if np.any(rt.joint_inertia) else float("nan")
    out = {
        "armature": "none" if armature is None else f"{armature}",
        "final_base_z": round(heights[-1], 4),
        "min_base_z": round(min(heights), 4),
        "upright_cos": round(rt.torso_upright_cos(), 4),
        "mean_torque_norm": round(float(np.mean(taus)), 2),
        "peak_torque_norm": round(float(np.max(taus)), 2),
        "max_kd_dt_over_I": round(worst_ratio, 4),
        "both_feet_contact": int(contacts.left_contact and contacts.right_contact),
        "stands": int(stands),
    }
    rt.close()
    return out


def main() -> int:
    rows = []

    # Scheme A: the brief's flat group gains.
    for kp_leg, kp_waist, kp_arm, kd in [
        (80, 40, 30, 2.5),
        (150, 80, 40, 4.0),
        (300, 150, 60, 8.0),
        (600, 300, 100, 15.0),
    ]:
        for armature in (None, G1MujocoRuntime.DEFAULT_ARMATURE):
            g = PDGains(kp_leg=kp_leg, kp_waist=kp_waist, kp_arm=kp_arm,
                        kd_leg=kd, kd_waist=kd, kd_arm=kd, inertia_scaled=False)
            r = trial(g, armature)
            r.update({"scheme": "flat", "param": f"kp={kp_leg}/{kp_waist}/{kp_arm} kd={kd}"})
            rows.append(r)
            print(f"flat  kp={kp_leg:4d}/{kp_waist:3d}/{kp_arm:3d} kd={kd:5.1f} "
                  f"arm={r['armature']:5s} -> z={r['final_base_z']:.4f} "
                  f"kd*dt/I={r['max_kd_dt_over_I']:.3f} stands={r['stands']}")

    # Scheme B: gains derived from the mass-matrix diagonal.
    for omega in (16.0, 26.0, 40.0):
        for zeta in (1.0, 1.5):
            for armature in (None, G1MujocoRuntime.DEFAULT_ARMATURE):
                g = PDGains(inertia_scaled=True, omega=omega, zeta=zeta)
                r = trial(g, armature)
                r.update({"scheme": "inertia_scaled", "param": f"omega={omega} zeta={zeta}"})
                rows.append(r)
                print(f"inert omega={omega:5.1f} zeta={zeta:3.1f}      "
                      f"arm={r['armature']:5s} -> z={r['final_base_z']:.4f} "
                      f"kd*dt/I={r['max_kd_dt_over_I']:.3f} stands={r['stands']}")

    # Scheme C: kp proportional to the joint's torque limit, kd from inertia.
    for alpha in (4.0, 8.0, 16.0):
        for armature in (None, G1MujocoRuntime.DEFAULT_ARMATURE):
            g = PDGains(inertia_scaled=False)
            rt_probe = G1MujocoRuntime(gains=g, armature=armature)
            g_used = PDGains(inertia_scaled=False)
            r = trial_torque_scaled(alpha, armature)
            r.update({"scheme": "torque_scaled", "param": f"alpha={alpha}"})
            rows.append(r)
            rt_probe.close()
            print(f"torq  alpha={alpha:5.1f}              "
                  f"arm={r['armature']:5s} -> z={r['final_base_z']:.4f} "
                  f"kd*dt/I={r['max_kd_dt_over_I']:.3f} stands={r['stands']}")

    os.makedirs("results_sim", exist_ok=True)
    path = "results_sim/gain_study.csv"
    fields = ["scheme", "param", "armature", "stands", "final_base_z", "min_base_z",
              "upright_cos", "both_feet_contact", "mean_torque_norm", "peak_torque_norm",
              "max_kd_dt_over_I"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    winners = [r for r in rows if r["stands"]]
    print(f"\nwrote {path}: {len(rows)} configurations, {len(winners)} stand")
    if winners:
        best = min(winners, key=lambda r: r["mean_torque_norm"])
        print(f"lowest-effort standing config: {best['scheme']} {best['param']} "
              f"armature={best['armature']} mean|tau|={best['mean_torque_norm']} Nm")
    else:
        print("NO configuration stands - this is a negative result, report it as such")
    return 0


def trial_torque_scaled(alpha: float, armature: float | None) -> dict:
    """kp = alpha * torque_limit, kd critically damped and stability-clamped."""
    rt = G1MujocoRuntime(gains=PDGains(inertia_scaled=True), armature=armature)
    rt.kp = alpha * rt.torque_limit
    kd = 2.0 * 1.1 * np.sqrt(np.maximum(rt.kp, 1e-9) * rt.joint_inertia)
    rt.kd = np.minimum(kd, rt.joint_inertia / rt.dt)
    rt.reset()
    target = rt.default_joint_targets()
    heights, taus = [], []
    for _ in range(int(STAND_SECONDS * rt.CONTROL_HZ)):
        tau = rt.control_step(target)
        heights.append(rt.base_height)
        taus.append(float(np.linalg.norm(tau)))
    c = rt.foot_contacts()
    stands = min(heights) > MIN_HEIGHT and heights[-1] > MIN_HEIGHT and c.left_contact and c.right_contact
    out = {
        "armature": "none" if armature is None else f"{armature}",
        "final_base_z": round(heights[-1], 4),
        "min_base_z": round(min(heights), 4),
        "upright_cos": round(rt.torso_upright_cos(), 4),
        "mean_torque_norm": round(float(np.mean(taus)), 2),
        "peak_torque_norm": round(float(np.max(taus)), 2),
        "max_kd_dt_over_I": round(float(np.max(rt.kd * rt.dt / rt.joint_inertia)), 4),
        "both_feet_contact": int(c.left_contact and c.right_contact),
        "stands": int(stands),
    }
    rt.close()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
