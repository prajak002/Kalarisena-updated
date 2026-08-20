#!/usr/bin/env python3
"""Stage-1 gate: MuJoCo scene + standing + cross-engine agreement.

Three checks, all of which must pass before any experiment is run:

  1. STANDING  - PD control holding the default pose keeps the robot up for 3
                 simulated seconds with base height > 0.5 m.
  2. COM       - Pinocchio CoM and MuJoCo subtree_com agree within 1 cm at 20
                 random states, routed through the explicit converters. This is
                 the real test of conventions (a) quaternion order, (b) joint
                 ordering and (c) velocity frame: any of the three being wrong
                 shows up here.
  3. CONTACTS  - Foot contact flags come from actual MuJoCo contacts on the
                 ankle_roll collision geoms, and both feet are loaded while
                 standing, with total normal force ~= robot weight.

Every number printed below is measured at run time.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics.pinocchio_wrapper import FOOT_CONTACT_OFFSETS, PinocchioWrapper
from src.sim.conventions import StateConverter
from src.sim.mujoco_runtime import G1MujocoRuntime

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
COM_TOL = 0.01  # metres
STAND_SECONDS = 3.0
MIN_BASE_HEIGHT = 0.5


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def check_foot_geometry(rt: G1MujocoRuntime) -> bool:
    """The Pinocchio-side foot offsets must match the MJCF's real spheres."""
    banner("CHECK 0: foot contact geometry (MJCF spheres vs wrapper constant)")
    mj_left = np.array(sorted(rt.left_foot_offsets.tolist()))
    const = np.array(sorted(FOOT_CONTACT_OFFSETS.tolist()))
    print(f"MJCF left foot spheres (body frame), n={len(mj_left)}:")
    for p in mj_left:
        print(f"    {p}")
    err = float(np.abs(mj_left - const).max()) if mj_left.shape == const.shape else np.inf
    span_x = float(mj_left[:, 0].max() - mj_left[:, 0].min())
    span_y = float(mj_left[:, 1].max() - mj_left[:, 1].min())
    print(f"foot footprint from MJCF: {span_x * 100:.1f} cm (x) x {span_y * 100:.1f} cm (y)")
    print(f"max |MJCF - FOOT_CONTACT_OFFSETS| = {err:.6f} m")
    ok = err < 1e-9
    print("PASS" if ok else "FAIL - wrapper foot geometry has drifted from the MJCF")
    return ok


def check_standing(rt: G1MujocoRuntime) -> tuple[bool, float]:
    banner(f"CHECK 1: PD standing for {STAND_SECONDS}s (base height > {MIN_BASE_HEIGHT} m)")
    rt.reset()
    target = rt.default_joint_targets()
    n_ticks = int(round(STAND_SECONDS * rt.CONTROL_HZ))
    heights, torques = [], []
    for k in range(n_ticks):
        tau = rt.control_step(target)
        heights.append(rt.base_height)
        torques.append(float(np.linalg.norm(tau)))
        if k % 25 == 0:
            c = rt.foot_contacts()
            print(
                f"  t={k / rt.CONTROL_HZ:4.2f}s  base_z={rt.base_height:.4f} m  "
                f"|tau|={torques[-1]:6.1f} Nm  contacts L/R={int(c.left_contact)}/{int(c.right_contact)}  "
                f"Fn={c.total_normal_force:6.1f} N"
            )
    final_h = rt.base_height
    min_h = float(np.min(heights))
    print(f"\nphysics dt={rt.dt}s ({1 / rt.dt:.0f} Hz), control {rt.CONTROL_HZ:.0f} Hz "
          f"(decimation {rt.decimation})")
    print(f"final base height : {final_h:.4f} m")
    print(f"minimum base height over the run: {min_h:.4f} m")
    print(f"mean |tau| : {np.mean(torques):.1f} Nm   peak |tau| : {np.max(torques):.1f} Nm")
    ok = final_h > MIN_BASE_HEIGHT and min_h > MIN_BASE_HEIGHT
    print("PASS" if ok else "FAIL - robot did not stay standing")
    return ok, final_h


def check_com_agreement(rt: G1MujocoRuntime, pin_wrap: PinocchioWrapper,
                        conv: StateConverter, n_states: int = 20) -> bool:
    banner(f"CHECK 2: Pinocchio CoM vs MuJoCo subtree_com at {n_states} random states")
    print(conv.describe())
    print(f"MuJoCo total mass {rt.total_mass:.4f} kg   "
          f"Pinocchio total mass {sum(i.mass for i in pin_wrap.model.inertias):.4f} kg")
    print(f"\n{'#':>3} {'MuJoCo CoM (x,y,z)':>34} {'Pinocchio CoM (x,y,z)':>34} {'err (m)':>10}")

    rng = np.random.default_rng(0)
    errors = []
    for i in range(n_states):
        qpos = rt.default_qpos()
        # Randomise base pose and every actuated joint, within joint limits.
        qpos[0:2] += rng.uniform(-0.3, 0.3, size=2)
        qpos[2] += rng.uniform(-0.15, 0.15)
        quat = rng.normal(size=4)
        qpos[3:7] = quat / np.linalg.norm(quat)
        for a in range(rt.model.nu):
            adr = rt.act_qadr[a]
            jid = int(rt.model.actuator_trnid[a, 0])
            lo, hi = rt.model.jnt_range[jid]
            qpos[adr] = rng.uniform(lo, hi) if hi > lo else rng.uniform(-0.5, 0.5)

        qvel = rng.uniform(-1.0, 1.0, size=rt.model.nv)
        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = qvel
        rt.mujoco.mj_forward(rt.model, rt.data)
        com_mj = rt.mj_subtree_com()

        q_pin = conv.mj_to_pin_q(qpos)
        v_pin = conv.mj_to_pin_v(qpos, qvel)
        com_pin, _ = pin_wrap.compute_com(q_pin, v_pin)

        err = float(np.linalg.norm(com_mj - com_pin))
        errors.append(err)
        print(f"{i:3d} [{com_mj[0]:+8.5f} {com_mj[1]:+8.5f} {com_mj[2]:+8.5f}] "
              f"[{com_pin[0]:+8.5f} {com_pin[1]:+8.5f} {com_pin[2]:+8.5f}] {err:10.6f}")

    max_err = float(np.max(errors))
    print(f"\nmax CoM error {max_err:.6f} m   mean {np.mean(errors):.6f} m   tolerance {COM_TOL} m")

    # Velocity-frame check: CoM velocity must also agree, which the position
    # check alone cannot detect. Compared against MuJoCo's own CoM Jacobian.
    jac = np.zeros((3, rt.model.nv))
    rt.mujoco.mj_jacSubtreeCom(rt.model, rt.data, jac, rt.pelvis_body)
    vcom_mj = jac @ rt.data.qvel
    _, vcom_pin = pin_wrap.compute_com(conv.mj_to_pin_q(rt.data.qpos),
                                       conv.mj_to_pin_v(rt.data.qpos, rt.data.qvel))
    vel_err = float(np.linalg.norm(vcom_mj - vcom_pin))
    print(f"CoM velocity check (last state): MuJoCo {vcom_mj}  Pinocchio {vcom_pin}")
    print(f"CoM velocity error {vel_err:.6f} m/s")

    ok = max_err < COM_TOL and vel_err < 0.05
    print("PASS" if ok else "FAIL - engines disagree; check conventions (a)/(b)/(c)")
    return ok


def check_contacts(rt: G1MujocoRuntime) -> bool:
    banner("CHECK 3: foot contacts from real MuJoCo contacts (not a height heuristic)")
    rt.reset()
    rt.settle(1.5)
    c = rt.foot_contacts()
    weight = rt.total_mass * 9.81
    print(f"left foot geoms  {rt.left_foot_geoms}  (4 spheres on left_ankle_roll_link)")
    print(f"right foot geoms {rt.right_foot_geoms}  (4 spheres on right_ankle_roll_link)")
    print(f"left  contact={c.left_contact}  Fn={c.left_force:.1f} N  points={len(c.left_points)}")
    print(f"right contact={c.right_contact}  Fn={c.right_force:.1f} N  points={len(c.right_points)}")
    print(f"total normal force {c.total_normal_force:.1f} N vs robot weight {weight:.1f} N "
          f"(ratio {c.total_normal_force / weight:.3f})")
    support = rt.foot_support_points(c)
    print(f"support points from simulator: {len(support)}")
    ok = c.left_contact and c.right_contact and abs(c.total_normal_force - weight) / weight < 0.20
    print("PASS" if ok else "FAIL - feet not properly loaded")
    return ok


def main() -> int:
    banner("STAGE 1 GATE - MuJoCo scene, standing, cross-engine agreement")
    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    conv = StateConverter(rt.model, pin_wrap.model)

    results = {
        "foot_geometry": check_foot_geometry(rt),
        "standing": check_standing(rt)[0],
        "com_agreement": check_com_agreement(rt, pin_wrap, conv),
        "contacts": check_contacts(rt),
    }

    banner("STAGE 1 GATE SUMMARY")
    for name, ok in results.items():
        print(f"  {name:16s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nGATE: {'PASS - proceed to Stage 2' if all_ok else 'FAIL - do not proceed'}")
    rt.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
