#!/usr/bin/env python3
"""Regression tests for the MuJoCo <-> Pinocchio conventions and foot geometry.

Run: .venv/bin/python tests/test_conventions.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics.pinocchio_wrapper import FOOT_CONTACT_OFFSETS, PinocchioWrapper
from src.sim.conventions import (StateConverter, quat_wxyz_to_matrix,
                                 quat_wxyz_to_xyzw, quat_xyzw_to_wxyz)
from src.sim.mujoco_runtime import G1MujocoRuntime

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        _FAILURES.append(name)


def test_quaternion_roundtrip() -> None:
    print("test_quaternion_roundtrip")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        worst = max(worst, float(np.abs(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q)) - q).max()))
    check("wxyz <-> xyzw roundtrip", worst < 1e-12, f"max err {worst:.2e}")

    # Order actually matters: a non-symmetric quaternion must not survive a
    # naive identity mapping, or the converter is a no-op.
    q = np.array([0.1, 0.2, 0.3, 0.9])
    q /= np.linalg.norm(q)
    check("converter is not the identity", not np.allclose(quat_wxyz_to_xyzw(q), q))


def test_rotation_matrix_matches_mujoco() -> None:
    print("test_rotation_matrix_matches_mujoco")
    import mujoco

    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q)
        worst = max(worst, float(np.abs(mat.reshape(3, 3) - quat_wxyz_to_matrix(q)).max()))
    check("quat_wxyz_to_matrix matches mju_quat2Mat", worst < 1e-12, f"max err {worst:.2e}")


def test_state_roundtrip_and_com(rt, pin_wrap, conv) -> None:
    print("test_state_roundtrip_and_com")
    rng = np.random.default_rng(3)
    worst_q = worst_v = worst_com = worst_vcom = 0.0
    for _ in range(30):
        qpos = rt.default_qpos()
        qpos[0:3] += rng.uniform(-0.3, 0.3, size=3)
        quat = rng.normal(size=4)
        qpos[3:7] = quat / np.linalg.norm(quat)
        qpos[rt.act_qadr] = rng.uniform(-0.6, 0.6, size=rt.model.nu)
        qvel = rng.uniform(-1.5, 1.5, size=rt.model.nv)

        worst_q = max(worst_q, float(np.abs(conv.pin_to_mj_q(conv.mj_to_pin_q(qpos)) - qpos).max()))
        v_pin = conv.mj_to_pin_v(qpos, qvel)
        worst_v = max(worst_v, float(np.abs(conv.pin_to_mj_v(qpos, v_pin) - qvel).max()))

        rt.data.qpos[:] = qpos
        rt.data.qvel[:] = qvel
        rt.mujoco.mj_forward(rt.model, rt.data)
        com_mj = rt.mj_subtree_com()
        jac = np.zeros((3, rt.model.nv))
        rt.mujoco.mj_jacSubtreeCom(rt.model, rt.data, jac, rt.pelvis_body)
        vcom_mj = jac @ qvel

        com_pin, vcom_pin = pin_wrap.compute_com(conv.mj_to_pin_q(qpos), v_pin)
        worst_com = max(worst_com, float(np.linalg.norm(com_mj - com_pin)))
        worst_vcom = max(worst_vcom, float(np.linalg.norm(vcom_mj - vcom_pin)))

    check("q roundtrip mj->pin->mj", worst_q < 1e-12, f"max err {worst_q:.2e}")
    check("v roundtrip mj->pin->mj", worst_v < 1e-10, f"max err {worst_v:.2e}")
    check("CoM agrees within 1 cm", worst_com < 0.01, f"max err {worst_com:.2e} m")
    check("CoM velocity agrees within 5 cm/s", worst_vcom < 0.05, f"max err {worst_vcom:.2e} m/s")


def test_velocity_frame_is_not_ignored(rt, pin_wrap, conv) -> None:
    """A rotated base must make the world->body linear conversion matter.

    Guards convention (c): if someone 'simplifies' mj_to_pin_v by dropping the
    R^T term, this test fails while positions still look fine.
    """
    print("test_velocity_frame_is_not_ignored")
    qpos = rt.default_qpos()
    qpos[3:7] = np.array([np.cos(0.6), 0.0, 0.0, np.sin(0.6)])  # 68 deg yaw
    qvel = np.zeros(rt.model.nv)
    qvel[0:3] = [1.0, 0.0, 0.0]  # 1 m/s along world +x
    v_pin = conv.mj_to_pin_v(qpos, qvel)
    check("rotated base changes the linear velocity expression",
          not np.allclose(v_pin[0:3], qvel[0:3], atol=1e-6),
          f"pin lin {v_pin[0:3].round(4)} vs mj lin {qvel[0:3]}")
    check("linear speed magnitude is preserved by the rotation",
          abs(np.linalg.norm(v_pin[0:3]) - 1.0) < 1e-9)


def test_foot_geometry(rt) -> None:
    """The wrapper's foot offsets must match the MJCF's real contact spheres."""
    print("test_foot_geometry")
    mj_left = np.array(sorted(rt.left_foot_offsets.tolist()))
    mj_right = np.array(sorted(rt.right_foot_offsets.tolist()))
    const = np.array(sorted(FOOT_CONTACT_OFFSETS.tolist()))
    check("left foot has 4 contact spheres", len(mj_left) == 4, f"{len(mj_left)}")
    check("right foot has 4 contact spheres", len(mj_right) == 4, f"{len(mj_right)}")
    check("FOOT_CONTACT_OFFSETS matches MJCF",
          mj_left.shape == const.shape and np.abs(mj_left - const).max() < 1e-9)

    span_x = float(mj_left[:, 0].max() - mj_left[:, 0].min())
    span_y = float(mj_left[:, 1].max() - mj_left[:, 1].min())
    check("foot is a real foot, not a 5 cm placeholder square",
          span_x > 0.10 and span_y > 0.05, f"{span_x * 100:.1f} x {span_y * 100:.1f} cm")


def test_support_polygon_uses_foot_geometry(pin_wrap) -> None:
    """Double-support polygon area must reflect the real 17x6 cm feet."""
    print("test_support_polygon_uses_foot_geometry")
    import pinocchio as pin

    q = pin.neutral(pin_wrap.model)
    dq = np.zeros(pin_wrap.model.nv)
    pts = np.vstack([pin_wrap.foot_contact_points(q, dq, "left"),
                     pin_wrap.foot_contact_points(q, dq, "right")])
    q[2] -= float(pts[:, 2].min())

    feats = pin_wrap.get_support_features(q, dq, np.array([True, True]))
    area = feats["support_area"]
    # Two 17x6 cm feet 23.7 cm apart: the hull is ~0.17 m x ~0.30 m.
    check("double-support area is physically plausible", 0.03 < area < 0.08,
          f"{area:.4f} m^2")
    # The old placeholder produced a 0.05 x 0.237 m sliver ~= 0.012 m^2.
    check("area is not the old placeholder sliver", area > 0.02, f"{area:.4f} m^2")

    single = pin_wrap.get_support_features(q, dq, np.array([True, False]))
    check("single support is smaller than double", single["support_area"] < area,
          f"{single['support_area']:.4f} < {area:.4f} m^2")
    check("single support uses 4 sphere hull, not a square",
          abs(single["support_area"] - 0.0025) > 1e-4,
          f"{single['support_area']:.4f} m^2")
    check("single-support mode is labelled", single["support_mode"] == "single_left")


def main() -> int:
    rt = G1MujocoRuntime()
    pin_wrap = PinocchioWrapper(URDF)
    conv = StateConverter(rt.model, pin_wrap.model)

    test_quaternion_roundtrip()
    test_rotation_matrix_matches_mujoco()
    test_state_roundtrip_and_com(rt, pin_wrap, conv)
    test_velocity_frame_is_not_ignored(rt, pin_wrap, conv)
    test_foot_geometry(rt)
    test_support_polygon_uses_foot_geometry(pin_wrap)

    rt.close()
    print(f"\n{'ALL TESTS PASSED' if not _FAILURES else 'FAILURES: ' + ', '.join(_FAILURES)}")
    return 0 if not _FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
