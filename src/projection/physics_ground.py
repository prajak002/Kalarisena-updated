"""Physics-grounded embodiment projection: Q0 -> Q*.

Measures contact slip, capture-point/support-polygon margin, and joint
torque vs. limits via PinocchioWrapper's RNEA and CoM Jacobian. Corrects
foot slip (per-run foot-locking) and capture-point margin (CoM-Jacobian
nudge on hip/waist roll); torque-limit feasibility is measured, not
actively corrected.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.sim.conventions import quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"
CP_MARGIN_EPS = 0.02
BALANCE_PULL_FRAC = 0.5
BALANCE_MAX_DELTA = 0.08
CONTACT_Z = 0.04


def _odd_window(n: int, target: int) -> int:
    w = min(target, n - 1 if n % 2 == 0 else n - 2)
    if w % 2 == 0:
        w -= 1
    return max(w, 5)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


class MotionProjector:
    def __init__(self):
        self.pin = PinocchioWrapper(URDF)
        self.rt = G1MujocoRuntime()
        self.balance_joints = [
            "waist_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint",
        ]
        self.balance_idx = [self.rt.act_joint_names.index(j) for j in self.balance_joints]

    # ------------------------------------------------------------------ io --
    def load(self, npz_path: str) -> dict:
        data = dict(np.load(npz_path, allow_pickle=True))
        cols = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
                for c in np.asarray(data["joint_cols"]).reshape(-1)]
        stripped = [c.removesuffix("_dof") for c in cols]
        idx = {n: i for i, n in enumerate(stripped)}
        col_for_act = np.array([idx.get(j, -1) for j in self.rt.act_joint_names])
        if (col_for_act < 0).any():
            missing = [j for j, c in zip(self.rt.act_joint_names, col_for_act) if c < 0]
            raise ValueError(f"{npz_path}: reference missing joints {missing}")
        joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float64)
        return {
            "npz": data,
            "joint_pos": joint_pos_raw[:, col_for_act],   # reordered to rt.act_joint_names
            "root_pos": np.asarray(data["root_pos"], dtype=np.float64).copy(),
            "root_quat_xyzw": np.asarray(data["root_quat_xyzw"], dtype=np.float64),
            "contacts": np.asarray(data.get("contacts", np.zeros((joint_pos_raw.shape[0], 2), bool)), dtype=bool),
            "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
            "col_for_act": col_for_act,
        }

    # -------------------------------------------------------------- pin q --
    def _pin_qdq(self, motion: dict):
        n = motion["joint_pos"].shape[0]
        fps = motion["fps"]
        q = np.zeros((n, self.pin.model.nq))
        for k in range(n):
            q[k, 0:3] = motion["root_pos"][k]
            q[k, 3:7] = motion["root_quat_xyzw"][k]        # pin freeflyer: xyzw
            q[k, 7:] = motion["joint_pos"][k]
        dt = 1.0 / fps
        dq_full = np.zeros((n, self.pin.model.nv))
        dq_full[:, 6:] = np.gradient(q[:, 7:], dt, axis=0)
        dq_full[:, 0:3] = np.gradient(motion["root_pos"], dt, axis=0)
        dq_full[:, 3:6] = self._local_angvel(motion["root_quat_xyzw"], dt)
        ddq_full = np.gradient(dq_full, dt, axis=0)
        return q, dq_full, ddq_full

    @staticmethod
    def _local_angvel(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
        n = quat_xyzw.shape[0]
        rot = Rotation.from_quat(quat_xyzw)
        omega = np.zeros((n, 3))
        rel = rot[:-1].inv() * rot[1:]
        rv = rel.as_rotvec() / dt
        omega[:-1] = rv
        omega[-1] = rv[-1] if n > 1 else 0.0
        return omega

    # ------------------------------------------------------------- measure --
    def measure(self, motion: dict) -> dict:
        n = motion["joint_pos"].shape[0]
        q, dq, ddq = self._pin_qdq(motion)
        contacts = motion["contacts"]

        slip_speeds = []
        cp_margins = []
        com_margins = []
        tau_ratios = []
        n_flight = 0
        for k in range(n):
            has_contact = bool(contacts[k].any())
            feats = self.pin.get_support_features(q[k], dq[k], contacts[k])
            if has_contact:
                cp_margins.append(feats["cp_margin"])
                com_margins.append(feats["com_margin"])
            else:
                n_flight += 1

            for side, is_contact in zip(("left", "right"), contacts[k]):
                if not is_contact:
                    continue
                J = self.pin.foot_jacobian(q[k], side)
                v = J @ dq[k]
                slip_speeds.append(float(np.linalg.norm(v[:2])))

            tau = self.pin.inverse_dynamics(q[k], dq[k], ddq[k])[6:]
            tau_ratios.append(np.abs(tau) / np.maximum(self.rt.torque_limit, 1e-6))

        tau_ratios = np.array(tau_ratios)
        cp_margins = np.array(cp_margins)
        return {
            "n_frames": int(n),
            "flight_frac": float(n_flight / n),
            "mean_contact_slip_mps": float(np.mean(slip_speeds)) if slip_speeds else 0.0,
            "max_contact_slip_mps": float(np.max(slip_speeds)) if slip_speeds else 0.0,
            "cp_margin_min": float(np.min(cp_margins)) if cp_margins.size else float("nan"),
            "cp_margin_mean": float(np.mean(cp_margins)) if cp_margins.size else float("nan"),
            "cp_infeasible_frac": float(np.mean(cp_margins < 0.0)) if cp_margins.size else float("nan"),
            "com_margin_mean": float(np.mean(com_margins)) if com_margins else float("nan"),
            "torque_violation_frac": float(np.mean(tau_ratios > 1.0)),
            "torque_ratio_p95": float(np.percentile(tau_ratios, 95)) if tau_ratios.size else 0.0,
        }

    def _lock_contacts(self, motion: dict) -> np.ndarray:
        n = motion["joint_pos"].shape[0]
        q, dq, _ = self._pin_qdq(motion)
        contacts = motion["contacts"]

        shift = np.zeros((n, 2))
        weight = np.zeros(n)
        for side_i, side in enumerate(("left", "right")):
            mask = contacts[:, side_i]
            for (i0, i1) in _contiguous_runs(mask):
                foot_xy = np.array([
                    self.pin.get_foot_positions(q[k], dq[k])[side_i][:2] for k in range(i0, i1 + 1)
                ])
                run_n = foot_xy.shape[0]
                if run_n >= 9:
                    anchor_path = savgol_filter(foot_xy, _odd_window(run_n, 21), 2, axis=0)
                else:
                    anchor_path = np.broadcast_to(np.median(foot_xy, axis=0), foot_xy.shape)
                for offset, k in enumerate(range(i0, i1 + 1)):
                    shift[k] += anchor_path[offset] - foot_xy[offset]
                    weight[k] += 1.0

        nz = weight > 0
        shift[nz] /= weight[nz][:, None]
        if n >= 9:
            win = _odd_window(n, 15)
            shift = savgol_filter(shift, win, 2, axis=0)
        return shift

    def _recover_balance(self, motion: dict) -> np.ndarray:
        n = motion["joint_pos"].shape[0]
        q, dq, _ = self._pin_qdq(motion)
        contacts = motion["contacts"]
        delta = np.zeros((n, len(self.balance_idx)))

        for k in range(n):
            feats = self.pin.get_support_features(q[k], dq[k], contacts[k])
            if feats["cp_margin"] >= CP_MARGIN_EPS or feats["support_area"] <= 0:
                continue
            com_xy = self.pin.compute_com(q[k], dq[k])[0][:2]
            target_xy = feats["support_center"]
            deficit = (target_xy - com_xy) * BALANCE_PULL_FRAC

            J_com = self.pin.compute_com_jacobian(q[k])[:2]          # [2, nv]
            v_idx = [6 + i for i in self.balance_idx]                # actuated -> nv index
            J_sub = J_com[:, v_idx]                                  # [2, 3]
            d, *_ = np.linalg.lstsq(J_sub, deficit, rcond=None)
            delta[k] = np.clip(d, -BALANCE_MAX_DELTA, BALANCE_MAX_DELTA)

        if n >= 9:
            win = _odd_window(n, 15)
            delta = savgol_filter(delta, win, 2, axis=0)
        return delta

    # ------------------------------------------------------------- driver --
    def project_one(self, npz_path: str) -> dict:
        motion = self.load(npz_path)
        before = self.measure(motion)

        shift_xy = self._lock_contacts(motion)
        motion["root_pos"][:, 0:2] += shift_xy

        delta_balance = self._recover_balance(motion)
        for col, j in enumerate(self.balance_idx):
            lo, hi = self.rt.model.jnt_range[
                int(self.rt.model.actuator_trnid[j, 0])]
            motion["joint_pos"][:, j] = np.clip(
                motion["joint_pos"][:, j] + delta_balance[:, col], lo, hi)

        shift_xy2 = self._lock_contacts(motion)
        motion["root_pos"][:, 0:2] += shift_xy2

        after = self.measure(motion)

        data = motion["npz"]
        fps = motion["fps"]
        joint_pos_full = np.asarray(data["joint_pos"], dtype=np.float64).copy()
        joint_pos_full[:, motion["col_for_act"]] = motion["joint_pos"]

        q = np.asarray(data["q"], dtype=np.float64).copy()
        q[:, 0:3] = motion["root_pos"]
        q[:, 3:7] = motion["root_quat_xyzw"]
        q[:, 7:] = joint_pos_full
        dq = np.gradient(q, 1.0 / fps, axis=0)

        data.update({"joint_pos": joint_pos_full, "root_pos": motion["root_pos"],
                     "q": q, "dq": dq})
        np.savez(npz_path, **data)

        return {"motion_id": os.path.splitext(os.path.basename(npz_path))[0],
                "before": before, "after": after}

    def close(self):
        self.rt.close()
