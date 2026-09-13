"""Multi-motion Stage A tracking environment: KalariTrackEnv extended to
sample one of N reference motions per episode instead of a fixed clip."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.kalari_track_env import KalariTrackEnv, load_reference
from src.sim.conventions import quat_wxyz_to_matrix, quat_xyzw_to_wxyz


class MultiMotionTrackEnv(KalariTrackEnv):
    def __init__(self, npz_paths: list[str], seed: int | None = None,
                 max_start_frac: float = 0.7, render_mode: str | None = None):
        if not npz_paths:
            raise ValueError("MultiMotionTrackEnv needs at least one npz path")
        super().__init__(npz_paths[0], seed=seed, max_start_frac=max_start_frac,
                          render_mode=render_mode)
        self.max_start_frac_ = max_start_frac
        self.npz_paths = list(npz_paths)
        self._refs = [self._prepare_ref(p) for p in self.npz_paths]
        self._set_active_ref(0)

    def _prepare_ref(self, npz_path: str) -> dict:
        ref = load_reference(npz_path)
        idx = {n: i for i, n in enumerate(ref["joint_names"])}
        col_for_act = np.array([idx.get(j, -1) for j in self.rt.act_joint_names])
        if (col_for_act < 0).any():
            missing = [j for j, c in zip(self.rt.act_joint_names, col_for_act) if c < 0]
            raise ValueError(f"{npz_path}: reference missing joints {missing}")
        n_frames = ref["joint_pos"].shape[0]
        ref_z = ref["root_pos"][:, 2]
        ref_up = np.array([
            quat_wxyz_to_matrix(quat_xyzw_to_wxyz(q))[2, 2]
            for q in ref["root_quat_xyzw"]])
        return {"ref": ref, "col_for_act": col_for_act, "n_frames": n_frames,
                "ref_z": ref_z, "ref_up": ref_up,
                "max_start": int(n_frames * self.max_start_frac_)}

    def _set_active_ref(self, i: int) -> None:
        r = self._refs[i]
        self.ref = r["ref"]
        self.col_for_act = r["col_for_act"]
        self.n_frames = r["n_frames"]
        self.ref_z = r["ref_z"]
        self.ref_up = r["ref_up"]
        self.max_start = r["max_start"]
        self._active_motion_idx = i

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        i = int(self._rng.integers(0, len(self._refs)))
        self._set_active_ref(i)
        self._frame = int(self._rng.integers(0, self.max_start + 1))
        self._set_state_to_reference(self._frame)
        self._prev_action = np.zeros(self.nu)
        return self._obs(), {"motion_id": self.ref["motion_id"], "start_frame": self._frame}
