"""Stage G: GA joint-pool evolution (docs/PHYSICAL_AI_STAGE_G_H_I_DRAFT.md sec 2).

Evolves a physically-plausible bridging trajectory between the ENDING pose of
one real Kalaripayattu clip and the STARTING pose of another, filling a
transition the motion corpus never captured. Pure kinematics + a Pinocchio
forward pass per fitness evaluation - no MuJoCo contact simulation, no PPO.
This is what makes the genetic algorithm the appropriate tool for this
sub-problem (a static trajectory-optimization search), reserving RL for the
actual sequential control problem (Stages A/B/C).

Chromosome: (K, nu) keyframe joint angles in G1MujocoRuntime actuator order,
Catmull-Rom-style cubic-spline interpolated to full frame rate. Root pose is
linearly blended between the two endpoints and held fixed (not evolved) - the
paper's Stage G is scoped to joint-angle search, not root trajectory search.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sim.conventions import quat_wxyz_to_matrix, quat_xyzw_to_wxyz
from src.sim.mujoco_runtime import G1MujocoRuntime
from src.dynamics.pinocchio_wrapper import PinocchioWrapper

CONTACT_Z_THRESHOLD = 0.05


@dataclass
class GAConfig:
    n_keyframes: int = 6
    n_frames: int = 60
    fps: float = 30.0
    population: int = 150
    generations: int = 80
    tournament_k: int = 3
    elitism: int = 6
    mutation_prob: float = 0.3
    mutation_sigma_frac: float = 0.06  # fraction of each joint's ROM
    w_stability: float = 1.0
    w_smoothness: float = 0.02
    w_coherence: float = 0.5
    seed: int = 0


def _load_clip_joints(npz_path: str, act_joint_names: list[str]) -> np.ndarray:
    """Return (T, nu) joint angles for one clip, reindexed to actuator order."""
    d = np.load(npz_path, allow_pickle=True)
    cols = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
            for c in np.asarray(d["joint_cols"]).reshape(-1)]
    stripped = [c.removesuffix("_dof") for c in cols]
    idx = {n: i for i, n in enumerate(stripped)}
    col_for_act = np.array([idx[j] for j in act_joint_names])
    joint_pos = np.asarray(d["joint_pos"], dtype=np.float64)
    return joint_pos[:, col_for_act]


def _joint_limits(rt: G1MujocoRuntime) -> tuple[np.ndarray, np.ndarray]:
    nu = rt.model.nu
    lo = np.zeros(nu)
    hi = np.zeros(nu)
    for a in range(nu):
        jid = int(rt.model.actuator_trnid[a, 0])
        lo[a], hi[a] = rt.model.jnt_range[jid]
    return lo, hi


def _foot_heights(rt: G1MujocoRuntime, root_pos: np.ndarray, root_quat_xyzw: np.ndarray,
                   joint_pos: np.ndarray) -> np.ndarray:
    """Forward-kinematics-only lowest-foot-sphere height per frame (no physics step)."""
    n = joint_pos.shape[0]
    default_q = rt.default_qpos()
    lowest = np.zeros(n)
    for k in range(n):
        qpos = default_q.copy()
        qpos[0:3] = root_pos[k]
        qpos[3:7] = quat_xyzw_to_wxyz(root_quat_xyzw[k])
        qpos[rt.act_qadr] = joint_pos[k]
        rt.data.qpos[:] = qpos
        rt.mujoco.mj_forward(rt.model, rt.data)
        heights = {g: float(rt.data.geom_xpos[g][2]) - float(rt.model.geom_size[g][0])
                   for g in rt.left_foot_geoms + rt.right_foot_geoms}
        left_h = min(heights[g] for g in rt.left_foot_geoms)
        right_h = min(heights[g] for g in rt.right_foot_geoms)
        lowest[k] = left_h if k % 1 == 0 else right_h  # placeholder, overwritten below
        lowest[k] = min(left_h, right_h)
    return lowest


class TransitionGA:
    def __init__(self, npz_a: str, npz_b: str, cfg: GAConfig | None = None):
        self.cfg = cfg or GAConfig()
        self.rt = G1MujocoRuntime()
        self.pin = PinocchioWrapper("assets/unitree_g1/g1_29dof_rev_1_0.urdf")
        self.act_names = list(self.rt.act_joint_names)
        self.nu = len(self.act_names)
        self.jnt_lo, self.jnt_hi = _joint_limits(self.rt)
        self.rom = np.maximum(self.jnt_hi - self.jnt_lo, 1e-3)

        joints_a = _load_clip_joints(npz_a, self.act_names)
        joints_b = _load_clip_joints(npz_b, self.act_names)
        self.pose_a = joints_a[-1]  # end of clip A
        self.pose_b = joints_b[0]   # start of clip B

        da = np.load(npz_a, allow_pickle=True)
        db = np.load(npz_b, allow_pickle=True)
        self.root_pos_a = np.asarray(da["root_pos"])[-1]
        self.root_pos_b = np.asarray(db["root_pos"])[0]
        self.root_quat_a = np.asarray(da["root_quat_xyzw"])[-1]
        self.root_quat_b = np.asarray(db["root_quat_xyzw"])[0]

        t = np.linspace(0, 1, self.cfg.n_frames)
        self.root_pos_path = (1 - t[:, None]) * self.root_pos_a + t[:, None] * self.root_pos_b
        # simple normalized-lerp quaternion blend (sufficient for a short bridge)
        q = (1 - t[:, None]) * self.root_quat_a + t[:, None] * self.root_quat_b
        self.root_quat_path = q / np.linalg.norm(q, axis=1, keepdims=True)

        self.linear_baseline = (1 - t[:, None]) * self.pose_a + t[:, None] * self.pose_b

        self._rng = np.random.default_rng(self.cfg.seed)
        self._keyframe_t = np.linspace(0, 1, self.cfg.n_keyframes)
        self._frame_t = t

    # ---------------------------------------------------------------- decode
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """(K, nu) keyframes -> (n_frames, nu) via natural cubic spline per joint."""
        spline = CubicSpline(self._keyframe_t, chromosome, axis=0, bc_type="natural")
        q = spline(self._frame_t)
        return np.clip(q, self.jnt_lo, self.jnt_hi)

    # --------------------------------------------------------------- fitness
    def fitness(self, chromosome: np.ndarray) -> float:
        if np.any(chromosome < self.jnt_lo - 1e-6) or np.any(chromosome > self.jnt_hi + 1e-6):
            return -1e9

        q = self.decode(chromosome)
        dt = 1.0 / self.cfg.fps

        lowest = _foot_heights(self.rt, self.root_pos_path, self.root_quat_path, q)
        contacts_dbl = lowest < CONTACT_Z_THRESHOLD

        stability = 0.0
        for k in range(q.shape[0]):
            full_q = np.concatenate([self.root_pos_path[k], self.root_quat_path[k], q[k]])
            if k == 0:
                full_dq = np.zeros(self.pin.model.nv)
            else:
                prev_q = np.concatenate([self.root_pos_path[k - 1], self.root_quat_path[k - 1], q[k - 1]])
                # Approximate: linear root velocity from position diff, angular
                # root velocity left at zero (bridge motion is short and slow),
                # joint velocity from finite difference. Good enough for a
                # fitness signal; not used for control.
                raw_diff = (full_q - prev_q) / dt
                full_dq = np.concatenate([raw_diff[:3], np.zeros(3), raw_diff[7:]])
            contacts = np.array([contacts_dbl[k], contacts_dbl[k]])
            feats = self.pin.get_support_features(full_q, full_dq, contacts)
            stability += min(feats["com_margin"], 0.2) + min(feats["cp_margin"], 0.2)
        stability /= q.shape[0]

        d2q = np.diff(q, n=2, axis=0)
        smoothness = -float(np.mean(np.sum(d2q ** 2, axis=1)))

        coherence = -float(np.mean(np.sum((q - self.linear_baseline) ** 2, axis=1)))

        return (
            self.cfg.w_stability * stability
            + self.cfg.w_smoothness * smoothness
            + self.cfg.w_coherence * coherence
        )

    # ------------------------------------------------------------ GA loop
    def _seed_individual(self) -> np.ndarray:
        base = (1 - self._keyframe_t[:, None]) * self.pose_a + self._keyframe_t[:, None] * self.pose_b
        jitter = self._rng.normal(0, self.cfg.mutation_sigma_frac, size=base.shape) * self.rom[None, :]
        return np.clip(base + jitter, self.jnt_lo, self.jnt_hi)

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self._rng.random(ind.shape) < self.cfg.mutation_prob
        sigma = self.cfg.mutation_sigma_frac * self.rom[None, :]
        noise = self._rng.normal(0, 1, size=ind.shape) * sigma * mask
        return np.clip(ind + noise, self.jnt_lo, self.jnt_hi)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        alpha = self._rng.uniform(0.0, 1.0, size=a.shape)
        return alpha * a + (1 - alpha) * b

    def _tournament(self, pop: list[np.ndarray], fit: np.ndarray) -> np.ndarray:
        idx = self._rng.choice(len(pop), size=self.cfg.tournament_k, replace=False)
        best = idx[np.argmax(fit[idx])]
        return pop[best]

    def run(self, log=print) -> tuple[np.ndarray, list[dict]]:
        pop = [self._seed_individual() for _ in range(self.cfg.population)]
        history: list[dict] = []
        best_ind = pop[0]
        best_fit = -np.inf

        for gen in range(self.cfg.generations):
            fit = np.array([self.fitness(ind) for ind in pop])
            order = np.argsort(-fit)
            pop = [pop[i] for i in order]
            fit = fit[order]

            if fit[0] > best_fit:
                best_fit = float(fit[0])
                best_ind = pop[0].copy()

            history.append({"generation": gen, "best_fitness": float(fit[0]),
                             "mean_fitness": float(np.mean(fit[fit > -1e8]))
                             if np.any(fit > -1e8) else float("nan")})
            if gen % 10 == 0 or gen == self.cfg.generations - 1:
                log(f"  gen {gen:3d}  best={fit[0]: .4f}  mean={history[-1]['mean_fitness']: .4f}")

            next_pop = pop[: self.cfg.elitism]
            while len(next_pop) < self.cfg.population:
                p1 = self._tournament(pop, fit)
                p2 = self._tournament(pop, fit)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)
            pop = next_pop

        return best_ind, history

    def close(self):
        self.rt.close()
