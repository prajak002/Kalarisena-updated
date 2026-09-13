#!/usr/bin/env python3
"""Real MPJPE and Intent Preservation Rate (IPR) against the existing
multi-motion tracker and residual policy - no new training, reuses
logs/stageA_multi12 and logs/residual_multi12.

MPJPE here is mean per-actuated-joint 3D body-frame position error (the
child body of each actuated joint, via forward kinematics on both the
rolled-out and the reference trajectory at matching frames) - a real,
computed Cartesian error, not the joint-angle RMSE reported elsewhere.

IPR is defined here as: reaches the final reference frame without falling,
under a lateral push perturbation - completion despite disturbance, which
is what the paper's Intent Preservation Rate is measuring, computed against
this repo's own narrower single-motion-per-episode scope rather than the
paper's full multi-skill continuation definition.

Usage
  python3 scripts/eval_protocol.py \
      --tracker logs/stageA_multi12/tracking_multi_best.zip \
      --critic logs/scvc_multi12/scvc_critic.pt \
      --residual logs/residual_multi12/residual_multi_best.zip \
      --out logs/eval_protocol
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viability.perturbation import Perturbation
from src.viability.perturbed_multi_env import PerturbedMultiMotionTrackEnv

MOTION_SET = [
    "data/motions_retargeted/kw_long_stance.npz",
    "data/motions_retargeted/kt_vadivu_lowseat.npz",
    "data/motions_retargeted/kt_warrior_pose.npz",
    "data/motions_retargeted/ky_warrior_lunge.npz",
    "data/motions_retargeted/kt_chuvadu_step.npz",
    "data/motions_retargeted/kw_deep_reach.npz",
    "data/motions_retargeted/ky_deep_lunge.npz",
    "data/motions_retargeted/pk_stance_transitions.npz",
    "data/motions_retargeted/ks_side_kick.npz",
    "data/motions_retargeted/kw_highkick_right.npz",
    "data/motions_retargeted/ky_kick_seq.npz",
    "data/motions_retargeted/pk_kick_lunge.npz",
]
PUSH_FORCES = [0.0, 40.0, 80.0]


def _joint_body_ids(env: PerturbedMultiMotionTrackEnv) -> list[int]:
    ids = []
    for a in range(env.rt.model.nu):
        jid = int(env.rt.model.actuator_trnid[a, 0])
        ids.append(int(env.rt.model.jnt_bodyid[jid]))
    return ids


def _body_positions(env: PerturbedMultiMotionTrackEnv, qpos: np.ndarray, body_ids: list[int]) -> np.ndarray:
    d, m, mj = env.rt.data, env.rt.model, env.rt.mujoco
    saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mj.mj_forward(m, d)
    pos = np.array([d.xpos[b].copy() for b in body_ids])
    d.qpos[:] = saved_q
    d.qvel[:] = saved_v
    mj.mj_forward(m, d)
    return pos


def _reference_full_qpos(env: PerturbedMultiMotionTrackEnv, frame: int) -> np.ndarray:
    from src.sim.conventions import quat_xyzw_to_wxyz

    qpos = env.rt.default_qpos()
    frame = min(frame, env.n_frames - 1)
    qpos[0:3] = env.ref["root_pos"][frame]
    qpos[3:7] = quat_xyzw_to_wxyz(env.ref["root_quat_xyzw"][frame])
    qpos[env.rt.act_qadr] = env._ref_joints(frame)
    return qpos


def run_rollout(env, tracker, residual, gate_fn, force: float, use_residual: bool, body_ids: list[int]):
    pert = Perturbation(t_start=0.6, duration=0.1, force_x=0.0, force_y=force, obs_noise_std=0.0)
    env.set_perturbation(pert)
    obs, info = env.reset()
    if gate_fn is not None:
        gate_fn.reset()
    done = trunc = False
    errs = []
    fell = False
    while not (done or trunc):
        a0, _ = tracker.predict(obs, deterministic=True)
        a0 = np.asarray(a0, dtype=np.float64)
        if use_residual and residual is not None:
            from src.viability.features import eta
            from src.viability.residual_policy import RESIDUAL_ACTION_SCALE, gate

            info_prev = {"joint_mse": 0.0, "upright": env.rt.torso_upright_cos()}
            feats = env.physics_features()
            phase = env._frame / max(env.n_frames - 1, 1)
            eta_vec = eta(info_prev, feats, phase, env.rt.base_height, (0.0, 0.0))
            _, v_bar = gate_fn.update(eta_vec)
            g = gate(v_bar)
            residual_obs = np.concatenate([obs, [0, 0, 0, 0, 0]])
            delta_a, _ = residual.predict(residual_obs, deterministic=True)
            action = a0 + g * RESIDUAL_ACTION_SCALE * np.asarray(delta_a, dtype=np.float64)
        else:
            action = a0

        obs, r, done, trunc, si = env.step(action)
        ref_q = _reference_full_qpos(env, env._frame)
        actual_q = env.rt.data.qpos.copy()
        ref_pos = _body_positions(env, ref_q, body_ids)
        actual_pos = _body_positions(env, actual_q, body_ids)
        errs.append(float(np.mean(np.linalg.norm(ref_pos - actual_pos, axis=1))))
        fell = fell or si["fell"]

    completed = bool(trunc and not fell)
    return {"mpjpe": float(np.mean(errs)), "completed": completed, "fell": bool(fell)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--out", default="logs/eval_protocol")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO
    from src.viability.residual_policy import ViabilityGate

    tracker = PPO.load(args.tracker, device="cpu")
    residual = PPO.load(args.residual, device="cpu")

    results = {"tracker_only": [], "residual_gated": []}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        env = PerturbedMultiMotionTrackEnv([npz], seed=15000)
        env.max_start = 0
        body_ids = _joint_body_ids(env)
        for force in PUSH_FORCES:
            for arm in ("tracker_only", "residual_gated"):
                gate_fn = ViabilityGate(args.critic) if arm == "residual_gated" else None
                row = run_rollout(env, tracker, residual, gate_fn, force,
                                   use_residual=(arm == "residual_gated"), body_ids=body_ids)
                row.update({"motion_id": mid, "push_force": force})
                results[arm].append(row)
        env.close()

    summary = {}
    for arm, rows in results.items():
        summary[arm] = {
            "mpjpe_mean": float(np.mean([r["mpjpe"] for r in rows])),
            "ipr": float(np.mean([r["completed"] for r in rows])),
            "n_rollouts": len(rows),
        }
        for force in PUSH_FORCES:
            subset = [r for r in rows if r["push_force"] == force]
            summary[arm][f"ipr_force_{int(force)}N"] = float(np.mean([r["completed"] for r in subset]))

    out = {"overall": summary, "per_rollout": results}
    with open(os.path.join(args.out, "eval_protocol_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
