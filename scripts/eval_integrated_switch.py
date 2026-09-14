#!/usr/bin/env python3
"""Stage F: integrates trained policies with the existing ModeSwitch
(src/switch/mode_switch.py) - real threshold-based switching between a
nominal tracker, a fall policy, and (when --recovery is given) the real
Stage E recovery policy, per train_switch.py's own docstring ("Start with
threshold-based switching... not learned switch").

RecoveryEnv (src/envs/recovery_env.py) has its own observation space (no
reference motion, so it can't consume the tracker's 124-d obs) - when
--recovery is given, its exact observation is reconstructed here from the
shared MuJoCo state instead of falling back to the nominal tracker's
action for RECOVERY mode.

Usage
  python3 scripts/eval_integrated_switch.py \
      --nominal logs/stageA_multi12/tracking_multi_best.zip \
      --fall logs/stageD_fall/fall_best.zip \
      --recovery logs/stageE_recovery_v2/recovery_best.zip \
      --out logs/stageF_switch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper
from src.envs.kalari_track_env import FALL_DUP, FALL_DZ
from src.envs.multi_motion_env import MultiMotionTrackEnv
from src.sim.conventions import quat_wxyz_to_matrix
from src.switch.mode_switch import Mode, ModeSwitch, SwitchConfig

RECOVERY_ACTION_SCALE = 0.5  # matches src/envs/recovery_env.py's own ACTION_SCALE

URDF = "assets/unitree_g1/g1_29dof_rev_1_0.urdf"

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


def _features(env: MultiMotionTrackEnv, pin: PinocchioWrapper, step_index: int) -> dict:
    d = env.rt.data
    q = np.concatenate([d.qpos[0:3], d.qpos[3:7][[1, 2, 3, 0]], d.qpos[env.rt.act_qadr]])
    dq = np.concatenate([d.qvel[0:3], d.qvel[3:6], d.qvel[env.rt.act_vadr]])
    left_h_ok, right_h_ok = True, True
    contacts = np.array([left_h_ok, right_h_ok])
    feats = pin.get_support_features(q, dq, contacts)
    hg = pin.compute_centroidal_momentum(q, dq)
    upright = env.rt.torso_upright_cos()
    return {
        "cp_margin": feats["cp_margin"] if feats["support_area"] > 0 else 0.0,
        "momentum_norm": float(np.linalg.norm(hg[3:])),
        "base_height": float(env.rt.base_height),
        "is_fallen": bool(upright < 0.3),
        "is_upright": bool(upright > 0.7),
        "step_index": step_index,
    }


def _torso_force(env: MultiMotionTrackEnv) -> float:
    torso_body = env.rt.mujoco.mj_name2id(env.rt.model, env.rt.mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_geoms = env.rt._collision_geoms_of_body(torso_body)
    return env.rt.geom_group_force(torso_geoms)


def _recovery_obs(env: MultiMotionTrackEnv) -> np.ndarray:
    """Exact reconstruction of RecoveryEnv._obs() (src/envs/recovery_env.py)
    from the shared MuJoCo state, since RecoveryEnv itself has no reference
    motion and can't be instantiated here."""
    d = env.rt.data
    q = d.qpos[env.rt.act_qadr]
    dq = d.qvel[env.rt.act_vadr] * 0.1
    rot = quat_wxyz_to_matrix(d.qpos[3:7])
    gravity_b = rot.T @ np.array([0.0, 0.0, -1.0])
    angvel_b = d.qvel[3:6] * 0.2
    upright = np.array([env.rt.torso_upright_cos()])
    return np.concatenate([q, dq, gravity_b, angvel_b, upright])


def _recovery_step(env: MultiMotionTrackEnv, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict]:
    """Applies the recovery policy's action the way RecoveryEnv itself does -
    an offset from the fixed default standing pose at scale 0.5 - instead of
    routing it through KalariTrackEnv.step()'s residual-on-moving-reference
    formula (q_cmd = ref_joints(frame) + 0.25*action). The two envs' actions
    are not interchangeable: feeding a RecoveryEnv-trained action into
    env.step() silently reinterprets it as a small offset from the current
    Kalaripayattu reference pose, which produced a 100% fall rate that was a
    plumbing bug, not a real result. Also freezes env._frame for the
    duration of RECOVERY mode (rather than letting KalariTrackEnv.step()
    advance it) so the reference motion resumes from where it left off once
    NOMINAL mode takes back over, instead of having skipped ahead while the
    robot was busy recovering."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    q_cmd = np.clip(env.rt.default_joint_targets() + RECOVERY_ACTION_SCALE * action,
                     env.jnt_lo, env.jnt_hi)
    env.rt.control_step(q_cmd)

    k = min(env._frame, env.n_frames - 1)
    z = env.rt.base_height
    upright = env.rt.torso_upright_cos()
    fell = bool(z < env.ref_z[k] - FALL_DZ or upright < env.ref_up[k] - FALL_DUP)
    truncated = bool(env._frame >= env.n_frames - 1)
    return env._obs(), fell, truncated, {"fell": fell, "upright": upright}


def run_episode(env, pin, nominal, fall, recovery, switch: ModeSwitch, use_switch: bool):
    obs, info = env.reset()
    switch.reset()
    done = trunc = False
    step = 0
    mode_counts = {m.value: 0 for m in Mode}
    fell_ever = False
    while not (done or trunc):
        feats = _features(env, pin, step)
        mode = switch.step(feats) if use_switch else Mode.NOMINAL
        mode_counts[mode.value] += 1

        if mode == Mode.RECOVERY and recovery is not None:
            action, _ = recovery.predict(_recovery_obs(env), deterministic=True)
            obs, fell, trunc, si = _recovery_step(env, action)
            done = fell
        else:
            if mode == Mode.FALL and fall is not None:
                fall_obs = np.concatenate([obs, [_torso_force(env), 0.0]])
                action, _ = fall.predict(fall_obs, deterministic=True)
            else:
                action, _ = nominal.predict(obs, deterministic=True)
            obs, r, done, trunc, si = env.step(action)

        fell_ever = fell_ever or si["fell"]
        step += 1
    return {"episode_len": step, "fell": int(fell_ever), "mode_counts": mode_counts,
            "transitions": len(switch.transition_log)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--fall", default=None)
    ap.add_argument("--recovery", default=None)
    ap.add_argument("--out", default="logs/stageF_switch")
    ap.add_argument("--episodes-per-motion", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from stable_baselines3 import PPO

    nominal = PPO.load(args.nominal, device="cpu")
    fall = PPO.load(args.fall, device="cpu") if args.fall else None
    recovery = PPO.load(args.recovery, device="cpu") if args.recovery else None
    pin = PinocchioWrapper(URDF)
    switch_cfg = SwitchConfig(delta1=0.05, delta2=15.0, delta3=0.20, min_dwell_steps=30)

    results = {"switched": [], "nominal_only": []}
    per_motion = {}
    for npz in MOTION_SET:
        mid = os.path.basename(npz)[:-4]
        env = MultiMotionTrackEnv([npz], seed=12000)
        env.max_start = 0
        rows = {"switched": [], "nominal_only": []}
        for arm, use_switch in (("switched", True), ("nominal_only", False)):
            for ep in range(args.episodes_per_motion):
                switch = ModeSwitch(switch_cfg)
                row = run_episode(env, pin, nominal, fall, recovery, switch, use_switch)
                rows[arm].append(row)
                results[arm].append(row)
        env.close()
        per_motion[mid] = {
            arm: {"fall_rate": float(np.mean([r["fell"] for r in rows[arm]])),
                  "mean_episode_len": float(np.mean([r["episode_len"] for r in rows[arm]])),
                  "mean_transitions": float(np.mean([r["transitions"] for r in rows[arm]]))}
            for arm in ("switched", "nominal_only")
        }

    overall = {arm: {"fall_rate": float(np.mean([r["fell"] for r in results[arm]])),
                      "mean_episode_len": float(np.mean([r["episode_len"] for r in results[arm]])),
                      "mean_transitions": float(np.mean([r["transitions"] for r in results[arm]]))}
               for arm in ("switched", "nominal_only")}

    summary = {"meta": {"nominal": args.nominal, "fall": args.fall, "recovery": args.recovery,
                         "motions": MOTION_SET,
                         "note": ("RECOVERY mode routed through the real Stage E policy via "
                                  "_recovery_obs()/_recovery_step() when --recovery is given "
                                  "(action applied as an offset from the fixed default stance, "
                                  "scale 0.5, matching RecoveryEnv itself, with the reference "
                                  "frame frozen for the duration - NOT via KalariTrackEnv's "
                                  "residual-on-moving-reference step()); else falls back to "
                                  "the nominal tracker's action")},
               "overall": overall, "per_motion": per_motion}
    with open(os.path.join(args.out, "eval_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
