#!/usr/bin/env python3
"""Train the Successor-Conditioned Viability Critic. The "intended
continuation" is instantiated as "reach the final phase of this motion
safely", not a distinct next skill.

Usage
  python3 scripts/train_viability_critic.py \
      --policy logs/stageA_kw_long_stance/tracking_best.zip \
      --npz data/motions_retargeted/kw_long_stance.npz \
      --episodes 400 --out logs/scvc_kw_long_stance
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viability.critic import ViabilityCritic, viability_loss
from src.viability.features import eta, FEATURE_NAMES, feature_scaler
from src.viability.manifold import SuccessorManifold
from src.viability.metrics import auroc, auprc, brier_score, expected_calibration_error
from src.viability.perturbation import sample_perturbation, null_perturbation
from src.viability.perturbed_env import PerturbedTrackEnv

SUCCESS_HORIZON_FRAC = 0.15  # "successor entry" = final 15% of the motion
EPSILON_G = 0.75             # manifold-distance threshold for Omega(g) (feature units)


def _rollout(env: PerturbedTrackEnv, policy, pert, rng) -> list[dict]:
    env.set_perturbation(pert)
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        if policy is None:
            # No trained checkpoint given: zero residual action (pure reference
            # tracking via the PD layer) rather than uniform-random, which
            # falls instantly and leaves nothing to build a manifold from.
            # Still clearly a smoke test, not a trained-policy result.
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        feats = env.physics_features()
        com_vel = (0.0, 0.0)
        feat_vec = eta(info, feats, info["phase"], env.rt.base_height, com_vel)
        records.append({"feat": feat_vec, "phase": info["phase"], "fell": info["fell"]})
        done = terminated or truncated
    return records


def build_successor_manifold(env: PerturbedTrackEnv, policy, n_nominal: int, rng) -> tuple[SuccessorManifold, np.ndarray, str]:
    """Returns (manifold, scaler, mode) where mode honestly records which case applied:
    'full-success' = built from real non-fallen late-phase frames (the paper's
    intended definition); 'best-effort-relaxed' = the policy never completed
    the motion without falling in any nominal rollout, so the manifold is
    built from each rollout's furthest-reached frame regardless of outcome -
    a real, reportable finding about the policy, not a fabricated success."""
    late_feats = []
    best_effort: list[np.ndarray] = []
    for _ in range(n_nominal):
        recs = _rollout(env, policy, null_perturbation(), rng)
        for r in recs:
            if r["phase"] >= 1.0 - SUCCESS_HORIZON_FRAC and not r["fell"]:
                late_feats.append(r["feat"])
        if recs:
            best_effort.append(recs[-1]["feat"])

    mode = "full-success"
    if not late_feats and best_effort:
        print("  [note] no non-fallen late-phase frames across any nominal rollout: "
              "the policy did not complete this motion without falling. Falling back "
              "to each rollout's furthest-reached frame as the successor-entry set. "
              "This is a real finding about the policy, reported as such.")
        late_feats = best_effort
        mode = "best-effort-relaxed"

    if not late_feats:
        raise SystemExit(
            "No rollouts produced any frames at all - check the environment/policy."
        )
    feats = np.array(late_feats)
    mean, std = feature_scaler(feats)
    normed = (feats - mean) / std
    manifold = SuccessorManifold(k=20).fit(normed)
    return manifold, (mean, std), mode


def label_rollout(records: list[dict], manifold: SuccessorManifold, scaler) -> list[dict]:
    mean, std = scaler
    out = []
    n = len(records)
    horizon = max(1, int(SUCCESS_HORIZON_FRAC * n))
    for t, r in enumerate(records):
        window = records[t: min(n, t + horizon * 4)]
        reached = False
        for w in window:
            if w["fell"]:
                break
            d = manifold.distance((w["feat"] - mean) / std)
            if d <= EPSILON_G:
                reached = True
                break
        y_safe = not r["fell"]
        y_via = float(y_safe and reached)
        out.append({"feat": r["feat"], "y_via": y_via})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="SB3 PPO .zip checkpoint; omit for a random-policy smoke test")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--nominal-episodes", type=int, default=20)
    ap.add_argument("--out", default="logs/scvc")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.episodes, args.nominal_episodes, args.epochs = 6, 4, 5

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    policy = None
    if args.policy:
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy)
        print(f"loaded policy {args.policy}")
    else:
        print("[warn] no --policy given: running with a RANDOM policy. "
              "This is a pipeline smoke test, not a real SCVC result.")

    env = PerturbedTrackEnv(args.npz, seed=args.seed)

    print(f"building successor manifold from {args.nominal_episodes} nominal rollouts...")
    manifold, scaler, manifold_mode = build_successor_manifold(env, policy, args.nominal_episodes, rng)
    print(f"  successor-manifold mode: {manifold_mode}")
    print(f"  manifold has {manifold._points.shape[0]} successor-entry points")

    print(f"collecting {args.episodes} counterfactual rollouts...")
    dataset = []
    n_fell = 0
    for ep in range(args.episodes):
        pert = sample_perturbation(rng, episode_duration=env.n_frames / env.rt.CONTROL_HZ)
        recs = _rollout(env, policy, pert, rng)
        n_fell += int(any(r["fell"] for r in recs))
        dataset.extend(label_rollout(recs, manifold, scaler))
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"  {ep + 1}/{args.episodes} episodes, fell so far: {n_fell}")

    feats = np.array([d["feat"] for d in dataset])
    labels = np.array([d["y_via"] for d in dataset])
    mean, std = scaler
    feats_n = (feats - mean) / std
    print(f"dataset: {len(labels)} labeled frames, {labels.mean():.1%} positive (Y_via=1)")

    n = len(labels)
    idx = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.tensor(feats_n[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    x_val = torch.tensor(feats_n[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.float32)

    critic = ViabilityCritic(in_dim=feats.shape[1])
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x_train))
        epoch_loss = 0.0
        for i in range(0, len(perm), args.batch_size):
            batch = perm[i: i + args.batch_size]
            opt.zero_grad()
            v = critic(x_train[batch])
            loss = viability_loss(v, y_train[batch])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(x_train)

        with torch.no_grad():
            v_val = critic(x_val).numpy()
        y_val_np = y_val.numpy()
        history.append({
            "epoch": epoch, "train_loss": epoch_loss,
            "val_auroc": auroc(y_val_np, v_val),
            "val_brier": brier_score(y_val_np, v_val),
        })
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            h = history[-1]
            print(f"  epoch {epoch:3d}  loss={h['train_loss']:.4f}  "
                  f"val_auroc={h['val_auroc']:.3f}  val_brier={h['val_brier']:.4f}")

    with torch.no_grad():
        v_val = critic(x_val).numpy()
    y_val_np = y_val.numpy()
    final_metrics = {
        "n_frames": int(n),
        "n_episodes": int(args.episodes),
        "positive_rate": float(labels.mean()),
        "auroc": auroc(y_val_np, v_val),
        "auprc": auprc(y_val_np, v_val),
        "brier": brier_score(y_val_np, v_val),
        "ece": expected_calibration_error(y_val_np, v_val),
        "policy": args.policy or "RANDOM (smoke test, not a real result)",
        "successor_manifold_mode": manifold_mode,
        "feature_names": FEATURE_NAMES,
    }
    print(json.dumps(final_metrics, indent=2))

    torch.save({"state_dict": critic.state_dict(), "mean": mean, "std": std,
                "in_dim": feats.shape[1]}, os.path.join(args.out, "scvc_critic.pt"))
    with open(os.path.join(args.out, "scvc_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(args.out, "scvc_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"wrote {args.out}/scvc_critic.pt, scvc_metrics.json, scvc_history.json")


if __name__ == "__main__":
    main()
