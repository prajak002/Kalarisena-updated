#!/usr/bin/env python3
"""Stage G CLI: evolve a bridging trajectory between two real clips (docs/PHYSICAL_AI_STAGE_G_H_I_DRAFT.md sec 2).

Usage
  python3 scripts/evolve_joint_pool.py \
      --clip-a data/motions_retargeted/ky_warrior_lunge.npz \
      --clip-b data/motions_retargeted/pk_kick_lunge.npz \
      --out data/motions_evolved --generations 80 --population 150

Writes:
  <out>/<motion_id>.npz          evolved trajectory, annotate_motion_library.py schema
  <out>/<motion_id>_history.json per-generation best/mean fitness (real GA run, not typed in)
  <out>/<motion_id>_fitness.png  fitness-vs-generation plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ga.joint_pool import GAConfig, TransitionGA


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip-a", required=True, help="NPZ whose ENDING pose starts the bridge")
    p.add_argument("--clip-b", required=True, help="NPZ whose STARTING pose ends the bridge")
    p.add_argument("--out", default="data/motions_evolved")
    p.add_argument("--population", type=int, default=150)
    p.add_argument("--generations", type=int, default=80)
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--motion-id", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)

    id_a = os.path.splitext(os.path.basename(args.clip_a))[0]
    id_b = os.path.splitext(os.path.basename(args.clip_b))[0]
    motion_id = args.motion_id or f"evo_{id_a}__to__{id_b}"

    cfg = GAConfig(
        population=args.population,
        generations=args.generations,
        n_frames=args.n_frames,
        fps=args.fps,
        seed=args.seed,
    )

    print(f"Stage G: evolving bridge {id_a} -> {id_b}")
    print(f"  population={cfg.population} generations={cfg.generations} n_frames={cfg.n_frames}")

    ga = TransitionGA(args.clip_a, args.clip_b, cfg)
    try:
        best, history = ga.run()
        best_frames = ga.decode(best)

        joint_cols = np.array([f"{n}_dof" for n in ga.act_names], dtype=np.bytes_)
        payload = {
            "q": np.concatenate([ga.root_pos_path, ga.root_quat_path, best_frames], axis=1),
            "joint_pos": best_frames,
            "root_pos": ga.root_pos_path,
            "root_quat_xyzw": ga.root_quat_path,
            "joint_cols": joint_cols,
            "fps": np.array(cfg.fps, dtype=np.float32),
            "motion_id": np.array(motion_id.encode("utf-8")),
            "family": np.array(b"evolved_transition"),
            "ga_source_a": np.array(id_a.encode("utf-8")),
            "ga_source_b": np.array(id_b.encode("utf-8")),
        }
        out_path = os.path.join(args.out, f"{motion_id}.npz")
        np.savez(out_path, **payload)

        hist_path = os.path.join(args.out, f"{motion_id}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gens = [h["generation"] for h in history]
            best_f = [h["best_fitness"] for h in history]
            mean_f = [h["mean_fitness"] for h in history]
            plt.figure(figsize=(6, 4))
            plt.plot(gens, best_f, label="best fitness")
            plt.plot(gens, mean_f, label="mean fitness", alpha=0.6)
            plt.xlabel("generation")
            plt.ylabel("fitness")
            plt.title(f"Stage G: {id_a} -> {id_b}")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(args.out, f"{motion_id}_fitness.png")
            plt.savefig(fig_path, dpi=140)
            print(f"  wrote {fig_path}")
        except Exception as exc:
            print(f"  [warn] plot skipped: {exc}")

        print(f"Wrote {out_path}")
        print(f"Wrote {hist_path}")
        print(f"Final best fitness: {history[-1]['best_fitness']:.4f}")
    finally:
        ga.close()


if __name__ == "__main__":
    main()
