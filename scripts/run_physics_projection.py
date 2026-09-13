#!/usr/bin/env python3
"""Run src/projection/physics_ground.py over the motion library, preserving
a pre-projection copy of each NPZ before modifying it in place.

Usage
  python3 scripts/run_physics_projection.py
  python3 scripts/run_physics_projection.py --npz-dir data/motions_retargeted
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/motions_retargeted")
    ap.add_argument("--preproj-dir", default="data/motions_retargeted_preproj")
    ap.add_argument("--out", default="results_paper/physics_projection_report.json")
    args = ap.parse_args()

    from src.projection.physics_ground import MotionProjector

    files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no NPZs in {args.npz_dir}")

    os.makedirs(args.preproj_dir, exist_ok=True)
    for f in files:
        dst = os.path.join(args.preproj_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(args.npz_dir, f), dst)
    print(f"pre-projection NPZs preserved in {args.preproj_dir}/\n")

    mp = MotionProjector()
    reports, failures = [], []
    for f in files:
        path = os.path.join(args.npz_dir, f)
        try:
            r = mp.project_one(path)
            reports.append(r)
            b, a = r["before"], r["after"]
            print(f"  {r['motion_id']:26s} "
                  f"cp_infeasible {b['cp_infeasible_frac']:.0%}->{a['cp_infeasible_frac']:.0%}  "
                  f"slip {b['mean_contact_slip_mps']:.2f}->{a['mean_contact_slip_mps']:.2f} m/s  "
                  f"tau_viol {b['torque_violation_frac']:.0%}->{a['torque_violation_frac']:.0%}")
        except Exception as e:
            failures.append({"motion_id": f[:-4], "error": str(e)})
            print(f"  {f[:-4]:26s} FAILED: {e}")
    mp.close()

    def _agg(key_path, arm):
        vals = [r[arm][key_path] for r in reports]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "n_motions": len(reports),
        "n_failed": len(failures),
        "failures": failures,
        "before": {k: _agg(k, "before") for k in reports[0]["before"]} if reports else {},
        "after": {k: _agg(k, "after") for k in reports[0]["after"]} if reports else {},
        "per_motion": reports,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{len(reports)}/{len(files)} motions projected ({len(failures)} failed)")
    print(json.dumps({"before": summary["before"], "after": summary["after"]}, indent=2))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
