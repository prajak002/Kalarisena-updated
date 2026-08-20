#!/usr/bin/env python3
"""Paper result pipeline (arXiv draft section 7.10): raw CSVs -> paper tables.

Implements the draft's fixed reporting protocol:
  raw per-trial rows (Table 39 schema)
    -> motion_summary.csv   (mean over trials of one motion)
    -> family_summary.csv   (mean over motions of one family)
    -> model_summary.json   (global summary, uniform over families)
    -> tables.tex           (LaTeX table bodies generated from the CSVs)

HONESTY CONTRACT
  * Every number is read from a real results CSV produced by an actual run
    (results_sim/push_sweep.csv, fall_ab.csv, track_*.csv). Nothing is typed in.
  * The controller behind these rows is the SCRIPTED PD + threshold switch
    (model_name "scripted_pd_switch_v0"), not a trained policy. The paper's
    Stage A-F policies do not exist yet; these tables are the protocol running
    end-to-end on the scripted baseline, and must be labelled as such.
  * Aggregation rule (stated per section 7.6): model level = uniform mean over
    families.

Usage:  python3 scripts/make_tables.py --results results_sim --out results_paper
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

MODEL_NAME = "scripted_pd_switch_v0"

# Motion -> family, using the taxonomy names of the draft (Table 28).
FAMILY = {
    "horse_stance_hold": "stable_stance",
    "single_leg_front_kick": "explosive_strike",
    "trunk_pivot_strike_prep": "rotational",
}

TRIAL_FIELDS = [
    "model_name", "motion_id", "family", "trial_id", "perturbation",
    "success", "fall", "slip", "impact",
    "recovery_success", "resume_success",
    "com_margin_mean", "cp_margin_mean", "momentum_norm_mean", "switch_count",
]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v, nd=3):
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def step_csv_stats(path: str) -> dict:
    """Per-trial means computed from a step-level CSV (real logged rows)."""
    com_m, cp_m, mom = [], [], []
    modes = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                cm, cpm = float(row["com_margin"]), float(row["cp_margin"])
            except (KeyError, ValueError):
                continue
            if cm > -900:
                com_m.append(cm)
            if cpm > -900:
                cp_m.append(cpm)
            mom.append(float(row["momentum_norm"]))
            modes.append(row["mode"])
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    return {
        "com_margin_mean": _mean(com_m),
        "cp_margin_mean": _mean(cp_m),
        "momentum_norm_mean": _mean(mom),
        "switch_count": switches,
    }


def collect_trials(results: str) -> list[dict]:
    trials: list[dict] = []

    # ---- nominal tracking (eval_nominal analogue) -------------------------
    summary_path = os.path.join(results, "track_summary.json")
    if os.path.isfile(summary_path):
        for s in json.load(open(summary_path)):
            mid = s["motion_id"]
            step_csv = os.path.join(results, f"track_{mid}.csv")
            stats = step_csv_stats(step_csv) if os.path.isfile(step_csv) else {}
            trials.append({
                "model_name": MODEL_NAME, "motion_id": mid,
                "family": FAMILY.get(mid, "unlabeled"), "trial_id": 0,
                "perturbation": "nominal",
                "success": int(s["stable"]), "fall": int(not s["stable"]),
                "slip": "", "impact": "",
                "recovery_success": "", "resume_success": "",
                **stats,
            })

    # ---- push perturbations (eval_perturb analogue) -----------------------
    push_path = os.path.join(results, "push_sweep.csv")
    if os.path.isfile(push_path):
        for row in csv.DictReader(open(push_path)):
            fell = int(row["fell"])
            trials.append({
                "model_name": MODEL_NAME, "motion_id": "horse_stance_hold",
                "family": FAMILY["horse_stance_hold"], "trial_id": int(row["trial"]),
                "perturbation": f"push_{float(row['force_N']):.0f}N",
                "success": int(not fell), "fall": fell, "slip": "", "impact": "",
                "recovery_success": "", "resume_success": "",
                "com_margin_mean": None,
                "cp_margin_mean": float(row["cp_margin_mean"]),
                "momentum_norm_mean": None,
                "switch_count": int(row["n_switch_events"]),
            })

    # ---- forced falls (eval_fall analogue) --------------------------------
    fall_path = os.path.join(results, "fall_ab.csv")
    if os.path.isfile(fall_path):
        for row in csv.DictReader(open(fall_path)):
            trials.append({
                "model_name": f"{MODEL_NAME}:{row['arm']}",
                "motion_id": "horse_stance_hold",
                "family": FAMILY["horse_stance_hold"], "trial_id": int(row["seed"]),
                "perturbation": "forced_fall",
                "success": 0, "fall": int(row["fell"]), "slip": "",
                "impact": float(row["peak_head_torso_N"]),
                "recovery_success": "", "resume_success": "",
                "com_margin_mean": None, "cp_margin_mean": None,
                "momentum_norm_mean": None,
                "switch_count": int(row["n_switch_events"]),
            })
    return trials


def aggregate(trials: list[dict]):
    by_motion = defaultdict(list)
    for t in trials:
        by_motion[(t["model_name"], t["motion_id"], t["perturbation"].split("_")[0])].append(t)

    motion_rows = []
    for (model, mid, pclass), ts in sorted(by_motion.items()):
        motion_rows.append({
            "model_name": model, "motion_id": mid, "family": ts[0]["family"],
            "perturbation_class": pclass, "n_trials": len(ts),
            "success_rate": _mean([t["success"] for t in ts]),
            "fall_rate": _mean([t["fall"] for t in ts]),
            "impact_mean": _mean([t["impact"] for t in ts if t["impact"] != ""]),
            "com_margin_mean": _mean([t["com_margin_mean"] for t in ts]),
            "cp_margin_mean": _mean([t["cp_margin_mean"] for t in ts]),
            "momentum_norm_mean": _mean([t["momentum_norm_mean"] for t in ts]),
            "switch_count_mean": _mean([t["switch_count"] for t in ts]),
        })

    by_family = defaultdict(list)
    for m in motion_rows:
        by_family[(m["model_name"], m["family"], m["perturbation_class"])].append(m)
    family_rows = []
    for (model, fam, pclass), ms in sorted(by_family.items()):
        family_rows.append({
            "model_name": model, "family": fam, "perturbation_class": pclass,
            "n_motions": len(ms),
            "success_rate": _mean([m["success_rate"] for m in ms]),
            "fall_rate": _mean([m["fall_rate"] for m in ms]),
            "impact_mean": _mean([m["impact_mean"] for m in ms]),
            "cp_margin_mean": _mean([m["cp_margin_mean"] for m in ms]),
            "switch_count_mean": _mean([m["switch_count_mean"] for m in ms]),
        })

    by_model = defaultdict(list)
    for f in family_rows:
        by_model[f["model_name"]].append(f)
    model_summary = {}
    for model, fs in sorted(by_model.items()):
        model_summary[model] = {
            "aggregation_rule": "uniform mean over families (section 7.6)",
            "n_family_rows": len(fs),
            "success_rate": _mean([f["success_rate"] for f in fs]),
            "fall_rate": _mean([f["fall_rate"] for f in fs]),
            "impact_mean": _mean([f["impact_mean"] for f in fs]),
        }
    return motion_rows, family_rows, model_summary


def latex_tables(trials, motion_rows) -> str:
    """LaTeX bodies for the draft, generated (never hand-assembled)."""
    out = []

    out.append("% ---- push-recovery sweep (real data: results_sim/push_sweep.csv) ----")
    out.append(r"\begin{tabular}{rrr} \toprule")
    out.append(r"Push (N) & Fall rate & mean $R_\mathrm{cp}$ after push (m) \\ \midrule")
    push = defaultdict(list)
    for t in trials:
        if t["perturbation"].startswith("push_"):
            push[float(t["perturbation"][5:-1])].append(t)
    # cp-after-push lives only in the raw csv; re-read for precision
    cp_after = defaultdict(list)
    for row in csv.DictReader(open("results_sim/push_sweep.csv")):
        cp_after[float(row["force_N"])].append(float(row["cp_margin_min_after_push"]))
    for force in sorted(push):
        ts = push[force]
        out.append(f"{force:.0f} & {_mean([t['fall'] for t in ts]):.2f} & "
                   f"{_mean(cp_after[force]):+.3f} \\\\")
    out.append(r"\bottomrule \end{tabular}")
    out.append("")

    out.append("% ---- impact-severity A/B (real data: results_sim/fall_ab.csv) ----")
    out.append("% NOTE: bimodal outcome; contact-rate decomposition is the honest read.")
    out.append(r"\begin{tabular}{lrrr} \toprule")
    out.append(r"Arm & Torso contact & Peak force given contact (N) & Impulse mean (N\,s) \\ \midrule")
    for arm in ("tracking_only", "protective"):
        rows = [r for r in csv.DictReader(open("results_sim/fall_ab.csv")) if r["arm"] == arm]
        hits = [float(r["peak_torso_force_N"]) for r in rows if float(r["peak_torso_force_N"]) > 0]
        imp = [float(r["torso_impulse_Ns"]) for r in rows]
        out.append(f"{arm.replace('_', ' ')} & {len(hits)}/{len(rows)} & "
                   f"{_fmt(_mean(hits), 1)} & {_fmt(_mean(imp), 1)} \\\\")
    out.append(r"\bottomrule \end{tabular}")
    out.append("")

    out.append("% ---- nominal tracking per motion (real data: track_*.csv) ----")
    out.append(r"\begin{tabular}{llrrr} \toprule")
    out.append(r"Motion & Family & Stable & mean $R_\mathrm{cp}$ (m) & mean $\|h_G\|$ \\ \midrule")
    for m in motion_rows:
        if m["perturbation_class"] != "nominal":
            continue
        out.append(f"{m['motion_id'].replace('_', ' ')} & {m['family'].replace('_', ' ')} & "
                   f"{'yes' if m['success_rate'] else 'no'} & "
                   f"{_fmt(m['cp_margin_mean'])} & {_fmt(m['momentum_norm_mean'], 2)} \\\\")
    out.append(r"\bottomrule \end{tabular}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_sim")
    ap.add_argument("--out", default="results_paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    trials = collect_trials(args.results)
    if not trials:
        raise SystemExit(f"no result CSVs found under {args.results}; run the experiments first")

    with open(os.path.join(args.out, "trials.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS)
        w.writeheader()
        for t in trials:
            w.writerow({k: ("" if t.get(k) is None else t.get(k, "")) for k in TRIAL_FIELDS})

    motion_rows, family_rows, model_summary = aggregate(trials)

    for name, rows in (("motion_summary.csv", motion_rows), ("family_summary.csv", family_rows)):
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    meta = {
        "model_name": MODEL_NAME,
        "controller": "scripted PD + threshold ModeSwitch (NOT a trained policy)",
        "source_files": sorted(os.path.basename(p) for p in
                               glob.glob(os.path.join(args.results, "*.csv"))),
        "note": "protocol run end-to-end on the scripted baseline; Stage A-F "
                "learned policies do not exist yet",
    }
    with open(os.path.join(args.out, "model_summary.json"), "w") as fh:
        json.dump({"meta": meta, "models": model_summary}, fh, indent=2)

    with open(os.path.join(args.out, "tables.tex"), "w") as fh:
        fh.write(latex_tables(trials, motion_rows))

    print(f"trials          : {len(trials)}")
    print(f"motion rows     : {len(motion_rows)}")
    print(f"family rows     : {len(family_rows)}")
    print(f"models          : {list(model_summary)}")
    for f in ("trials.csv", "motion_summary.csv", "family_summary.csv",
              "model_summary.json", "tables.tex"):
        print(f"wrote {os.path.join(args.out, f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
