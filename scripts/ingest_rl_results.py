#!/usr/bin/env python3
"""Ingest Stage A RL evaluation results into the review page.

Reads training-run directories (logs/stageA_<motion_id>/ from
scripts/train_tracking.py), copies the policy-execution eval video into
results_review/rl_<motion_id>.mp4 (transcoding to browser-safe H.264 if needed),
and writes results_review/rl_manifest.json for build_review_page.py.

HONESTY RULES
  * A motion appears with RL content only if a real trained checkpoint produced
    a real eval (eval_summary.json). No placeholders.
  * The policy-vs-PD comparison numbers are surfaced verbatim from the eval, so
    a policy that fails to beat the baseline is shown failing.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_gemx_outputs import copy_as_h264

OUT = "results_review"


def main() -> int:
    runs = sorted(glob.glob("logs/stageA_*"))
    records = []
    for run in runs:
        mid = os.path.basename(run).removeprefix("stageA_")
        summary_path = os.path.join(run, "eval_summary.json")
        video_path = os.path.join(run, "eval_policy.mp4")
        if not os.path.isfile(summary_path):
            print(f"  {mid}: no eval_summary.json yet (training incomplete) - skipped")
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)["summary"]
        pol, pd = summary["policy"], summary["pd_baseline"]

        video_out = None
        if os.path.isfile(video_path):
            video_out = os.path.join(OUT, f"rl_{mid}.mp4")
            copy_as_h264(video_path, video_out)

        rec = {
            "motion_id": mid,
            "video": video_out,
            "status_note": None,
            "metrics": {
                "rmse_policy": round(pol["tracking_rmse_mean"], 3),
                "rmse_pd": round(pd["tracking_rmse_mean"], 3),
                "len_policy": round(pol["mean_episode_len"], 1),
                "len_pd": round(pd["mean_episode_len"], 1),
                "fall_policy": round(pol["fall_rate"], 2),
                "fall_pd": round(pd["fall_rate"], 2),
            },
        }
        records.append(rec)
        better = pol["mean_episode_len"] > pd["mean_episode_len"]
        print(f"  {mid}: policy len {pol['mean_episode_len']:.0f} vs PD "
              f"{pd['mean_episode_len']:.0f} steps -> "
              f"{'POLICY BETTER' if better else 'policy NOT better (reported as-is)'}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rl_manifest.json"), "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\n{len(records)} RL result(s) -> {OUT}/rl_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
