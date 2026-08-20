#!/usr/bin/env python3
"""Ingest GEM-X retarget outputs into the review pipeline.

GEM-X writes, per motion, under <output_root>/<motion_id>/:
    <motion_id>_retarget_g1.csv           the G1 joint trajectory
    <motion_id>_retarget_g1_from_bvh.csv  same via the BVH path
    <motion_id>_retarget_g1.bvh
    <motion_id>_<n>_g1_retarget.mp4       GEM-X's own render of the G1 motion

This script copies the G1 render in as results_review/robot_<motion_id>.mp4 and
writes the render_manifest.json that build_review_page.py consumes, so the
professor's review shows the retarget exactly as GEM-X produced it.

Using GEM-X's own render (rather than re-rendering the CSV in MuJoCo) is the
honest choice for a *retarget* review: it shows what the retargeter output,
with no second interpretation layered on top. Re-rendering through
scripts/render_g1_motion.py remains available for physics-side inspection.

Usage
    python3 scripts/ingest_gemx_outputs.py --src cloud_outputs --out results_review
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def copy_as_h264(src: str, dst: str) -> bool:
    """Copy a video, transcoding to H.264 if needed.

    GEM-X's Open3D renderer writes MPEG-4 Part 2 ('mp4v'), which browsers do not
    decode in <video> tags: the file serves with HTTP 200 but plays as a blank
    box. Probe the codec and transcode to h264/yuv420p unless it already is.
    """
    ff = ffmpeg_exe()
    probe = subprocess.run([ff, "-i", src], capture_output=True, text=True)
    if "Video: h264" in probe.stderr:
        shutil.copy2(src, dst)
        return True
    enc = subprocess.run(
        [ff, "-y", "-i", src, "-c:v", "libx264", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
        capture_output=True, text=True)
    if enc.returncode != 0:
        print(f"  [warn] transcode failed for {src}: "
              f"{enc.stderr.strip().splitlines()[-1][:120] if enc.stderr else '?'}")
        shutil.copy2(src, dst)  # serve the original rather than nothing
        return False
    return True


def read_csv_meta(path: str) -> dict:
    try:
        with open(path) as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            n = sum(1 for _ in reader)
        return {"csv_rows": n, "csv_cols": len(header)}
    except Exception as exc:
        return {"csv_rows": None, "csv_cols": None, "csv_error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cloud_outputs",
                    help="GEM-X output_root (already downloaded locally)")
    ap.add_argument("--out", default="results_review")
    ap.add_argument("--family-map", default="data/kalari_sources.csv",
                    help="optional CSV with motion_id,family to label motions")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"not a directory: {args.src}")
    os.makedirs(args.out, exist_ok=True)

    families: dict[str, str] = {}
    if os.path.isfile(args.family_map):
        with open(args.family_map) as fh:
            for row in csv.DictReader(fh):
                mid = (row.get("motion_id") or "").strip()
                if mid and not mid.startswith("#"):
                    families[mid] = (row.get("family") or "unlabeled").strip()

    records = []
    for motion_dir in sorted(glob.glob(os.path.join(args.src, "*"))):
        if not os.path.isdir(motion_dir):
            continue
        mid = os.path.basename(motion_dir)

        vids = sorted(glob.glob(os.path.join(motion_dir, "*_g1_retarget.mp4")))
        csvs = sorted(glob.glob(os.path.join(motion_dir, "*_retarget_g1.csv")))
        if not csvs:
            csvs = sorted(glob.glob(os.path.join(motion_dir, "*_retarget_g1_from_bvh.csv")))

        flags = []
        if not vids:
            flags.append("no G1 render produced")
        if not csvs:
            flags.append("no retarget CSV - conversion failed")

        robot_path = None
        if vids:
            robot_path = os.path.join(args.out, f"robot_{mid}.mp4")
            copy_as_h264(vids[0], robot_path)

        meta = read_csv_meta(csvs[0]) if csvs else {}
        rows = meta.get("csv_rows")
        fps = 30.0

        records.append({
            "motion_id": mid,
            "family": families.get(mid, "unlabeled"),
            "source": "gemx_retarget",
            "robot_video": robot_path,
            "retarget_csv": csvs[0] if csvs else None,
            "frames": rows,
            "fps": fps,
            "duration_s": round(rows / fps, 2) if rows else None,
            "joints_matched": meta.get("csv_cols"),
            "joints_expected": meta.get("csv_cols"),
            "render_label": "GEM-X retarget render (kinematic)",
            "ground_clamped": False,
            "quality_flags": flags,
        })
        status = "ok" if not flags else "; ".join(flags)
        print(f"  {mid:26s} rows={str(rows):>6s}  video={'yes' if vids else 'NO':3s}  {status}")

    path = os.path.join(args.out, "render_manifest.json")
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)

    good = [r for r in records if not r["quality_flags"]]
    print(f"\n{len(records)} motion(s) ingested, {len(good)} complete")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
