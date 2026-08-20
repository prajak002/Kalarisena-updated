#!/usr/bin/env python3
"""Build the Kalari clip dataset: search YouTube, cut atomic clips, screen quality.

Three modes:

  search   Find candidate source videos.
             python3 scripts/build_video_dataset.py search "kalaripayattu vadivu" -n 10

  probe    Download one source at low quality and print a contact sheet of
           timestamps, so you can pick clip in/out points quickly.
             python3 scripts/build_video_dataset.py probe --video-id yVr3ap7i6W4

  cut      Read a clip spec CSV and produce data/kalari_videos/<motion_id>.mp4
             python3 scripts/build_video_dataset.py cut --spec data/kalari_sources.csv

Clip spec CSV columns:
    motion_id,family,video_id,start,end,notes
e.g.
    vadivu_gaja,vadivu,yVr3ap7i6W4,00:00:31,00:00:39,elephant stance hold

Only the requested section is downloaded (yt-dlp --download-sections), so cutting
70 clips does not mean downloading 70 full videos.

Clips are normalised for GEM-X: 30 fps, no audio, H.264, even dimensions.

NOTE ON RIGHTS: these are third-party videos. Using them for research/retargeting
is one thing; republishing the clips on a public URL is another. If you deploy the
review page publicly, prefer password-protecting it (Vercel project settings ->
Deployment Protection) or self-record footage. This script does not check licences.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

RAW_DIR = "data/kalari_raw"
OUT_DIR = "data/kalari_videos"
SPEC = "data/kalari_sources.csv"


def ytdlp() -> str:
    local = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    return local if os.path.exists(local) else "yt-dlp"


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run(cmd: list[str], quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("  $", " ".join(cmd[:6]), "...")
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_search(args) -> int:
    p = run([ytdlp(), f"ytsearch{args.n}:{args.query}", "--flat-playlist", "--no-warnings",
             "--print", "%(duration)s|%(id)s|%(channel).24s|%(title).70s"], quiet=True)
    rows = [l for l in p.stdout.splitlines() if "|" in l]
    print(f"{'dur':>6} {'video_id':<13} {'channel':<24} title")
    for r in sorted(rows, key=lambda x: int(x.split("|")[0] or 0)):
        d, vid, ch, title = (r.split("|", 3) + ["", "", ""])[:4]
        mins = f"{int(d)//60}:{int(d)%60:02d}" if d.isdigit() else d
        print(f"{mins:>6} {vid:<13} {ch:<24} {title}")
    if not rows:
        print("(no results)", p.stderr[-300:])
    return 0


def cmd_probe(args) -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={args.video_id}"
    out = os.path.join(RAW_DIR, f"{args.video_id}.mp4")
    if not os.path.exists(out):
        print(f"downloading preview of {args.video_id} ...")
        p = run([ytdlp(), "-f", "bv[height<=480]+ba/b[height<=480]/b",
                 "--ffmpeg-location", ffmpeg(),
                 "--merge-output-format", "mp4", "-o", out, url])
        if p.returncode != 0:
            print(p.stderr[-800:])
            return 1
    sheet = os.path.join(RAW_DIR, f"{args.video_id}_contact.jpg")
    run([ffmpeg(), "-y", "-i", out, "-vf",
         "fps=1/3,scale=240:-2,tile=6x5,drawtext=text='%{pts\\:hms}':fontsize=12:"
         "fontcolor=yellow:x=4:y=4", "-frames:v", "1", sheet])
    print(f"\ncontact sheet: {sheet}  (one frame every 3s, timestamps burned in)")
    print(f"source video : {out}")
    print("\nOpen the sheet, note in/out times, then add rows to your spec CSV.")
    return 0


def cmd_cut(args) -> int:
    if not os.path.isfile(args.spec):
        raise SystemExit(f"spec not found: {args.spec}\nSee --help for the CSV format.")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(args.spec) as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("motion_id") and not r["motion_id"].startswith("#")]
    print(f"{len(rows)} clips in {args.spec}\n")

    manifest, ok_n, fail_n = [], 0, 0
    for i, r in enumerate(rows, 1):
        mid = r["motion_id"].strip()
        dest = os.path.join(OUT_DIR, f"{mid}.mp4")
        if os.path.exists(dest) and not args.force:
            print(f"[{i}/{len(rows)}] {mid}: exists, skipping")
            manifest.append({**r, "file": dest, "status": "cached"})
            ok_n += 1
            continue

        url = f"https://www.youtube.com/watch?v={r['video_id'].strip()}"
        section = f"*{r['start'].strip()}-{r['end'].strip()}"
        tmp = os.path.join(RAW_DIR, f"_tmp_{mid}.mp4")
        os.makedirs(RAW_DIR, exist_ok=True)

        print(f"[{i}/{len(rows)}] {mid}  <- {r['video_id']} {r['start']}..{r['end']}")
        # yt-dlp needs ffmpeg on PATH for --download-sections; ours is the
        # binary bundled with imageio-ffmpeg, so point at it explicitly.
        p = run([ytdlp(), "--download-sections", section, "--force-keyframes-at-cuts",
                 "--ffmpeg-location", ffmpeg(),
                 "-f", "bv[height<=1080]+ba/b", "--merge-output-format", "mp4",
                 "-o", tmp, "--no-warnings", url], quiet=True)
        ss_args: list[str] = []
        if p.returncode != 0 or not os.path.exists(tmp):
            # Section download can fail on some sources (ffmpeg keyframe errors).
            # Fall back to caching the full video once and cutting locally.
            full = os.path.join(RAW_DIR, f"{r['video_id'].strip()}.mp4")
            if not os.path.exists(full):
                print("   section download failed, fetching full video as fallback ...")
                f = run([ytdlp(), "-f", "bv[height<=1080]+ba/b", "--merge-output-format", "mp4",
                         "--ffmpeg-location", ffmpeg(), "-o", full, "--no-warnings", url],
                        quiet=True)
                if f.returncode != 0 or not os.path.exists(full):
                    print(f"   DOWNLOAD FAILED: "
                          f"{f.stderr.strip().splitlines()[-1][:160] if f.stderr else '?'}")
                    manifest.append({**r, "file": None, "status": "download_failed"})
                    fail_n += 1
                    continue
            tmp = full
            ss_args = ["-ss", r["start"].strip(), "-to", r["end"].strip()]

        # Optional crop, applied BEFORE scaling. GEM-X tracks a single person, so
        # footage with two performers side by side must be cropped to isolate one.
        # Value is a raw ffmpeg crop expression, e.g. "iw/2:ih:iw/2:0" (right half).
        crop = (r.get("crop") or "").strip()
        vf = f"crop={crop}," if crop else ""
        vf += "scale=trunc(iw/2)*2:trunc(ih/2)*2"

        # normalise for GEM-X: 30fps, no audio, even dims, H.264
        q = run([ffmpeg(), "-y", *ss_args, "-i", tmp, "-an", "-r", "30",
                 "-vf", vf,
                 "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", dest], quiet=True)
        if tmp.startswith(os.path.join(RAW_DIR, "_tmp_")):
            os.remove(tmp)
        if q.returncode != 0 or not os.path.exists(dest):
            print(f"   ENCODE FAILED: {q.stderr.strip().splitlines()[-1][:160] if q.stderr else '?'}")
            manifest.append({**r, "file": None, "status": "encode_failed"})
            fail_n += 1
            continue

        info = probe_clip(dest)
        manifest.append({**r, "file": dest, "status": "ok", **info})
        flag = "  <-- CHECK: " + info["flags"] if info["flags"] else ""
        print(f"   ok  {info['w']}x{info['h']} {info['fps']:.0f}fps {info['dur']:.1f}s{flag}")
        ok_n += 1

    path = os.path.join(OUT_DIR, "dataset_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{ok_n} ok, {fail_n} failed -> {OUT_DIR}/")
    print(f"manifest: {path}")
    flagged = [m for m in manifest if m.get("flags")]
    if flagged:
        print(f"\n{len(flagged)} clip(s) flagged for manual review:")
        for m in flagged:
            print(f"  {m['motion_id']}: {m['flags']}")
    return 0


def probe_clip(path: str) -> dict:
    import cv2  # opencv is optional; fall back to ffprobe-free defaults

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = n / fps if fps else 0.0
    flags = []
    if h < 720:
        flags.append("low-res")
    if dur < 2:
        flags.append("too short")
    if dur > 20:
        flags.append("too long, segment further")
    return {"w": w, "h": h, "fps": fps, "dur": dur, "flags": "; ".join(flags)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=10)
    s.set_defaults(func=cmd_search)

    p = sub.add_parser("probe"); p.add_argument("--video-id", required=True)
    p.set_defaults(func=cmd_probe)

    c = sub.add_parser("cut"); c.add_argument("--spec", default=SPEC)
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_cut)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
