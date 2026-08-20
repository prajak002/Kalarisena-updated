#!/usr/bin/env python3
"""Paper Figure 1 (motion-corpus montage) and Figure 5 (runtime pipeline diagram).

Figure 1: representative frames from the REAL curated Kalari clips in
data/kalari_videos/, grouped by family per data/kalari_sources.csv, color-coded
by support pattern as the draft's placeholder requests. The montage shows the
actual dataset, not stock imagery; the current corpus size is printed on the
figure so it cannot silently overclaim "100+" while 12 exist.

Figure 5: the runtime physics pipeline exactly as implemented (state readout ->
StateConverter -> PinocchioWrapper -> support/CP features -> ModeSwitch ->
policy/PD -> MuJoCo), matching src/sim/controller.py.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

OUT = "results_paper"

FAMILY_COLORS = {
    "vadivu": "#2a9d8f", "chuvadu": "#e9c46a", "kicks": "#c1121f",
    "empty_hand": "#457b9d", "meypayattu": "#8d5a97", "unlabeled": "#888888",
}
SUPPORT = {
    "vadivu": "double support", "chuvadu": "rapid transfer", "kicks": "single support",
    "empty_hand": "double support", "meypayattu": "mixed",
}


def load_spec() -> dict[str, str]:
    fam = {}
    path = "data/kalari_sources.csv"
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)):
            mid = (r.get("motion_id") or "").strip()
            if mid and not mid.startswith("#"):
                fam[mid] = (r.get("family") or "unlabeled").strip()
    return fam


def grab_frame(path: str, frac: float = 0.5):
    import cv2

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_fig1() -> None:
    fam_map = load_spec()
    vids = sorted(f for f in os.listdir("data/kalari_videos") if f.endswith(".mp4"))
    entries = []
    for v in vids:
        mid = v[:-4]
        if mid.startswith("test_"):
            continue  # keep only the curated, family-labelled corpus
        entries.append((fam_map.get(mid, "unlabeled"), mid, os.path.join("data/kalari_videos", v)))
    entries.sort()
    if not entries:
        raise SystemExit("no curated clips found in data/kalari_videos")

    ncols = 5
    nrows = (len(entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows))
    axes = axes.reshape(nrows, ncols)
    for ax in axes.flat:
        ax.axis("off")

    for i, (fam, mid, path) in enumerate(entries):
        ax = axes[i // ncols][i % ncols]
        frame = grab_frame(path)
        if frame is None:
            continue
        ax.imshow(frame)
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.set_title(mid.replace("_", " "), fontsize=8)
        ax.text(0.02, 0.04, f"{fam} · {SUPPORT.get(fam, '?')}",
                transform=ax.transAxes, fontsize=7, color="white",
                bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"))
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(3)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Kalaripayattu motion corpus - current curated set: {len(entries)} clips "
        f"(target 70+; sources: Kerala Tourism, Kalari Warriors)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig1_motion_corpus.{ext}")
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor="#1c2322", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.0012,
                            head_width=0.012, head_length=0.014,
                            length_includes_head=True, color="#1c2322"))


def make_fig5() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    row1, row2 = 0.66, 0.16
    bh = 0.22
    c_state, c_phys, c_ctl, c_out = "#dbe7f4", "#d9efe9", "#f6e7cf", "#efd9d7"

    _box(ax, 0.015, row1, 0.13, bh, "MuJoCo state\n$q, \\dot q$, contacts\n(500 Hz)", c_state)
    _box(ax, 0.175, row1, 0.14, bh, "StateConverter\nquat wxyz$\\to$xyzw\nname-keyed joints\nworld$\\to$body vel", c_state)
    _box(ax, 0.345, row1, 0.15, bh, "PinocchioWrapper\nCoM, $\\dot c$, $h_G$\nframe placements", c_phys)
    _box(ax, 0.525, row1, 0.155, bh, "Support features\npolygon (4-sphere feet)\n$\\xi = c + \\dot c/\\omega$\n$R_{cp}, R_{com}$", c_phys)
    _box(ax, 0.71, row1, 0.13, bh, "ModeSwitch\n$\\delta_1, \\delta_2, \\delta_3$\nhysteresis + dwell", c_ctl)
    _box(ax, 0.87, row1, 0.115, bh, "Mode\nnominal / fall /\nrecovery", c_ctl)

    _box(ax, 0.87, row2, 0.115, bh, "StepLog CSV\n38 columns\n(section 7 schema)", c_out)
    _box(ax, 0.71, row2, 0.13, bh, "PD torque\n$\\tau = k_p e - k_d \\dot q$\nclip to limits", c_ctl)
    _box(ax, 0.525, row2, 0.155, bh, "Target selection\nnominal: reference $q^\\star$\nfall: protective crouch\n(scripted)", c_ctl)
    _box(ax, 0.345, row2, 0.15, bh, "Reference motion\nNPZ contract or\nauthored keyframes", c_state)
    _box(ax, 0.015, row2, 0.29, bh, "MuJoCo mj_step x10 (decimation)\n50 Hz control / 500 Hz physics\narmature 0.01 (stability bound $k_d\\,\\Delta t/I<2$)", c_out)

    _arrow(ax, 0.145, row1 + bh / 2, 0.175, row1 + bh / 2)
    _arrow(ax, 0.315, row1 + bh / 2, 0.345, row1 + bh / 2)
    _arrow(ax, 0.495, row1 + bh / 2, 0.525, row1 + bh / 2)
    _arrow(ax, 0.68, row1 + bh / 2, 0.71, row1 + bh / 2)
    _arrow(ax, 0.84, row1 + bh / 2, 0.87, row1 + bh / 2)
    _arrow(ax, 0.9275, row1, 0.9275, row2 + bh)          # mode -> log
    _arrow(ax, 0.87, row2 + bh / 2, 0.84, row2 + bh / 2)  # pd <- log side flows leftwards
    _arrow(ax, 0.71, row2 + bh / 2, 0.68, row2 + bh / 2)
    _arrow(ax, 0.525, row2 + bh / 2, 0.495, row2 + bh / 2)
    _arrow(ax, 0.345, row2 + bh / 2, 0.305, row2 + bh / 2)
    _arrow(ax, 0.08, row2 + bh, 0.08, row1)               # sim state feeds back up

    ax.set_title("Runtime physics pipeline as implemented "
                 "(src/sim/controller.py; cross-engine CoM agreement 1e-6 m)",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig5_runtime_pipeline.{ext}")
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_fig1()
    make_fig5()
