#!/usr/bin/env bash
# Reproduce every artifact in results_sim/ from scratch, in order.
#
#   bash scripts/run_demo_all.sh
#
# Requires the py3.11 virtualenv at .venv (see README / DEMO_REPORT.md setup).
# Each stage gates the next: if the Stage-1 gate fails, nothing downstream runs.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found."
  echo "Create it with:"
  echo "  uv venv --python python3.11 .venv"
  echo "  VIRTUAL_ENV=\$PWD/.venv uv pip install numpy scipy matplotlib pandas \\"
  echo "      mujoco imageio imageio-ffmpeg pyyaml shapely pin"
  exit 1
fi

mkdir -p results_sim results_demo

banner() { echo; echo "############ $* ############"; echo; }

banner "STAGE 0  module smoke tests"
$PY src/dynamics/pinocchio_wrapper.py
$PY src/switch/mode_switch.py

banner "STAGE 0b  regression tests"
$PY tests/test_conventions.py
$PY tests/test_mode_switch.py

banner "STAGE 1  MuJoCo gate (standing + cross-engine agreement)"
$PY scripts/test_sim_stage1.py

banner "STAGE 1b  PD gain study -> results_sim/gain_study.csv"
$PY scripts/gain_study.py

banner "STAGE 2  reference tracking in contact dynamics"
$PY scripts/sim_track_motion.py --all --out results_sim

banner "STAGE 2b  kinematic replay videos (labelled kinematic)"
$PY scripts/sim_kinematic_replay.py --all --out results_sim

banner "STAGE 3A  push-recovery sweep"
$PY scripts/sim_push_sweep.py --out results_sim

banner "STAGE 3B  impact-severity A/B"
$PY scripts/sim_fall_ab.py --out results_sim

banner "DONE"
echo "Artifacts in results_sim/:"
ls -la results_sim
echo
echo "Read results_sim/DEMO_REPORT.md for the write-up."
