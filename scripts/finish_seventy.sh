#!/usr/bin/env bash
# One-shot finishing chain after the GPU batch completes all 70 retargets.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IP=${1:-65.2.29.24}

echo "== 1/7 pull outputs from GPU box =="
[ "${SKIP_PULL:-0}" = "1" ] && echo "  (skipped)" || rsync -az --include='*/' --include='*_retarget_g1*.csv' --include='*_g1_retarget.mp4' --exclude='*' \
  -e "ssh -o StrictHostKeyChecking=no -i $HOME/.ssh/kalari-gemx.pem" \
  ubuntu@$IP:~/cloud_outputs/ cloud_outputs/

echo "== 2/7 ingest GEM-X renders (h264 transcode) =="
$PY scripts/ingest_gemx_outputs.py --src cloud_outputs --out results_review

echo "== 3/7 CSV -> NPZ library =="
$PY scripts/annotate_motion_library.py --csv-root cloud_outputs \
  --output-dir data/motions_retargeted --splits-dir data/splits \
  --urdf assets/unitree_g1/g1.urdf --fps 30 --pos-scale 0.01 --angles-deg \
  --root-euler-order xyz --quat-order xyzw

echo "== 4/7 ground correction =="
$PY scripts/ground_correct_motions.py
cp results_review/diagnostics.json results_review/diagnostics_before.json 2>/dev/null || true

echo "== 5/7 physics diagnostics (before snapshot taken from raw pass) =="
# raw-state diagnostics for the before/after chips: run on raw dir via temp swap
$PY - <<'EOF'
import json, os, shutil, subprocess, sys
# 'before' = diagnostics of RAW npz: run diagnose on raw dir quickly (no video)
if os.path.isdir('data/motions_retargeted_raw'):
    env=dict(os.environ)
    r=subprocess.run(['.venv/bin/python','-c','''
import sys; sys.path.insert(0,'.')
import scripts.motion_diagnostics as MD
MD.NPZ_DIR='data/motions_retargeted_raw'
sys.argv=['x','--no-video']
MD.main()
'''])
    if r.returncode==0:
        shutil.copy2('results_review/diagnostics.json','results_review/diagnostics_before.json')
        print('raw (before) diagnostics captured')
EOF
$PY scripts/motion_diagnostics.py

echo "== 6/7 fixed-camera raw/corrected/overlay renders + stills =="
$PY scripts/render_compare.py

echo "== 7/7 page build + compressed deploy =="
$PY scripts/build_review_page.py --human-dir data/kalari_videos --require-human \
  --out results_review/review.html
$PY scripts/build_web_deploy.py --clean --compress
rm -rf kalarisena-review && cp -r web kalarisena-review
mkdir -p kalarisena-review/.vercel/output/static
cp kalarisena-review/index.html kalarisena-review/*.mp4 kalarisena-review/*.png kalarisena-review/.vercel/output/static/ 2>/dev/null || true
cp -r kalarisena-review/human kalarisena-review/.vercel/output/static/ 2>/dev/null || true
echo '{"version":3,"routes":[{"handle":"filesystem"}]}' > kalarisena-review/.vercel/output/config.json
(cd kalarisena-review && vercel deploy --prebuilt --prod --yes --scope mun-team </dev/null | grep '"url"' | head -1)
echo "DONE - https://kalarisena-review.vercel.app"
