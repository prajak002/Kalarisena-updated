# KalariSena

Kalaripayattu (Indian martial art) motion capture → Unitree G1 humanoid, trained
in MuJoCo with a staged PPO curriculum (tracking → CoM/capture-point →
momentum → fall-impact → recovery → mode switching), plus a learned viability
critic and a residual safety policy on top. This README describes what is
**actually implemented and measured** in this repo, not a target design.

## Status at a glance

Every row below has real code, a real trained checkpoint under `logs/`, and a
real measured number — several are honestly negative. Re-run
`--eval-only` on any stage yourself; nothing here is asserted without a
`logs/<stage>/eval_summary.json` backing it.

| Stage | What it does | Result |
|---|---|---|
| A — Tracking (single motion) | PPO residual tracking, `kw_long_stance` only | `logs/stageA_kw_long_stance/` — 3M steps, reward 6.65→44.8, falls every episode |
| A — Tracking (12-motion) | Same, sampled across 12 motions / all 3 families | `logs/stageA_multi12/` — fall rate 83.3% vs. 95.8% PD baseline |
| A — Tracking (full corpus) | Trains on the 56-motion train split, evaluates on 14 held-out motions never trained on | `logs/stageA_multi_full/` (in progress on GPU — see below) |
| B — CoM / capture-point | Adds `com_support`/`capture_point_margin` reward terms | `logs/stageB_com/` — cp_margin stays negative (-0.27), fall rate 100% |
| C — Momentum regulation | Adds phase-weighted angular-momentum penalty | `logs/stageC_momentum/` — momentum norm 3.83→0.79 (real drop), fall rate still 100% |
| D — Fall-impact minimization | Trains from near-fall states, penalizes impact force | `logs/stageD_fall/` — peak torso impact 31N→9N (the cleanest positive result in B–D) |
| E — Recovery-to-standing | Get up from a randomized fallen pose, no reference motion | Was 0% success (`logs/stageE_recovery/`) — root cause found (reward had zero gradient below the success threshold) and fixed; retraining in `logs/stageE_recovery_v2/` |
| F — Mode switching | Threshold FSM: NOMINAL / FALL / RECOVERY on capture-point margin + momentum | `logs/stageF_switch/` — switching to the fall policy shortens episodes (21.0 vs 38.2 steps) while lowering impact severity |
| SCVC viability critic | Learned success/failure classifier over rollout states | Single motion: AUROC 0.93 (`logs/scvc_kw_long_stance/`). 12 motions pooled: AUROC drops to 0.79 (`logs/scvc_multi12/`) — cross-motion generalization is measurably harder |
| Residual safety policy | Learned correction on top of the frozen tracker | Negative both at single-motion and 12-motion scope — underperforms the tracker alone (`logs/residual_kw_long_stance/`, `logs/residual_multi12/`) |
| Eval protocol (IPR / MPJPE) | Intent Preservation Rate + forward-kinematics body-position error, under a push | **IPR = 0%** for both the tracker and the residual-gated version, all 12 motions × 3 push levels (`logs/eval_protocol/`) — the single most important honest number in this repo: nothing here yet reliably completes a motion under a push |

**Still missing, not attempted here:** baselines (KungfuBot, CoRE, SONIC, etc.
— no confirmed public code to reproduce against), the full ~100-hour corpus
(currently ~70 retargeted clips), physical hardware deployment (explicitly
out of scope for now — simulation only).

## Repo layout

```
src/
  sim/           MuJoCo runtime, camera/render, rollout + video helpers
  dynamics/      PinocchioWrapper — CoM, capture-point, support polygon,
                 centroidal momentum. Every reward/feature reuses this;
                 nothing reimplements the physics.
  envs/          Gymnasium envs: kalari_track_env, multi_motion_env,
                 com_refine_env, momentum_env, fall_env, recovery_env
  rewards/       reward_builder.py — all reward terms, config-driven
  switch/        mode_switch.py — the NOMINAL/FALL/RECOVERY FSM
  viability/     SCVC critic, successor manifold, perturbation sampler,
                 residual policy
  projection/    Physics-grounding of retargeted motion (contact slip,
                 capture-point feasibility, torque limits)
  ga/            Joint-pool evolution (motion augmentation, experimental)
scripts/
  run_pipeline.py              video -> GEM-X retarget -> annotated NPZ
  train_tracking.py            Stage A, single motion
  train_tracking_multi.py      Stage A, fixed 12-motion subset
  train_tracking_multi_full.py Stage A, full train split + held-out eval
  train_com.py / train_momentum.py / train_fall.py / train_recovery.py
                                Stages B-E
  train_viability_critic*.py   SCVC critic (single- and multi-motion)
  train_residual_policy*.py    Residual policy (single- and multi-motion)
  eval_integrated_switch.py    Stage F: nominal + fall policy through the
                                real ModeSwitch
  eval_thrust_response.py      Single controllable push against a policy
  sim_controlled_perturbation.py
                                Multi-push, annotated-video demo — see below
  eval_protocol.py             IPR / MPJPE evaluation
  run_physics_projection.py    Motion corpus physics-grounding audit
configs/
  com.yaml, momentum.yaml, fall.yaml, recovery.yaml, switch.yaml
  motion_families.yaml   ~70 clips tagged stable_stance / translational /
                          explosive_strike, drives curriculum oversampling
data/
  motions_retargeted/    70 retargeted G1 reference clips (npz)
  splits/                train_ids.txt (56) / val_ids.txt (7) / test_ids.txt (7)
                          — used by train_tracking_multi_full.py for genuine
                          held-out generalization, not by the older 12-motion script
assets/unitree_g1/       G1 MJCF/URDF models and meshes
logs/                    every experiment's meta.json + eval_summary.json +
                          checkpoint — the source of truth for every number
                          in this README
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, mujoco, gymnasium, stable-baselines3,
                                   # torch, pinocchio (pin), scipy, shapely, imageio
```

GPU is optional and mostly irrelevant to training speed here: PPO with an
MlpPolicy over vectorized CPU MuJoCo envs is CPU-core-bound (env stepping),
not GPU-bound — SB3 itself warns against `device="cuda"` for this shape of
problem. What actually helps is more CPU cores for more parallel envs
(`--n-envs`); a rented box is useful for that, not for its GPU.

```bash
# BLAS thread explosion across subprocess envs will crash SubprocVecEnv above
# ~20-30 envs unless you cap this first:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 scripts/train_tracking_multi_full.py --steps 30000000 --n-envs 24 \
    --out logs/stageA_multi_full
```

## Data pipeline: video → retargeted motion

```bash
chmod +x scripts/install_subprojects.sh && bash scripts/install_subprojects.sh   # GEM-X, unitree_rl_mjlab
chmod +x scripts/install_pipeline.sh && bash scripts/install_pipeline.sh

python scripts/run_pipeline.py \
  --video /path/to/input.mp4 \
  --urdf /path/to/g1.urdf \
  --root-euler-order xyz \
  --angles-deg
```

This runs GEM-X retargeting (GPU, or ONNX Runtime on macOS without CUDA),
writes raw retarget output under `cloud_outputs/<motion_id>/`, then annotates
and writes the training-ready reference to `data/motions_retargeted/*.npz`.
Skip either half with `--skip-gemx` / `--skip-annotate`. GEM-X's own install
notes are at `GEM-X/README.md` / `GEM-X/docs/INSTALL.md`
(`GEM-X/docs/INSTALL_MACOS.md` for the no-CUDA path).

### Hugging Face dataset sync (private)

```bash
export HF_TOKEN="<your_hf_token>"

# pull source videos
python scripts/sync_hf_dataset.py --repo-id <org>/<dataset> --mode download \
  --remote-video-prefix videos --download-dir inputs/hf_videos

# push retargeted G1 references back
python scripts/sync_hf_dataset.py --repo-id <org>/<dataset> --mode upload \
  --upload-source cloud_outputs --remote-retarget-prefix retargeted_g1 --skip-existing

# one-shot pull -> process -> push loop, for cloud workers
python scripts/hf_e2e_pipeline.py --repo-id <org>/<dataset> --mode all \
  --remote-video-prefix videos --local-video-dir inputs/hf_videos \
  --output-root cloud_outputs --remote-retarget-prefix retargeted_g1 \
  --skip-existing-motion --skip-existing-remote --continue-on-error
```

## The controlled-perturbation demo

`scripts/sim_controlled_perturbation.py` applies a push pattern you specify
(timing, direction, magnitude, one push or a whole sequence) to a trained
policy and renders an annotated video: live capture-point margin, angular
momentum, active mode (when routed through the real `ModeSwitch`), and fall
status burned into the frames.

```bash
# one push, tracker only
python3 scripts/sim_controlled_perturbation.py \
  --nominal logs/stageA_multi_full/tracking_multi_full_best.zip \
  --npz data/motions_retargeted/kw_long_stance.npz \
  --push 1.0 90 80 0.1 --out logs/controlled_demo/single_push

# multi-push pattern, with real NOMINAL/FALL mode switching
python3 scripts/sim_controlled_perturbation.py \
  --nominal logs/stageA_multi12/tracking_multi_best.zip \
  --fall logs/stageD_fall/fall_best.zip \
  --npz data/motions_retargeted/kw_long_stance.npz \
  --preset relentless --out logs/controlled_demo/relentless
```

`--push T ANGLE_DEG MAGNITUDE_N DURATION_S` may be repeated; `--preset
{single,double,relentless,growing}` gives ready-made patterns. Nothing here
is scripted after the fact — what the video shows is whatever the loaded
policy (and switch) actually does under that exact push.

## Required diagnostics before retargeting a new URDF

If you change the G1 URDF/model, re-run and re-paste the diagnostics in:
- `src/dynamics/pinocchio_wrapper.py`
- `scripts/annotate_motion_library.py`

This locks in `model.nq`/`model.nv`/joint names, left/right foot frame names,
and the NPZ/CSV schema everything downstream assumes.

## Git hygiene

`data/`, `assets/`, `cloud_outputs/`, checkpoints' companion media (`*.mp4`,
`*.npz`, `*.pt`), and the bundled GEM-X/unitree_rl_mjlab subprojects are
gitignored — see `.gitignore`. `logs/*.zip` checkpoints and `*.json` results
are small enough to commit and are the evidence behind this README; keep
committing them as experiments finish.

## License

Follow the licenses of bundled subprojects (GEM-X, unitree_rl_mjlab) for any
redistribution or deployment.
