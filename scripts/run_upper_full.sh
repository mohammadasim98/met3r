#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/conda/envs/pixal3d/bin/python
CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full
DA3_REPO=/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3

"$PYTHON" -m met3r.upper_cli prepare "$CONFIG" --scope full

shoes=(
  air_huarache_run_ultra_111449
  air_max_90_nrg_dc6083_500
  air_max_90_surplus_cq7743_300
  air_tuned_max_cv6984_001
)
gpus=(4 5 6 7)
pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" -m met3r.upper_cli geometry-worker \
    "$RUN_ROOT" --shoe-id "${shoes[$index]}" --device cuda:0 \
    --da3-repository "$DA3_REPO" \
    >"$RUN_ROOT/logs/geometry_${shoes[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0

backbones=(
  dino_vitb16
  dinov2_vitb14
  dinov3_vitb16
  dinov3_vitl16_pixal3d
)
pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" -m met3r.upper_cli feature-worker \
    "$RUN_ROOT" --backbone "${backbones[$index]}" --device cuda:0 \
    >"$RUN_ROOT/logs/features_${backbones[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0

"$PYTHON" -m met3r.upper_cli aggregate "$RUN_ROOT"
"$PYTHON" -m met3r.upper_cli analyze "$RUN_ROOT"
