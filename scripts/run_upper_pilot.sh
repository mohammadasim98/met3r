#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/conda/envs/pixal3d/bin/python
CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v1/pilot
DA3_REPO=/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3

"$PYTHON" -m met3r.upper_cli prepare "$CONFIG" --scope pilot

for shoe in \
  air_huarache_run_ultra_111449 \
  air_max_90_surplus_cq7743_300
do
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m met3r.upper_cli geometry-worker \
    "$RUN_ROOT" --shoe-id "$shoe" --device cuda:0 --da3-repository "$DA3_REPO"
done

CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m met3r.upper_cli feature-worker \
  "$RUN_ROOT" --backbone dinov3_vitl16_pixal3d --device cuda:0

"$PYTHON" -m met3r.upper_cli aggregate "$RUN_ROOT"
CUDA_VISIBLE_DEVICES=4 "$PYTHON" -m met3r.upper_cli validate-pilot \
  "$RUN_ROOT" --rerun-sentinel
