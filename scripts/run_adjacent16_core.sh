#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/opt/conda/envs/pixal3d/bin/python}
CONFIG=${CONFIG:-/root/dev/met3r-worktrees/known-camera-met3r/configs/adjacent16_core.json}
RUN_ROOT=${RUN_ROOT:-/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1,2,3}
DA3_REPO=${DA3_REPO:-/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3}

phase=${1-}
case "$phase" in
  stress|geometry|sentinel|full|finalize) ;;
  *)
    echo "usage: $0 {stress|geometry|sentinel|full|finalize}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a gpus <<<"$PHYSICAL_GPUS"
if [[ "$PHYSICAL_GPUS" != "0,1,2,3" ]]; then
  echo "PHYSICAL_GPUS must be exactly 0,1,2,3" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"

shoes=(
  air_huarache_run_ultra_111449
  air_max_90_nrg_dc6083_500
  air_max_90_surplus_cq7743_300
  air_tuned_max_cv6984_001
)
backbones=(
  dino_vitb16
  dinov2_vitb14
  dinov3_vitb16
  dinov3_vitl16_pixal3d
)

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_geometry_workers() {
  local index
  local pids=()
  for index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" \
      -m met3r.adjacent16_cli geometry-worker "$RUN_ROOT" \
      --shoe-id "${shoes[$index]}" --device cuda:0 \
      --da3-repository "$DA3_REPO" \
      >"$RUN_ROOT/logs/geometry_geometry_${shoes[$index]}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_feature_workers() {
  local log_phase=$1
  local job_file_mode=$2
  local index
  local pids=()
  local job_args=()
  for index in 0 1 2 3; do
    job_args=()
    if [[ "$job_file_mode" == "sentinel" ]]; then
      job_args=(
        --job-id-file
        "$RUN_ROOT/manifests/timed_sentinel/${backbones[$index]}.json"
      )
    fi
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" \
      -m met3r.adjacent16_cli feature-worker "$RUN_ROOT" \
      --backbone "${backbones[$index]}" --device cuda:0 \
      "${job_args[@]}" \
      >"$RUN_ROOT/logs/${log_phase}_features_${backbones[$index]}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

require_passed_gate() {
  local path=$1
  "$PYTHON" -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); value=json.loads(p.read_text()); raise SystemExit(0 if value.get("passed") is True else 1)' \
    "$path"
}

case "$phase" in
  stress)
    "$PYTHON" -m met3r.adjacent16_cli prepare-stress "$CONFIG" \
      --run-root "$RUN_ROOT" --physical-gpus "$PHYSICAL_GPUS" \
      >"$RUN_ROOT/logs/stress_prepare.log" 2>&1
    ;;
  geometry)
    "$PYTHON" -m met3r.adjacent16_cli prepare "$CONFIG" \
      --output-root "$RUN_ROOT" \
      >"$RUN_ROOT/logs/geometry_prepare.log" 2>&1
    run_geometry_workers
    "$PYTHON" -m met3r.adjacent16_cli geometry-preflight "$RUN_ROOT" \
      >"$RUN_ROOT/logs/geometry_preflight.log" 2>&1
    ;;
  sentinel)
    "$PYTHON" -m met3r.adjacent16_cli sentinel-manifest "$RUN_ROOT" \
      >"$RUN_ROOT/logs/sentinel_manifest.log" 2>&1
    run_feature_workers sentinel sentinel
    "$PYTHON" -m met3r.adjacent16_cli runtime-gate "$RUN_ROOT" \
      >"$RUN_ROOT/logs/sentinel_runtime_gate.log" 2>&1
    ;;
  full)
    if ! require_passed_gate "$RUN_ROOT/reports/geometry_preflight.json"; then
      echo "full phase requires persisted geometry preflight passed=true" >&2
      exit 1
    fi
    if ! require_passed_gate "$RUN_ROOT/reports/runtime_projection.json"; then
      echo "full phase requires persisted runtime projection passed=true" >&2
      exit 1
    fi
    run_feature_workers full all
    ;;
  finalize)
    "$PYTHON" -m met3r.adjacent16_cli aggregate "$RUN_ROOT" \
      >"$RUN_ROOT/logs/finalize_aggregate.log" 2>&1
    "$PYTHON" -m met3r.adjacent16_cli final-validate "$RUN_ROOT" \
      >"$RUN_ROOT/logs/finalize_validate.log" 2>&1
    "$PYTHON" -m met3r.adjacent16_cli analyze "$RUN_ROOT" \
      >"$RUN_ROOT/logs/finalize_analyze.log" 2>&1
    CUDA_VISIBLE_DEVICES="${gpus[3]}" "$PYTHON" \
      -m met3r.adjacent16_cli visualize "$RUN_ROOT" --device cuda:0 \
      >"$RUN_ROOT/logs/finalize_visualize.log" 2>&1
    ;;
esac
