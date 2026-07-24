# Task 8 Report: Resumable Pilot Gates, Aggregation, CLI, and GPU Launcher

## Scope and baseline

- Worktree: `/root/dev/met3r-worktrees/known-camera-met3r`
- Branch: `feature/known-camera-met3r`
- Required and verified baseline:
  `b8e1393ae671e29af547739b9043ed79bd88cec1`
- Protected pre-existing untracked
  `configs/environment_pixal3d.json` was not read, modified, staged, or
  committed.
- Task 9 analysis and the full-run launcher were not implemented.

## RED evidence

The pilot gate and CLI tests were written before either production module:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q

ERROR tests/test_pilot_gate.py
ModuleNotFoundError: No module named 'met3r.pilot_gate'
ERROR tests/test_upper_cli.py
ModuleNotFoundError: No module named 'met3r.upper_cli'
2 errors in 1.81s
```

An additional status/hash regression was added during hardening:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_cli.py::test_status_require_complete_revalidates_completed_output_hashes -q

FAILED test_status_require_complete_revalidates_completed_output_hashes
Failed: DID NOT RAISE <class 'SystemExit'>
1 failed in 2.59s
```

The failure demonstrated that a registry row still labeled `completed` could
make `status --require-complete` succeed after its output bytes were corrupted.

## GREEN evidence

The status hash regression passed after completed rows were revalidated through
`JobRegistry.validated_state` before counts were reported:

```text
1 passed in 2.56s
```

Final Task 8 focused command:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q

10 passed in 3.37s
```

Task 8 plus packaging:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py tests/test_packaging.py -q

12 passed in 3.96s
```

## Implementation

### Pilot gates and provenance

- Enforced exactly 32 pilot geometry jobs and 64 pilot feature jobs, with the
  exact two shoes, two feature tracks, one pinned Pixal3D DINOv3-L backbone,
  and no bottom view.
- Required every registered job to be either hash-validated `completed` or
  explicitly `abstained` with a reason.
- Recomputed registry payload hashes and compared every payload config hash
  with the frozen config. Completed outputs are revalidated by the Task 4
  registry and invalid output is reset to pending.
- Validated the frozen camera label, seed, 504 staging, metric thresholds,
  GPU policy, DA3 model/revision, backbone revision, NAF revision, fixed-camera
  provenance, upper view order, PCA fit-image hashes, and the one-basis
  fingerprint per backbone.
- Required exactly 1,320 controlled metric rows, 44 design IDs, 30 feature
  variants per design, unique `(design_id, variant_key)` primary keys, finite
  non-abstained scores/effects, and at least 0.90 clean directional coverage.
- Rejected `features.npz`, `features.npy`, PyTorch tensor files, and renamed
  NPZ artifacts declaring `artifact_type=dino_features`.
- Compared live git commit/dirty-tree provenance when those fields are present
  in the prepared run manifest.

### Sentinel rerun

- Selects one completed controlled feature job deterministically.
- Captures its original Parquet frame, then resets only that registry row to
  pending under `BEGIN IMMEDIATE`.
- Calls `run_feature_worker(run_root, backbone, "cuda:0")` by default. Tests
  inject the worker boundary but still claim, rewrite, hash, and complete the
  real registry job path; they do not compare a frame with itself.
- Revalidates the newly registered output hash, requires identical
  `(design_id, variant_key)` rows, and compares score/effect/overlap/cycle/depth
  metrics with `rtol=1e-5` and `atol=1e-6`.
- Atomically records `reports/sentinel_rerun.json`. A killed process leaves the
  ordinary pending/running registry state for the existing stale-job recovery
  path; a mismatch remains an explicit failed pilot gate.

### Aggregation and storage

- Reads only registry-declared Parquet parts and revalidates every completed
  part/output hash before aggregation.
- Uses PyArrow datasets, track-specific primary keys, duplicate rejection, and
  stable mergesort beginning with `design_id` and `variant_key`.
- Atomically writes and reads back:
  - `metrics/controlled_fidelity.parquet`
  - `metrics/geometry_calibration.parquet`
  - `metrics/real_deformation_stress.parquet`
  - `metrics/summary_preview.csv`
- A duplicate-primary-key failure-injection test confirms the prior aggregate
  remains byte-identical.
- Projects persistent storage by artifact category rather than one pilot-tree
  multiplier. The report records fixed metadata/PCA-fit input, states,
  geometry plus 272 full calibration artifacts, both metric part and aggregate
  copies for 174,096 full rows, per-backbone PCA, and visualization/job units.
- Atomically writes `reports/pilot_storage_projection.json` and
  `reports/pilot_gate.json`, and hard-stops above 300 GiB.

### CLI and launcher

- Added all six commands:
  `prepare`, `geometry-worker`, `feature-worker`, `aggregate`,
  `validate-pilot`, and `status`.
- `status --require-complete` returns nonzero for pending/failed/running jobs
  and for completed jobs whose declared output hashes no longer validate.
- Added `met3r-upper-experiment=met3r.upper_cli:main`.
- Added the exact upper-only pilot launcher using the required Pixal3D Python,
  frozen config, pilot run root, DA3 repository, two pilot shoes, physical GPU
  4 through `CUDA_VISIBLE_DEVICES=4`, and local `cuda:0`. The sentinel rerun is
  also masked to physical GPU 4.

## Integration evidence

Cumulative Task 4–8 boundary:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_job_registry.py tests/test_upper_prepare.py \
  tests/test_upper_geometry.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_upper_metric_worker.py \
  tests/test_pipeline_integration.py tests/test_pilot_gate.py \
  tests/test_upper_cli.py -q

71 passed in 98.26s (0:01:38)
```

CLI and launcher checks:

```text
bash -n scripts/run_upper_pilot.sh
exit 0

python -m met3r.upper_cli --help
lists exactly prepare, geometry-worker, feature-worker, aggregate,
validate-pilot, and status

all six `COMMAND --help` invocations
exit 0
```

## Full-suite evidence

Fresh final full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q

180 passed in 130.60s (0:02:10)
```

Additional checks:

```text
/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
exit 0

git diff --check
exit 0
```

## Self-review and concerns

- Pilot aggregation covers only the controlled track; the two absent pilot
  tracks are written as valid empty Parquet files so the output contract stays
  stable without inventing Task 9 analysis.
- The category projection intentionally accounts for the full design's 344
  states, 616 base/calibration geometry units, 174,096 metric rows with both
  parts and aggregate copies, 1,740 feature jobs, and four PCA backbones.
- No raw DINO feature tensor is persisted.
- No real DA3/DINO GPU inference was run in Task 8. The real physical-GPU-4
  pilot is explicitly Task 10.
- The protected `configs/environment_pixal3d.json` remains untouched and
  untracked.
