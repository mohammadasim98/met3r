# Task 7 Report: Streaming Upper-View Metrics to Parquet

## Scope and baseline

- Worktree: `/root/dev/met3r-worktrees/known-camera-met3r`
- Branch: `feature/known-camera-met3r`
- Required and verified baseline:
  `ae99cfbbab097114ad282554978331a48d85fa10`
- Protected pre-existing untracked
  `configs/environment_pixal3d.json` was not read, modified, staged, or
  committed.
- Fresh baseline suite:

  ```text
  /opt/conda/envs/pixal3d/bin/python -m pytest -q
  159 passed in 97.94s (0:01:37)
  ```

## TDD evidence

### RED

Directional factorial fields were specified first:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_results.py::test_flatten_factorial_result_contains_scores_effects_and_diagnostics -q

1 failed in 1.36s
```

The failure showed that all 16 explicit `s00`/`s10`/`s01`/`s11`
first-to-second and second-to-first score/overlap fields were absent.

The worker integration contract was then run before its production module
existed:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_metric_worker.py -q

ModuleNotFoundError: No module named 'met3r.upper_metric_worker'
1 error in 2.75s
```

### GREEN and root-cause correction

The first worker run produced six passing tests and one reproducible
calibration failure:

```text
ValueError: target_depth must match the target feature resolution
6 passed, 1 failed in 11.78s
```

Tracing the data flow showed that factorial scoring resized 504 geometry
through `score_multiscale_factorial`, while calibration passed 504 depth
directly to a 2x2/4x4 feature scorer. The correction applies the same
confidence-filtered shared-support and camera-resize boundary before
`score_geometry_sensitivity_pair`.

Focused calibration verification:

```text
1 passed in 15.98s
```

This covers two variants times four directed changed views times 17 non-zero
perturbations times the three pairs containing each view, for 408 rows.

Final Task 7 focused command:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_results.py tests/test_visualization.py \
  tests/test_upper_metric_worker.py tests/test_pipeline_integration.py -q

15 passed in 36.91s
```

## Implementation

- Extended `flatten_factorial_result` without changing its symmetric fields.
  Every factorial cell now retains both directional scores and overlap
  fractions; `s11` cycle-error and relative-depth diagnostics remain present.
- Added one-load-per-process backbone and NAF orchestration. Jobs are drained
  by feature track, with only the active shoe/track's clean native four-view
  tensor cached. Changed and generated variants are released between jobs.
- Fit or validated one fixed PCA basis from exactly all 16 staged upper-view
  images. Every assigned native input size contributes all 16 images, so the
  Pixal3D DINOv3-L 592 and 512 batches share one combined basis. The manifest
  hashes all fit images and records the basis fingerprint.
- Enforced canonical `left`, `back`, `right`, `front` views, RGB 504 staging,
  exact feature-variant keys/native grids, pinned model/NAF revisions, metric
  thresholds, config hash, fixed geometry provenance, and report label
  `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Scored every pair attached to each controlled or stress state with the full
  common-support `s00`/`s10`/`s01`/`s11` factorial. Stress rows retain
  semantic, preservation, camera, outsole, and fit-relation covariates for
  both views.
- Read the Task 5 aggregate calibration manifest, validated its config/hash,
  exact four-view order, exact configured magnitudes, unique identities, and
  exactly 17 artifacts per changed view. Each feature variant is reused while
  scoring every pair containing that changed view.
- Retained diagnostics for all pilot cells, clean cells, corruption-specific
  middle severity, maximum severity, and abstentions. Both directions are
  saved with condition ID, variant key, and direction in each filename.
  PCA PNG retention delegates to the native/highest-only Task 6 boundary.
- Added stable nullable stress-covariate Parquet types and per-job PCA and
  diagnostic manifests.
- Added a persistent-feature guard that rejects both `features.npz` and any
  renamed NPZ whose metadata declares `artifact_type=dino_features`.
- Transient retry is restricted to CUDA OOM and the Task 5 errno whitelist.
  OOM paths unwind job tensors, clear the active clean cache and traceback,
  collect references, and only then call `torch.cuda.empty_cache()`.

## Atomic output and resume evidence

Parquet parts are written as `{job_id}.parquet.tmp`, read back with PyArrow,
checked for exact schema and row count, renamed, and hashed. Job-specific
Parquet/PCA/diagnostic replacements retain prior outputs or an explicit
prior-absence marker until registry completion.

Failure-injection subset:

```text
3 passed in 10.86s
```

It verifies:

- registry completion failure restores byte-identical prior
  Parquet/PCA/diagnostic outputs;
- first-install completion failure exposes no final output or temporary,
  backup, or absence marker;
- a valid completed resume keeps committed outputs and removes only its exact
  backup/tmp/staging debris.

Additional focused tests verify PyArrow readback failure preserves the prior
part, corrupted completed output is reset and recomputed with a fresh attempt
budget, and both persistent-feature guard modes fail closed.

## Integration and full verification

Fresh cumulative Task 4-7 boundary command:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_job_registry.py tests/test_upper_prepare.py \
  tests/test_bytesize_data.py tests/test_da3_depth.py \
  tests/test_geometry_perturbations.py tests/test_upper_geometry.py \
  tests/test_features.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_results.py \
  tests/test_visualization.py tests/test_upper_metric_worker.py \
  tests/test_pipeline_integration.py -q

94 passed in 116.27s (0:01:56)
```

Fresh full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q

170 passed in 129.09s (0:02:09)
```

Additional checks:

```text
/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
exit 0

git diff --check
exit 0
```

Black and Ruff are not installed in the required Pixal3D environment; Black
reported `No module named black`.

## Self-review and concerns

- Confirmed Task 7 creates Parquet, fixed-basis PCA, diagnostic PNG, and small
  manifest artifacts only; it never writes a raw DINO tensor.
- Confirmed controlled, stress, and geometry-calibration tracks remain
  distinct, and no Task 8 CLI, pilot gate, aggregation, or shell orchestration
  was implemented.
- Confirmed output recovery is scoped to exact job destinations and never
  touches fresh running feature jobs.
- Confirmed normal feature iteration does not call
  `torch.cuda.empty_cache()`.
- Real model/GPU inference remains the planned pilot boundary; Task 7 tests use
  contract-faithful CPU fakes.
