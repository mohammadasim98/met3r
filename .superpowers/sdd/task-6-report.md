# Task 6 Report: Fixed-Basis PCA and Streaming Features

## Scope and baseline

- Worktree: `/root/dev/met3r-worktrees/known-camera-met3r`
- Branch: `feature/known-camera-met3r`
- Required baseline: `ee59d8ec147c0b6ab4ebe912ea7f60409cd26168`
- Verified starting `HEAD`: `ee59d8ec147c0b6ab4ebe912ea7f60409cd26168`
- Pre-existing untracked `configs/environment_pixal3d.json` was not read,
  modified, staged, or committed.
- Baseline full suite:

  ```text
  /opt/conda/envs/pixal3d/bin/python -m pytest -q
  138 passed in 97.48s (0:01:37)
  ```

## TDD evidence

### RED

Tests were created before either new production module.

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pca_features.py tests/test_streaming_features.py -q

ERROR tests/test_pca_features.py
ModuleNotFoundError: No module named 'met3r.pca_features'
ERROR tests/test_streaming_features.py
ModuleNotFoundError: No module named 'met3r.streaming_features'
2 errors in 2.40s
```

The failure was the expected missing-feature failure, not a test typo or
unrelated regression.

### GREEN

After the minimal implementation:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pca_features.py tests/test_streaming_features.py -q
17 passed in 2.44s
```

The same focused tests with warnings promoted to errors:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pca_features.py tests/test_streaming_features.py -W error -q
17 passed in 2.49s
```

## Implementation

### `met3r/pca_features.py`

- Added immutable `PCABasis` records containing only mean, three components,
  stored 1st/99th-percentile bounds, and a SHA-256 fingerprint.
- Added compressed save/load with exact-field, fixed-seed, finite-value,
  shape, and fingerprint validation. No feature tensor field is persisted.
- Added deterministic all-batch token flattening and seed-42 subsampling capped
  at 200,000 tokens.
- Added centered `torch.pca_lowrank(..., q=3, center=False)` fitting and
  deterministic component-sign canonicalization.
- Added fixed-bound BCHW-to-uint8-RGB projection with no per-image
  normalization.
- Added PNG rendering retained only for native variants or a track's highest
  resolution.

### `met3r/streaming_features.py`

- Added ordered native, bilinear, and NAF feature iteration.
- NAF guides are built lazily at the first NAF variant and reused once for the
  batch.
- Only the current generated feature is held by the iterator; it is deleted
  when the caller advances or closes the generator.
- Normal iteration never calls `torch.cuda.empty_cache()`.
- A CUDA OOM gets one retry, with `empty_cache()` called only between the failed
  attempt and retry.
- Unknown feature modes and incompatible input batch contracts fail explicitly.

### `met3r/features.py`

- Added `MultiScaleFeatureExtractor.extract_native(images)`.
- Preserved the existing `extract(...)` API and behavior by delegating native
  extraction through the new method.

### Tests

- Deterministic capped PCA fitting and sign canonicalization.
- Shared/global display bounds and non-per-image projection.
- PCA save/load and tamper rejection.
- Native/highest-only PNG retention; 74/148/296 intermediate variants are
  skipped and native/592 variants are written.
- Ordered one-at-a-time streaming and previous-tensor release.
- Lazy-once guide construction.
- No normal-loop cache clearing and OOM-only retry/cache clearing.
- Existing extractor API compatibility and invalid-mode/input contracts.

## Verification evidence

Relevant feature and upper-run integration:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_features.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_upper_design.py \
  tests/test_upper_prepare.py tests/test_pipeline_integration.py -q
38 passed in 53.15s
```

This includes the existing pilot preparation assertion that all 16 clean upper
images are staged under `pca_fit_images/`, even for the two-shoe pilot.

Full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q
155 passed in 97.62s (0:01:37)
```

Additional checks:

```text
/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r
exit 0

find . -type f -name 'features.npz' -print
<no output>

git diff --check
exit 0
```

`ruff` was not available in the required Pixal3D environment
(`No module named ruff`); this is not a project-declared verification command.

## Self-review

- Confirmed projection always uses basis-level stored bounds.
- Confirmed sign canonicalization occurs before fit-token bounds are computed.
- Confirmed the fingerprint covers dtype, shape, and bytes for every numeric
  PCA parameter.
- Confirmed saved NPZ fields cannot contain raw features.
- Confirmed skipped PCA variants do not create parent directories or files.
- Confirmed generator cleanup runs in a `finally` block after every generated
  yield.
- Confirmed no `empty_cache()` exists in the normal inner loop.
- Confirmed no Parquet, registry, metric-worker, or geometry-calibration
  orchestration from Task 7 was implemented.

## Concerns and boundary notes

- Task 6 intentionally supplies the multi-batch PCA primitive. Task 7 owns
  loading the 16 staged clean images and its documented worker sequence will
  pass all clean native batches to that primitive. The same interface accepts
  both Pixal3D 592-input and 512-input native batches in one call, producing
  the required combined backbone basis.
- The focused tests use fake CPU backbones/NAF. Real-model GPU execution remains
  the pilot integration gate specified by the approved experiment design.
