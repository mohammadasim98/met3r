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

---

## Third review remediation: bounded generations and exact all-track inputs

The third Task 8 review identified three important gaps in commit `48b3b22`:
published immutable generations accumulated without a retention bound or
matching storage projection, non-object feature-variant list entries were
silently filtered, and general aggregation accepted registry-authorized
Parquet files from arbitrary directories below `metrics/parts/`.

The design was recorded and approved in
`docs/superpowers/specs/2026-07-24-task8-third-review-remediation-design.md`.
Repository inspection confirmed that Task 7's `_part_path` uses the payload
feature-track directory for both controlled and stress jobs; only
geometry-calibration uses a fixed directory. The remediation therefore
validates that actual writer contract rather than introducing a Task 7 path
change.

### Third review RED evidence

Strict feature-variant entries:

```text
4 failed, 1 passed, 33 deselected in 7.02s
```

Appending a string, null, or list and adding an unexpected object key all
passed the old gate. The wrong-key object already failed through `KeyError`,
which established the expected failure classification.

Exact auxiliary part paths and payload routing:

```text
3 failed, 38 deselected in 3.82s
```

The old general aggregator accepted a stress part moved from its feature-track
directory to `real_deformation_stress`, accepted a calibration part moved out
of `geometry_calibration`, and accepted a stress job labeled with controlled
analysis routing after all registered hashes were refreshed.

Generation retention and publisher serialization:

```text
2 failed, 41 deselected in 3.84s
```

Three distinct publications left three complete generation directories, and a
second publisher entered aggregate-frame construction while the first was
still inside the publication operation.

A final lock audit parametrized the competing publisher as either full-set or
single-track aggregation. The full-set case passed after the first lock fix,
while the legacy pre-generation single-track path still entered concurrently:

```text
1 failed, 1 passed, 45 deselected in 3.86s
```

Retention-aware storage projection:

```text
3 failed, 43 deselected in 4.79s
```

The report exposed only the old hardcoded `metrics × 2` category. It had no
separate metric-part and generation categories, no measured generation
count/bytes, and no exact two-generation multiplier.

### Third review implementation

- `feature_variants.json` is compared as a raw ordered JSON list with
  `[asdict(item) for item in build_upper_design(...).feature_variants]`.
  No entry is filtered or coercively reconstructed.
- General aggregation derives and requires Task 7's exact resolved part path
  for every completed job:
  - controlled and stress:
    `metrics/parts/<feature_track>/<job_id>.parquet`;
  - geometry calibration:
    `metrics/parts/geometry_calibration/<job_id>.parquet`.
- Job type, state track, and `analysis_track` must agree before a registered
  file enters a dataset. Sharing a feature-track directory cannot route stress
  rows into controlled aggregation or the reverse.
- `AGGREGATE_GENERATION_RETENTION` is fixed at two.
- A filesystem `flock` serializes publishers across the full cleanup, build,
  switch, alias, and retention operation.
- After the atomic pointer switch, cleanup preserves the current generation
  and the resolved pre-switch generation and removes other complete published
  generations. Staging and unpublished cleanup remains inside the publisher
  lock.
- Storage projection separates `metric_parts` and
  `aggregate_generations`. The latter reports actual complete-generation
  count and bytes, retention limit, projection basis, and projected generation
  count. It projects one full-design part copy plus exactly two full-design
  aggregate copies; a zero-generation pilot uses measured part bytes as its
  pre-aggregation proxy.

### Third review GREEN evidence

Incremental results:

```text
strict feature-variant entries: 5 passed, 33 deselected in 4.95s
exact auxiliary paths/routing plus duplicate-PK boundary:
  4 passed, 37 deselected in 4.27s
atomic visibility, crash, retention, and publisher lock:
  5 passed, 38 deselected in 6.38s
full-set and single-track publisher lock plus stable aliases:
  3 passed, 44 deselected in 4.66s
retention projection, cap, and PCA category split:
  4 passed, 42 deselected in 4.59s
```

The retained-generation test publishes three distinct content generations,
asserts that only current and immediate prior remain complete, verifies the
oldest was removed, measures their real directory bytes, and checks the
projection category reports the same count and byte total. The storage-cap
test also demonstrates that parts plus one generation remain below 300 GiB
while parts plus both retained generations exceed it and fail the gate.

Final Task 8 focused boundary:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_pilot_gate.py tests/test_upper_cli.py

54 passed in 37.98s
```

Cumulative Task 4–8 boundary:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_job_registry.py tests/test_upper_prepare.py \
  tests/test_upper_geometry.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_upper_metric_worker.py \
  tests/test_pipeline_integration.py tests/test_pilot_gate.py \
  tests/test_upper_cli.py

116 passed in 136.54s (0:02:16)
```

Fresh full-suite and static surface verification:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q
224 passed in 162.80s (0:02:42)

/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
exit 0

bash -n scripts/run_upper_pilot.sh
exit 0

met3r.upper_cli top-level help and all six command help invocations
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

---

## Review remediation: exact ownership and generation-atomic aggregation

The first Task 8 review rejected commit `217c68d` for one critical pilot-gate
ownership loophole and four important operational gaps.

### Root causes

- The gate checked 1,320 rows and coarse 44-by-30 cardinality without parsing
  the prepared state, pair-condition, or feature-variant manifests. A single
  feature job could therefore own every row, including invented design and
  variant keys.
- PCA storage was one category scaled by backbone count, even though fixed
  basis files and per-job render/manifests have different full-run units.
- Empty pilot tracks produced zero-column Parquet files.
- `aggregate_all_metric_parts` published each output independently, so a
  later-track or preview failure could expose a mixed aggregate generation.
- `status --require-complete` treated every `abstained` row as terminal even
  when its reason was empty or whitespace.

### Review RED evidence

The fixture was rebuilt to match the real Task 4/7 distribution: 32 image
states, 44 pair conditions, 30 exact variants, 64 completed feature jobs, one
Parquet part per `(state, feature_track)` owner, and exactly 1,320 rows.

Initial review-regression command:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q

9 failed, 9 passed in 7.83s
```

The nine expected failures demonstrated:

- a missing `feature_variants.json` still passed;
- invented design/variant keys lacked ownership failure;
- swapping exact-key rows across two job parts still passed;
- fixed basis and per-job PCA bytes were not separated;
- stress and geometry-calibration empty aggregates had no columns;
- a later calibration duplicate replaced the controlled aggregate before
  failing;
- empty and whitespace abstention reasons both made
  `status --require-complete` succeed.

The explicit preview boundary was also specified before its injectable writer:

```text
FAILED test_aggregate_all_preview_failure_preserves_entire_prior_generation
AttributeError: module 'met3r.pilot_gate' has no attribute
'_write_aggregate_preview'
1 failed in 3.01s
```

The exact Task 7 nullable-string boundary was specified separately:

```text
FAILED test_optional_normalization_handles_pandas_na
assert <NA> is None
1 failed in 2.30s
```

The final explicit-abstention audit also reproduced the same whitespace
loophole in the pilot gate itself:

```text
FAILED test_gate_rejects_whitespace_only_abstention_reason
assert 'job_completion' in ()
1 failed in 3.35s
```

### Review GREEN evidence

Incremental review fixes:

```text
exact manifest/ownership subset: 4 passed in 5.65s
storage category split: 1 passed in 3.01s
typed empty schemas: 2 passed in 3.43s
explicit abstention status: 2 passed in 2.52s
aggregate rollback/crash recovery: 2 passed in 4.13s
preview-stage rollback: 1 passed in 3.35s
Task 7 nullable string handling: 2 passed in 4.13s
pilot-gate whitespace abstention: 1 passed in 3.61s
```

Final review-hardening focused command:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q

23 passed in 13.84s
```

### Critical ownership correction

- Parses Task 4's actual `image_states.jsonl`,
  `pair_conditions.parquet`, and `feature_variants.json` formats.
- Derives the exact frozen pilot state coordinates, state/pair cells, ordered
  config-selected `FeatureVariant` records, and 1,320
  `(design_id, variant_key)` primary-key set.
- Requires 32 unique controlled states, 44 unique condition cells, and the
  exact 30 Pixal3D variants from the frozen config.
- Validates every condition's shoe, pair, upper views, changed view, state,
  corruption, severity, and track against its state manifest.
- Validates every row's condition and variant metadata against those
  manifests.
- Requires every completed part basename and row `job_id` to match its
  registry owner. The payload's state, feature track, backbone, analysis
  track, shoe, and config define the exact rows allowed in that part.
- Compares each part's exact key set with its owned
  `state × feature-track variants` set, then compares the global exact set.
  A job cannot own another state's row even when the global row count and keys
  remain unchanged.
- Normalizes Task 7 nullable Pandas string values, including `pd.NA`, before
  manifest comparison.

### Important corrections

- Storage projection now separates:
  - fixed `basis.npz`/`basis.json` bytes scaled from one pilot backbone to four
    full backbones;
  - per-job PCA renders/manifests scaled from 64 pilot feature jobs to 1,740
    full feature jobs.
- Empty controlled/stress aggregates retain typed string
  `design_id`/`variant_key` columns. Empty geometry calibration also retains
  typed string `changed_view`/`perturbation_kind` and float64 `magnitude`.
- All three aggregate Parquets and `summary_preview.csv` are staged and
  readback-validated before installation. One journal records the complete
  target generation.
- Existing outputs move to transaction-local backups and remain recoverable
  through the whole set installation. Any normal later-track, readback, or
  preview error restores all prior files byte-for-byte.
- A process death during installation leaves the journal/backups intact. The
  next aggregate invocation rolls the partial generation back before staging
  a new one. A committed journal keeps the complete new generation and only
  removes debris.
- `status --require-complete` now requires every abstained row to have a
  nonempty, non-whitespace reason.

### Review integration and full-suite evidence

Cumulative Task 4–8 boundary:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_job_registry.py tests/test_upper_prepare.py \
  tests/test_upper_geometry.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_upper_metric_worker.py \
  tests/test_pipeline_integration.py tests/test_pilot_gate.py \
  tests/test_upper_cli.py -q

85 passed in 113.92s (0:01:53)
```

Fresh full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q

193 passed in 140.91s (0:02:20)
```

The remediation remains limited to Task 8 production, tests, and this report.
No Task 9 analysis, full launcher, raw DINO feature persistence, bottom view,
or protected environment-config change was introduced.

---

## Second review remediation: canonical provenance and atomic visibility

The second Task 8 review rejected commit `a61286e` because the gate still
accepted noncanonical-but-self-consistent Task 3 identifiers and an
insufficiently pinned Task 7 basis manifest. It also found that the prior
multi-file transaction could expose mixed root files to readers during
installation.

### Second review RED evidence

The Task 3, Task 7, and exact part-path mutation suite initially produced:

```text
15 failed, 1 passed in 21.12s
```

The failures independently changed a state direction, seed, donor shoe,
bottom donor view, state ID, numeric design-ID assignment, state/condition
order, PCA artifact type, PCA seed, input-size order, fit-image count, sample
count, fit-image membership, and controlled part directory. Each mutation had
previously remained internally consistent enough to pass.

The immutable-generation tests were specified before implementation:

```text
2 failed in 3.48s
AttributeError: module 'met3r.pilot_gate' has no attribute
'resolve_aggregate_generation'
```

They model a reader resolving the aggregate during publication and a process
death before the pointer switch. A final single-track regression exposed one
remaining alias escape:

```text
FAILED test_single_track_aggregation_preserves_published_generation_aliases
assert resolve_aggregate_generation(run_root) != old_generation
1 failed in 3.24s
```

### Canonical Task 3 provenance

- Reconstructs the exact prepared four-shoe/four-view input expected by Task 3
  from `manifests/upper_manifest.json`.
- Calls Task 3's existing `build_upper_design(..., scope="pilot")` with the
  frozen config.
- Requires `feature_variants.json` to equal the canonical ordered
  `FeatureVariant` objects.
- Requires the ordered JSONL state objects to equal the canonical
  `ImageState` objects field-for-field, including IDs, direction, seed,
  donor shoe/view, corruption, severity, and provenance fields.
- Requires the pair-condition Parquet to have exactly the Task 3 columns and
  its ordered records to equal the canonical `PairCondition` objects. Numeric
  design IDs cannot be reassigned or reordered even when registry rows,
  metric rows, and sentinel metadata are rewritten consistently.

### Exact Task 7 PCA provenance

- Requires exactly the 16 canonical upper-view fit paths under the four
  expected shoe directories, with no missing or additional files.
- Recomputes every fit-image SHA-256 and requires the basis manifest to equal
  Task 7's complete schema: artifact type, backbone/revision, frozen config
  hash, seed 42, fit count 16, sample count 32, sorted combined Pixal input
  sizes `[512, 592]`, ordered path/hash records, and basis fingerprint.
- Continues to load the basis NPZ and compare its computed fingerprint with
  the manifest.

### Generation-atomic aggregate publication

- Writes and validates all three Parquets plus the preview in a private
  staging directory.
- Renames the completed staging directory to an immutable, content-addressed
  generation under `metrics/aggregate_generations/`.
- Publishes the entire set with one atomic replacement of the
  `metrics/current` symlink. Canonical multi-file readers call
  `resolve_aggregate_generation()` once and read all four files from that
  resolved directory.
- Maintains the documented root filenames as symlink aliases through
  `metrics/current`.
- Cleans unpublished crash debris while retaining published immutable
  generations so readers that resolved an old generation remain valid.
- Routes a single-track aggregate request through a complete new generation
  after publication, preventing that API from replacing one stable alias.
- Requires every registered feature part to be exactly
  `metrics/parts/<feature_track>/<job_id>.parquet`.

### Second review GREEN evidence

Incremental focused results:

```text
exact Task 3 provenance mutations: 8 passed in 8.20s
exact Task 7 PCA provenance mutations: 8 passed in 9.83s
exact feature-track part path: 1 passed in 3.66s
atomic visibility, crash, and failure injection: 4 passed in 5.24s
single-track alias regression plus pointer/crash checks: 3 passed in 4.38s
```

Final Task 8 module and CLI:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_pilot_gate.py tests/test_upper_cli.py

40 passed in 30.12s
```

Cumulative Task 4–8 boundary:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_job_registry.py tests/test_upper_prepare.py \
  tests/test_upper_geometry.py tests/test_pca_features.py \
  tests/test_streaming_features.py tests/test_upper_metric_worker.py \
  tests/test_pipeline_integration.py tests/test_pilot_gate.py \
  tests/test_upper_cli.py

102 passed in 120.71s (0:02:00)
```

Fresh full-suite and static surface verification:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q
210 passed in 153.83s (0:02:33)

/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
exit 0

bash -n scripts/run_upper_pilot.sh
exit 0

met3r.upper_cli top-level help and all six command help invocations
exit 0

git diff --check
exit 0
```
