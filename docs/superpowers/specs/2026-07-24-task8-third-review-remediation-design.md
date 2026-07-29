# Task 8 Third-Review Remediation Design

## Scope

This remediation closes three Task 8 review gaps without changing Task 7's
writer contract or adding Task 9 analysis:

1. bound immutable aggregate-generation retention and account for it in the
   storage gate;
2. require the feature-variant manifest to be the exact canonical JSON list;
3. require every aggregated Parquet part to occupy the exact path emitted by
   Task 7 for its job type.

The protected untracked `configs/environment_pixal3d.json` remains outside the
change.

## Aggregate publication and retention

`AGGREGATE_GENERATION_RETENTION` is fixed at `2`. A filesystem advisory lock
under `metrics/` serializes aggregate publishers across processes. While
holding the lock, a publisher:

1. removes only abandoned staging and unpublished generation directories;
2. builds and validates a complete private generation;
3. resolves the prior `metrics/current`, if present;
4. atomically switches `metrics/current` to the complete generation;
5. repairs the stable root aliases;
6. deletes published generation directories outside the retained set,
   preserving the current generation and its immediate predecessor.

Cleanup never removes `current`, a staging directory being published, or the
prior generation needed by a reader that resolved immediately before the
switch. A process death before the switch leaves the old pointer intact; a
death after the switch can temporarily leave extra generations, which the next
locked publication cleans safely.

## Storage projection

The projection separates `metric_parts` from `aggregate_generations`.

- Metric parts project one full-design copy from measured pilot part bytes.
- Aggregate generations expose measured generation bytes, measured generation
  count, and the retention limit. Their full-design projection uses the
  measured mean generation size when generations exist, or the measured part
  size as a conservative pre-aggregation proxy, and multiplies by exactly
  `AGGREGATE_GENERATION_RETENTION`.
- Stable aliases are symlinks and contribute no data-copy multiplier.

The report therefore projects one parts copy plus exactly two aggregate
copies. The existing persistent-storage cap applies to the resulting total.

## Exact manifest and part ownership

`feature_variants.json` must be a list whose every entry is a mapping with
exactly the five canonical dataclass keys and canonical JSON-compatible value
types. The ordered raw list must equal the ordered Task 3 canonical list before
objects are constructed. Non-object, malformed, missing-key, extra-key, null,
or reordered entries fail.

The aggregate reader derives one exact part path per completed metric job from
Task 7's current `_part_path` contract:

- controlled feature jobs:
  `metrics/parts/<payload.feature_track>/<job_id>.parquet`;
- stress jobs:
  `metrics/parts/<payload.feature_track>/<job_id>.parquet`;
- geometry-calibration jobs:
  `metrics/parts/geometry_calibration/<job_id>.parquet`.

Job type and `analysis_track` must agree with routing before the part is read,
so controlled and stress jobs sharing a feature-track directory cannot
cross-contaminate aggregate tracks.

## Tests

Tests follow RED/GREEN cycles:

- append string, null, list, missing-key, extra-key, and wrong-key objects to
  the feature-variant manifest and require design-provenance failure;
- move completed stress and geometry-calibration parts to wrong directories,
  refresh their registered hashes, and require aggregation rejection;
- publish more than two distinct generations and assert the directory count is
  bounded, current and immediate prior snapshots remain complete, and a reader
  sees only one resolved generation;
- assert publishers are serialized at the publication boundary;
- assert projection fields expose generation count/bytes/retention and compute
  one parts copy plus two generation copies;
- assert the storage gate rejects a total that exceeds the cap because of the
  retained generations.

Focused Task 8 tests, the cumulative Task 4–8 boundary, and the complete suite
must pass before the remediation commit is submitted for re-review.
