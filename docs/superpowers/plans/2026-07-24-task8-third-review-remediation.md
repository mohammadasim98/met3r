# Task 8 Third-Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the third Task 8 review by bounding aggregate generations,
projecting their storage exactly, and enforcing canonical manifests and part
paths.

**Architecture:** Keep Task 7 as the writer authority. Task 8 validates its
payload-to-path contract before reading, publishes complete content-addressed
generations under a cross-process publisher lock, retains current plus prior,
and separates part and generation storage categories.

**Tech Stack:** Python 3.11, pathlib, fcntl, PyArrow/Pandas, pytest, SQLite job
registry.

## Global Constraints

- `AGGREGATE_GENERATION_RETENTION` is exactly `2`.
- Controlled and stress parts use `<payload.feature_track>`; calibration uses
  `geometry_calibration`.
- Root aggregate aliases remain symlinks through `metrics/current`.
- Do not change or stage `configs/environment_pixal3d.json`.
- Use test-first RED/GREEN cycles for every production behavior change.

---

### Task 1: Exact feature-variant manifest

**Files:**
- Modify: `tests/test_pilot_gate.py`
- Modify: `met3r/pilot_gate.py`

**Interfaces:**
- Consumes: Task 3 `build_upper_design(...).feature_variants`
- Produces: strict `_load_pilot_design()` list validation

- [x] **Step 1: Write the failing tests**

Add a parametrized test that appends each of `"extra"`, `None`, `[]`, an object
with a missing canonical key, an object with an extra key, and an object with
wrong canonical values. Require `validate_pilot()` to report
`design_provenance`.

- [x] **Step 2: Run tests and verify RED**

Run:
`/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_pilot_gate.py -k malformed_feature_variant`

Expected: non-object appended entries pass or error outside the gate instead of
returning the required failed check.

- [x] **Step 3: Implement exact raw-list validation**

Compare the raw JSON list directly with
`[asdict(item) for item in canonical.feature_variants]`; reject inequality
before constructing the canonical mapping.

- [x] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all parametrized cases pass.

### Task 2: Exact all-track registry part paths

**Files:**
- Modify: `tests/test_pilot_gate.py`
- Modify: `met3r/pilot_gate.py`

**Interfaces:**
- Consumes: `_RegistryRow.payload`, Task 7 `_part_path` contract
- Produces: `_expected_metric_part(root, row, track) -> Path`

- [x] **Step 1: Write failing stress and calibration tests**

Register one completed stress job and one completed calibration job with valid
Parquet output, move each to a wrong directory, refresh the registry output
hash, and require `aggregate_metric_parts()` to raise an exact-path error.
Also mutate `analysis_track` and require routing rejection.

- [x] **Step 2: Run tests and verify RED**

Run the new exact-path tests. Expected: aggregation accepts the relocated file.

- [x] **Step 3: Implement path and routing validation**

Derive expected job type, analysis track, and directory. Require exactly one
registered Parquet equal to the resolved expected path before adding it to the
dataset.

- [x] **Step 4: Run tests and verify GREEN**

Run the new tests plus existing controlled ownership tests. Expected: pass.

### Task 3: Bounded aggregate generations

**Files:**
- Modify: `tests/test_pilot_gate.py`
- Modify: `met3r/pilot_gate.py`

**Interfaces:**
- Produces: `AGGREGATE_GENERATION_RETENTION = 2`
- Produces: locked publisher and post-switch retention cleanup

- [x] **Step 1: Write failing retention and serialization tests**

Publish three distinct aggregates by mutating a registered part between
publishes. Capture each resolved snapshot. Assert only two published
directories remain, the current and immediate prior are complete, and the
oldest is removed. Use two processes or a lock probe to prove a second
publisher cannot enter the critical section before the first exits.

- [x] **Step 2: Run tests and verify RED**

Expected: three immutable directories accumulate and no publisher lock exists.

- [x] **Step 3: Implement lock and cleanup**

Use `fcntl.flock(LOCK_EX)` on `metrics/.aggregate-publisher.lock` around cleanup,
frame creation, staging, switch, alias repair, and retention cleanup. Preserve
the resolved pre-switch current and new current; delete other complete
generation directories only after successful publication.

- [x] **Step 4: Run tests and verify GREEN**

Run all atomic visibility, crash, retention, and serialization tests.

### Task 4: Retention-aware storage projection

**Files:**
- Modify: `tests/test_pilot_gate.py`
- Modify: `met3r/pilot_gate.py`

**Interfaces:**
- Produces: `metric_parts` and `aggregate_generations` projection categories

- [x] **Step 1: Write failing projection tests**

Inject known part bytes and generation bytes/count. Assert part projection is
one copy, aggregate projection is the per-generation full-row estimate times
two, and fields expose measured generation count/bytes and retention. Add a
gate test whose total crosses the configured cap only with the second retained
generation.

- [x] **Step 2: Run tests and verify RED**

Expected: the existing hardcoded `metrics * 2` category lacks the fields and
under-counts one aggregate copy.

- [x] **Step 3: Implement projection categories**

Measure only complete published generation directories. Use their mean bytes
per pilot row, or part bytes as the zero-generation proxy. Project
`FULL_DESIGN_COUNTS["metric_rows"]` for parts and the same row count times
retention for aggregate generations.

- [x] **Step 4: Run tests and verify GREEN**

Run storage projection and storage-limit gate tests.

### Task 5: Documentation, verification, and commit

**Files:**
- Modify: `.superpowers/sdd/task-8-report.md`
- Modify: `docs/superpowers/plans/2026-07-24-task8-third-review-remediation.md`

- [x] **Step 1: Append third-review RED/GREEN evidence**

Record exact failure counts, implementation contract, focused/cumulative/full
commands, and the Task 7 stress-directory decision.

- [x] **Step 2: Run focused verification**

Run `tests/test_pilot_gate.py tests/test_upper_cli.py`.

- [x] **Step 3: Run cumulative Task 4–8 verification**

Run the nine Task 4–8 test modules used in prior review rounds.

- [x] **Step 4: Run full and static verification**

Run the full suite, compileall, launcher syntax, all CLI help surfaces, and
`git diff --check`.

- [x] **Step 5: Commit and submit for review**

Stage only the report, plan, `met3r/pilot_gate.py`, and
`tests/test_pilot_gate.py`; commit separately from the design commit. Send the
cumulative baseline range and latest remediation range to the same reviewer.
