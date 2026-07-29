# MEt3R Upper Pilot V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid all-pair V1 pilot gate with the approved overlap-aware V2 protocol, run the complete pilot on physical GPU 1, and unblock the unchanged four-GPU full experiment only if every revised gate passes.

**Architecture:** The frozen configuration owns the V2 experiment identity, 8% overlap threshold, and pair roles. The pilot gate evaluates `right__front` primary coverage separately from `left__right` expected abstention, while the launcher exposes an overridable physical pilot GPU whose default is 1. Existing V1 artifacts remain untouched and V2 uses a new output root.

**Tech Stack:** Python 3.11, pytest, pandas/Parquet, Bash, PyTorch/CUDA, DA3, DINOv3, NAF.

## Global Constraints

- Report cameras exactly as `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Use only `left`, `back`, `right`, and `front`; never include `bottom`.
- Use physical GPU 1 for the V2 pilot and physical GPUs 4–7 for the full run.
- Keep relative depth tolerance `0.03`, minimum overlap fraction `0.08`, and confidence percentile `10.0`.
- Retain only PCA PNG visualizations; never persist raw DINO feature tensors.
- Preserve `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v1`.
- Do not stage, edit, or remove the unrelated `configs/environment_pixal3d.json`.

---

### Task 1: Freeze the V2 Configuration and Executable Launchers

**Files:**
- Modify: `configs/upper_dino_fidelity.json`
- Modify: `scripts/run_upper_pilot.sh`
- Modify: `scripts/run_upper_full.sh`
- Modify: `tests/test_packaging.py`
- Modify: `tests/test_upper_design.py`
- Modify: `tests/test_upper_cli.py`

**Interfaces:**
- Consumes: the approved V2 design document.
- Produces: experiment ID `20260724_upper_v2`, V2 pilot/full roots, pilot pairs `right__front` and `left__right`, metric overlap `0.08`, and `PILOT_GPU` defaulting to physical GPU 1.

- [ ] **Step 1: Write failing configuration and design tests**

Update the literal expectations in `tests/test_packaging.py` and
`tests/test_upper_design.py`:

```python
assert config["experiment_id"] == "20260724_upper_v2"
assert config["output_root"].endswith("/20260724_upper_v2")
assert config["metric"]["min_overlap_fraction"] == 0.08
assert config["pilot"]["pairs"] == ["right__front", "left__right"]
assert {row.pair_id for row in design.pair_conditions} == {
    "right__front",
    "left__right",
}
```

Replace the pilot launcher source-text test with a behavioral test that runs
the script against a fake Python executable, records
`CUDA_VISIBLE_DEVICES`, and asserts that all three GPU commands default to
physical GPU 1 while all paths contain `20260724_upper_v2`.

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_packaging.py::test_frozen_upper_configuration_binds_every_experiment_value \
  tests/test_upper_design.py::test_pilot_design_is_exact \
  tests/test_upper_cli.py -k pilot_launcher
```

Expected: failures showing the V1 identifier, old pair set, 10% overlap, and
GPU 4 behavior.

- [ ] **Step 3: Implement the minimal V2 configuration and launcher behavior**

Set the JSON values to:

```json
{
  "experiment_id": "20260724_upper_v2",
  "output_root": "/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2",
  "metric": {
    "relative_depth_tolerance": 0.03,
    "min_overlap_fraction": 0.08,
    "confidence_percentile": 10.0
  },
  "pilot": {
    "pairs": ["right__front", "left__right"]
  }
}
```

In `scripts/run_upper_pilot.sh`, use overridable defaults:

```bash
PYTHON=${PYTHON:-/opt/conda/envs/pixal3d/bin/python}
CONFIG=${CONFIG:-/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json}
RUN_ROOT=${RUN_ROOT:-/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot}
DA3_REPO=${DA3_REPO:-/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3}
PILOT_GPU=${PILOT_GPU:-1}
```

Pass `CUDA_VISIBLE_DEVICES="$PILOT_GPU"` to both geometry calls, the feature
worker, and sentinel validation. Change only the full launcher's run root to
`20260724_upper_v2/full`; retain `gpus=(4 5 6 7)`.

- [ ] **Step 4: Verify GREEN**

Run the focused command from Step 2 and `bash -n` on both launchers.
Expected: all selected tests pass and both scripts parse successfully.

- [ ] **Step 5: Commit**

```bash
git add configs/upper_dino_fidelity.json scripts/run_upper_pilot.sh \
  scripts/run_upper_full.sh tests/test_packaging.py tests/test_upper_design.py \
  tests/test_upper_cli.py
git commit -m "feat: freeze overlap-aware upper pilot v2"
```

---

### Task 2: Split Primary Coverage From Negative-Control Abstention

**Files:**
- Modify: `met3r/pilot_gate.py`
- Modify: `tests/test_pilot_gate.py`

**Interfaces:**
- Consumes: V2 pair roles and the configured 8% overlap.
- Produces: `clean_primary_directional_coverage`,
  `clean_primary_pair_coverage`, `clean_negative_control_abstention`, and the
  compatibility alias `clean_directional_coverage`.

- [ ] **Step 1: Write failing role-aware gate tests**

Make the valid fixture assign finite 0.50 scores and 0.20 overlap to clean
`right__front` rows. Assign `NaN` directional and symmetric scores, 0.01
overlap, and `abstained=True` to clean `left__right` rows.

Add tests with hand-derived expectations:

```python
def test_clean_gate_separates_primary_from_negative_control(valid_pilot_run):
    report = validate_pilot(valid_pilot_run)
    assert report.clean_primary_directional_coverage == 1.0
    assert report.clean_primary_pair_coverage == 1.0
    assert report.clean_negative_control_abstention == 1.0
    assert report.clean_directional_coverage == 1.0

def test_clean_gate_rejects_primary_abstention(valid_pilot_run):
    # Rewrite clean right__front rows to NaN/0.01 and refresh manifests.
    report = validate_pilot(valid_pilot_run)
    assert "clean_primary_directional_coverage" in report.failed_checks
    assert "clean_primary_pair_coverage" in report.failed_checks

def test_clean_gate_rejects_finite_negative_control(valid_pilot_run):
    # Rewrite clean left__right rows to finite scores/0.20 and refresh manifests.
    report = validate_pilot(valid_pilot_run)
    assert "clean_negative_control_abstention" in report.failed_checks
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_pilot_gate.py \
  -k "clean_gate or gate_passes_complete_operational_pilot"
```

Expected: failures because the report lacks the three role-aware fields and
the current helper mixes both pairs at a hard-coded 10%.

- [ ] **Step 3: Implement role-aware calculations**

Define frozen pair roles:

```python
EXPECTED_PILOT_PAIRS = ("right__front", "left__right")
PRIMARY_PILOT_PAIR = "right__front"
NEGATIVE_CONTROL_PILOT_PAIR = "left__right"
MIN_OVERLAP_FRACTION = 0.08
```

Filter clean rows by `pair_id`. Directional primary coverage counts finite
score and overlap at least `MIN_OVERLAP_FRACTION`. Pair coverage counts
finite symmetric `s11` with `abstained=False`. Negative-control abstention
counts a direction when its score is non-finite or overlap is below the
threshold. Missing role subsets return `0.0`.

Extend `PilotGateReport` with the three role-aware fields. Fail the
corresponding checks below 0.90, and write the primary directional value to
the legacy `clean_directional_coverage` field.

Update config provenance validation to require overlap `0.08`.

- [ ] **Step 4: Verify GREEN**

Run all `tests/test_pilot_gate.py`, followed by:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_upper_design.py tests/test_upper_prepare.py \
  tests/test_upper_metric_worker.py tests/test_pilot_gate.py
```

Expected: every selected test passes.

- [ ] **Step 5: Commit**

```bash
git add met3r/pilot_gate.py tests/test_pilot_gate.py
git commit -m "fix: gate upper pilot by overlap role"
```

---

### Task 3: Update the Frozen Runbook and Verify the Software

**Files:**
- Modify: `docs/superpowers/plans/2026-07-24-met3r-upper-dino-fidelity.md`
- Modify: `.superpowers/sdd/task-10-brief.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: a V2/GPU-1 Task 10 runbook and a V2 tag name.

- [ ] **Step 1: Update the runbook**

Replace V1 pilot/full roots with V2 roots, set the pilot GPU to 1, document
the three role-aware gate assertions, and change the tag to
`upper-dino-fidelity-v2-pilot-passed`. Record V1 as a retained failed
protocol with 1,320 rows and 0.22916666666666666 mixed-pair coverage.

- [ ] **Step 2: Verify documentation and full software**

Run:

```bash
rg -n "20260724_upper_v1|CUDA_VISIBLE_DEVICES=4|0\\.10" \
  .superpowers/sdd/task-10-brief.md
/opt/conda/envs/pixal3d/bin/python -m pytest -q
git diff --check
```

Expected: the first command returns no stale Task 10 matches, all tests pass,
and `git diff --check` exits 0.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-24-met3r-upper-dino-fidelity.md \
  .superpowers/sdd/task-10-brief.md .superpowers/sdd/progress.md
git commit -m "docs: revise upper experiment runbook for v2"
```

---

### Task 4: Run and Validate the Real GPU-1 Pilot

**Files:**
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot`

**Interfaces:**
- Consumes: verified V2 code and cached pinned model weights.
- Produces: 32 geometry jobs, 64 feature jobs, 1,320 metric rows, role-aware
  gate JSON, deterministic sentinel JSON, storage projection, and PCA PNGs.

- [ ] **Step 1: Verify GPU 1 and disk**

```bash
nvidia-smi -i 1 --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
df -h /root/dev/ByteSize/Reconstruction
```

Expected: GPU 1 is addressable and the target filesystem has more than
300 GiB free.

- [ ] **Step 2: Run the complete V2 pilot**

```bash
PILOT_GPU=1 bash scripts/run_upper_pilot.sh
```

Expected: exit 0. Poll the process/logs at intervals shorter than 60 seconds.

- [ ] **Step 3: Verify hard gates and retained artifacts**

Read `reports/pilot_gate.json` and assert:

```python
assert gate["passed"]
assert gate["actual_metric_rows"] == 1320
assert gate["clean_primary_directional_coverage"] >= 0.90
assert gate["clean_primary_pair_coverage"] >= 0.90
assert gate["clean_negative_control_abstention"] >= 0.90
assert not gate["raw_feature_artifacts"]
assert projection["projected_total_bytes"] <= 300 * 1024**3
```

Also verify that no `features.npz`, `.pt`, or raw feature tensor exists.

- [ ] **Step 4: Review representative PCA outputs**

Open native and highest-resolution clean/changed PNGs for both pilot shoes.
Verify a shared PCA fingerprint, common display bounds, and presence of
bilinear, NAF normal, blurred-guide, and shuffled-guide highest-resolution
panels.

- [ ] **Step 5: Tag only a passing pilot**

After fresh full tests, `git diff --check`, and a passing gate:

```bash
git tag -a upper-dino-fidelity-v2-pilot-passed \
  -m "Pilot passed for 20260724_upper_v2 on physical GPU 1"
```

Do not create the tag if any hard gate fails.

---

### Task 5: Run the Four-GPU Full Experiment and Produce the Report

**Files:**
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full`

**Interfaces:**
- Consumes: exact V2 pilot-passed tag.
- Produces: controlled fidelity, geometry calibration, real-deformation
  stress, PCA visualizations, statistical tables, figures, and final report.

- [ ] **Step 1: Confirm the exact pilot tag and gate**

Run `git describe --exact-match --tags HEAD` and verify V2 pilot gate
`passed=true`.

- [ ] **Step 2: Run the full experiment**

```bash
bash scripts/run_upper_full.sh
```

Expected: geometry and feature workers complete on physical GPUs 4–7,
aggregation succeeds, and analysis exits 0.

- [ ] **Step 3: Verify counts and deliverables**

Require 92,880 controlled rows, 73,440 geometry-calibration rows, 2,160
stress rows, zero failed jobs, zero invalid output hashes, and zero raw
feature artifacts. Rerun aggregate/analyze and require identical content
hashes.

- [ ] **Step 4: Review the final scientific report**

Confirm that the report separates visibility, feature, adaptive geometry,
interaction, and known perturbation evidence; compares backbone
discrimination rather than raw magnitudes; and states upper-only,
four-shoe, and non-GT-camera limitations.
