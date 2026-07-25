# MEt3R Full-Run GPU Runtime Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen upper-view full experiment on physical GPUs `0,1,2,3` while recording the actual operational mapping without changing the passing GPU-1 pilot or its scientific configuration.

**Architecture:** `prepare_run` accepts an optional four-GPU operational override, validates it independently from the frozen config, and writes the resolved mapping plus its source into the immutable run manifest. The CLI parses a comma-separated override, and the full launcher passes the same value to preparation and `CUDA_VISIBLE_DEVICES`, keeping manifest provenance and process placement synchronized.

**Tech Stack:** Python 3.11, argparse, Bash, pytest, CUDA, JSON provenance manifests.

## Global Constraints

- Do not change image states, views, pairs, perturbations, feature variants, metric thresholds, model revisions, seeds, or camera artifacts.
- Continue to report cameras as `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Do not alter or rerun the passing pilot at `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot`.
- Preserve the tag `upper-dino-fidelity-v2-pilot-passed`.
- Use physical GPUs `0,1,2,3` for the full experiment.
- Keep geometry and feature phases sequential and use four parallel workers only within each phase.
- Do not stage, edit, or remove `configs/environment_pixal3d.json`.

---

### Task 1: Bind an Operational GPU Override Into the Immutable Run Manifest

**Files:**
- Modify: `met3r/upper_prepare.py`
- Modify: `tests/test_upper_prepare.py`

**Interfaces:**
- Consumes: frozen config key `physical_gpus` and optional `prepare_run(..., physical_gpus: Sequence[int] | None = None)`.
- Produces: `_physical_gpu_policy(config, physical_gpus=None)` with resolved assignments and `source`, plus immutable-policy conflict detection.

- [ ] **Step 1: Write failing override and validation tests**

Add tests that call:

```python
prepared = prepare_run(
    CONFIG,
    scope="pilot",
    output_root=tmp_path,
    physical_gpus=(0, 1, 2, 3),
)
policy = json.loads(
    (prepared.root / "run_manifest.json").read_text()
)["physical_gpu_policy"]
assert policy["source"] == "runtime_override"
assert policy["physical_gpus"] == [0, 1, 2, 3]
assert list(policy["geometry_assignments"].values()) == [0, 1, 2, 3]
assert policy["feature_assignments"] == {
    "dino_vitb16": 0,
    "dinov2_vitb14": 1,
    "dinov3_vitb16": 2,
    "dinov3_vitl16_pixal3d": 3,
}
```

Parametrize invalid overrides:

```python
@pytest.mark.parametrize(
    "value",
    [
        (0, 1, 2),
        (0, 1, 2, 3, 4),
        (0, 1, 1, 3),
        (-1, 0, 1, 2),
        (0, 1, 2, "3"),
    ],
)
def test_prepare_rejects_invalid_physical_gpu_override(tmp_path, value):
    with pytest.raises(ValueError, match="physical_gpus"):
        prepare_run(
            CONFIG,
            scope="pilot",
            output_root=tmp_path,
            physical_gpus=value,
        )
```

Extend the existing default-policy test to require
`policy["source"] == "frozen_config"`. Add an immutability test that prepares
with `(0,1,2,3)`, then calls the same root with `(4,5,6,7)` and expects
`ValueError("existing run manifest is incompatible")`.

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_upper_prepare.py \
  -k "gpu_policy or physical_gpu_override or conflicting"
```

Expected: failure because `prepare_run` has no `physical_gpus` keyword and
the existing manifest has no `source`.

- [ ] **Step 3: Implement the minimal policy resolution**

Change the signature to:

```python
def prepare_run(
    config_path: str | Path,
    scope: Literal["pilot", "full"],
    *,
    output_root: str | Path | None = None,
    physical_gpus: Sequence[int] | None = None,
) -> PreparedRun:
```

Validate exactly four unique non-negative `int` values, explicitly rejecting
`bool`. Resolve `physical_gpus` from the override when present and otherwise
from `config["physical_gpus"]`. Build geometry assignments by zipping
`EXPECTED_SHOES` and feature assignments by zipping:

```python
(
    "dino_vitb16",
    "dinov2_vitb14",
    "dinov3_vitb16",
    "dinov3_vitl16_pixal3d",
)
```

Set `source` to `runtime_override` or `frozen_config`. Pass the resolved policy
to `_write_run_manifest`. When the manifest already exists, require its
`physical_gpu_policy` to equal the requested policy in addition to the
existing scope and config-hash checks.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_upper_prepare.py
```

Expected: all preparation tests pass.

- [ ] **Step 5: Commit**

```bash
git add met3r/upper_prepare.py tests/test_upper_prepare.py
git commit -m "feat: record full-run gpu override"
```

---

### Task 2: Route the Same Override Through the CLI and Full Launcher

**Files:**
- Modify: `met3r/upper_cli.py`
- Modify: `scripts/run_upper_full.sh`
- Modify: `tests/test_upper_cli.py`

**Interfaces:**
- Consumes: `prepare_run(..., physical_gpus=...)` from Task 1 and
  `FULL_GPUS=0,1,2,3`.
- Produces: `prepare --physical-gpus 0,1,2,3` and exact ordered worker
  placement on physical GPUs `0,1,2,3`.

- [ ] **Step 1: Write failing CLI and behavioral launcher tests**

Add a CLI forwarding test by monkeypatching `prepare_run` and invoking:

```python
main([
    "prepare",
    str(config),
    "--scope",
    "full",
    "--physical-gpus",
    "0,1,2,3",
])
```

Assert that the captured tuple is `(0, 1, 2, 3)`.

Replace the full launcher's source-only GPU assertion with a behavioral test.
Copy the script to `tmp_path`, replace its default Python executable with a
fake command that writes `CUDA_VISIBLE_DEVICES|arguments` to a log, and set
safe temporary `CONFIG`, `RUN_ROOT`, and `DA3_REPO` values. Create
`RUN_ROOT/logs` before executing the safe copy because worker stdout is
redirected there. Require:

```python
assert prepare_call.endswith(
    "prepare <config> --scope full --physical-gpus 0,1,2,3"
)
assert geometry_gpus == ["0", "1", "2", "3"]
assert feature_gpus == ["0", "1", "2", "3"]
```

Also require aggregate and analyze calls and no `bottom` token.

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_upper_cli.py \
  -k "prepare_forwards_physical_gpus or full_launcher"
```

Expected: CLI rejects `--physical-gpus` and the launcher still maps `4–7`.

- [ ] **Step 3: Implement CLI parsing and launcher mapping**

Add:

```python
def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "physical GPUs must be comma-separated integers"
        ) from error
```

Register `prepare.add_argument("--physical-gpus", type=_gpu_indices)` and pass
`args.physical_gpus` to `prepare_run`.

Make the full launcher paths overridable, then parse:

```bash
PYTHON=${PYTHON:-/opt/conda/envs/pixal3d/bin/python}
CONFIG=${CONFIG:-/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json}
RUN_ROOT=${RUN_ROOT:-/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full}
DA3_REPO=${DA3_REPO:-/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3}
FULL_GPUS=${FULL_GPUS:-0,1,2,3}
IFS=',' read -r -a gpus <<<"$FULL_GPUS"
```

Call prepare with `--physical-gpus "$FULL_GPUS"` and remove the old
`gpus=(4 5 6 7)` declaration. Continue using
`CUDA_VISIBLE_DEVICES="${gpus[$index]}"` and `--device cuda:0`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_upper_cli.py tests/test_upper_prepare.py
bash -n scripts/run_upper_full.sh
```

Expected: all selected tests pass and the launcher parses successfully.

- [ ] **Step 5: Commit**

```bash
git add met3r/upper_cli.py scripts/run_upper_full.sh \
  tests/test_upper_cli.py
git commit -m "feat: run full experiment on gpu override"
```

---

### Task 3: Verify Provenance and Run the Full Experiment on GPUs 0–3

**Files:**
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full`

**Interfaces:**
- Consumes: passing pilot tag and the tested GPU override path.
- Produces: the complete full experiment, aggregate tables, figures, and
  scientific report.

- [ ] **Step 1: Verify the implementation and pilot ancestry**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q
git diff --check
git merge-base --is-ancestor upper-dino-fidelity-v2-pilot-passed HEAD
git diff --name-only upper-dino-fidelity-v2-pilot-passed..HEAD
```

Expected: all tests pass; only the design/plan, prepare/CLI/launcher, and their
tests differ from the pilot tag; the unrelated environment config remains
untracked.

- [ ] **Step 2: Verify GPU and disk availability**

Run:

```bash
nvidia-smi -i 0,1,2,3 \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
df -h /root/dev/ByteSize/Reconstruction
```

Require enough free device memory for each assigned worker and more than
300 GiB free on the target filesystem.

- [ ] **Step 3: Start the full run**

Run:

```bash
FULL_GPUS=0,1,2,3 bash scripts/run_upper_full.sh
```

Poll the process and per-worker logs at intervals shorter than 60 seconds.
Do not stop or modify unrelated GPU workloads.

- [ ] **Step 4: Verify operational provenance**

Read `full/run_manifest.json` and require:

```python
assert policy["source"] == "runtime_override"
assert policy["physical_gpus"] == [0, 1, 2, 3]
assert list(policy["geometry_assignments"].values()) == [0, 1, 2, 3]
assert list(policy["feature_assignments"].values()) == [0, 1, 2, 3]
assert policy["phase_order"] == ["geometry", "feature"]
assert policy["allow_phase_overlap"] is False
```

- [ ] **Step 5: Verify experiment deliverables**

Run status with `--require-complete`. Require 92,880 controlled rows, 73,440
geometry-calibration rows, 2,160 stress rows, zero failed jobs, zero invalid
output hashes, and zero raw feature artifacts. Rerun aggregate and analysis
and require identical content hashes.

- [ ] **Step 6: Review and report**

Review representative PCA PNGs and the generated report. Report backbone
discrimination and resolution/upsampler effects separately from visibility,
fixed/precomputed DA3 camera error, and interaction effects. State the
upper-only, four-shoe, and non-GT-camera limitations.
