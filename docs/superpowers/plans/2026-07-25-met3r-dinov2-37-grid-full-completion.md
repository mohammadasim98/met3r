# MEt3R DINOv2 37-Grid and Full Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain bilinear `37×37` evaluation for DINOv2, exclude only invalid DINOv2 NAF-37 variants, then complete, verify, analyze, visualize, and scientifically interpret the corrected full experiment.

**Architecture:** Feature enumeration filters NAF targets below a backbone's native grid while preserving bilinear downsampling and native features. Streaming execution independently rejects any invalid below-native NAF manifest. A clean 90-variant full run executes on physical GPUs `0,1,2,3`; aggregation and analysis then produce immutable metric tables, statistical summaries, figures, PCA visualizations, and a research report.

**Tech Stack:** Python 3.11, pytest, PyTorch/CUDA, DINO/DINOv2/DINOv3, NAF/NATTEN, pandas/Parquet, NumPy, Matplotlib, Bash.

## Global Constraints

- Use only `left`, `back`, `right`, and `front`; never include `bottom`.
- Report cameras exactly as `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Keep DINOv2 native `42×42` and bilinear `37×37`.
- Exclude only DINOv2 NAF modes at `37×37`; DINOv2 NAF starts at `74×74`.
- Keep the other backbone tracks and the Pixal3D current track unchanged.
- Use physical GPUs `0,1,2,3` with manifest source `runtime_override`.
- Retain only PCA PNG visualizations; never persist raw DINO feature tensors.
- Preserve the passing pilot and all failed full-run artifacts; do not delete them.
- Do not stage, edit, or remove `configs/environment_pixal3d.json`.
- Do not stop after metric generation: complete analysis, visual review, interpretation, and final reporting.

---

### Task 1: Enumerate the Correct 90 Feature Variants

**Files:**
- Modify: `met3r/design.py`
- Modify: `tests/test_upper_design.py`

**Interfaces:**
- Consumes: `feature_variants(config)` and backbone `native_grid`.
- Produces: 90 full variants; DINOv2 native 42, bilinear 37+, and NAF 74+.

- [ ] **Step 1: Write failing exact-enumeration tests**

Change the full and public-variant expectations from 93 to 90. Add:

```python
def test_dinov2_retains_bilinear_37_but_excludes_naf_37(frozen_config):
    variants = [
        item
        for item in feature_variants(frozen_config)
        if item.backbone == "dinov2_vitb14"
    ]
    assert [(item.feature_mode, item.resolution) for item in variants] == [
        ("native", 42),
        ("bilinear", 37),
        ("bilinear", 74),
        ("bilinear", 148),
        ("bilinear", 296),
        ("bilinear", 592),
        ("naf_normal_guide", 74),
        ("naf_normal_guide", 148),
        ("naf_normal_guide", 296),
        ("naf_normal_guide", 592),
        ("naf_blurred_guide", 74),
        ("naf_blurred_guide", 148),
        ("naf_blurred_guide", 296),
        ("naf_blurred_guide", 592),
        ("naf_shuffled_guide", 74),
        ("naf_shuffled_guide", 148),
        ("naf_shuffled_guide", 296),
        ("naf_shuffled_guide", 592),
    ]
```

Keep the pilot assertion at exactly 30 variants.

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_upper_design.py
```

Expected: failures showing 93 variants and the three invalid DINOv2 NAF-37
records.

- [ ] **Step 3: Implement the minimal enumeration filter**

Inside the mode/target loop in `met3r.design.feature_variants`, after
validating `resolution`:

```python
if mode.startswith("naf_") and resolution < native_grid:
    continue
```

Do not remove bilinear 37 or any target equal to the native grid.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_upper_design.py tests/test_design.py
```

Expected: all selected tests pass and pilot remains at 30 variants.

- [ ] **Step 5: Commit**

```bash
git add met3r/design.py tests/test_upper_design.py
git commit -m "fix: exclude below-native naf variants"
```

---

### Task 2: Defensively Reject Invalid External NAF Manifests

**Files:**
- Modify: `met3r/streaming_features.py`
- Modify: `tests/test_streaming_features.py`

**Interfaces:**
- Consumes: `iter_feature_variants(native, images, variants, naf)`.
- Produces: a clear pre-NAF error for a target below the native spatial grid.

- [ ] **Step 1: Write the failing runtime-contract test**

```python
def test_naf_target_below_native_grid_is_rejected():
    variant = FeatureVariant("t", "b", 8, "naf_normal_guide", 1)
    with pytest.raises(
        ValueError,
        match="NAF target resolution must be at least the native grid",
    ):
        list(
            iter_feature_variants(
                torch.ones(4, 8, 2, 2),
                torch.ones(4, 3, 8, 8),
                (variant,),
                FakeNAF(),
            )
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/test_streaming_features.py::test_naf_target_below_native_grid_is_rejected
```

Expected: failure because `FakeNAF` currently accepts the downsample.

- [ ] **Step 3: Implement the defensive check**

Before building NAF guides:

```python
if variant.resolution < max(native.shape[-2:]):
    raise ValueError(
        "NAF target resolution must be at least the native grid"
    )
```

Apply this only to NAF modes, not bilinear.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q tests/test_streaming_features.py
```

Expected: all streaming tests pass.

- [ ] **Step 5: Commit**

```bash
git add met3r/streaming_features.py tests/test_streaming_features.py
git commit -m "fix: reject naf feature downsampling"
```

---

### Task 3: Synchronize Frozen Counts and Verify the Software

**Files:**
- Modify: `docs/superpowers/specs/2026-07-24-met3r-upper-dino-fidelity-design.md`
- Modify: `docs/superpowers/plans/2026-07-24-met3r-upper-dino-fidelity.md`
- Modify: `docs/superpowers/plans/2026-07-25-met3r-upper-pilot-v2.md`
- Modify: `docs/superpowers/plans/2026-07-25-met3r-full-gpu-runtime-override.md`

**Interfaces:**
- Consumes: the corrected 90-variant design.
- Produces: exact expected counts in every active runbook.

- [ ] **Step 1: Update exact counts and interpretation**

Replace full-run expectations with:

```text
feature variants: 90
controlled rows: 92,880
geometry-calibration rows: 73,440
stress rows: 2,160
total full metric rows: 168,480
```

Document that the common lowest resolution is bilinear 37 for DINOv2, its
native resolution is 42, and cross-backbone NAF comparisons start at 74.

- [ ] **Step 2: Verify documentation and complete software**

Run:

```bash
rg -n "95,976|75,888|2,232|93 feature variants|exact count 93" \
  docs/superpowers/specs/2026-07-24-met3r-upper-dino-fidelity-design.md \
  docs/superpowers/plans/2026-07-24-met3r-upper-dino-fidelity.md \
  docs/superpowers/plans/2026-07-25-met3r-upper-pilot-v2.md \
  docs/superpowers/plans/2026-07-25-met3r-full-gpu-runtime-override.md
/opt/conda/envs/pixal3d/bin/python -m pytest -q
bash -n scripts/run_upper_full.sh
git diff --check
```

Expected: the search returns no stale active expectations, all tests pass, the
launcher parses, and the diff is clean.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-24-met3r-upper-dino-fidelity-design.md \
  docs/superpowers/plans/2026-07-24-met3r-upper-dino-fidelity.md \
  docs/superpowers/plans/2026-07-25-met3r-upper-pilot-v2.md \
  docs/superpowers/plans/2026-07-25-met3r-full-gpu-runtime-override.md
git commit -m "docs: revise full metric counts"
```

---

### Task 4: Preserve the Invalid Run and Start a Clean Corrected Full Run

**Files:**
- Preserve outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full_failed_dinov2_naf37_20260725T1022Z`
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full`

**Interfaces:**
- Consumes: verified 90-variant code and GPU override.
- Produces: a clean immutable full run on GPUs 0–3.

- [ ] **Step 1: Confirm no experiment workers are alive**

Run:

```bash
ps -eo pid,cmd | rg "met3r.upper_cli (geometry-worker|feature-worker)"
```

Expected: no worker for the V2 full root.

- [ ] **Step 2: Recoverably preserve the invalid root**

Verify the destination is absent, then move the exact root:

```bash
mv \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full_failed_dinov2_naf37_20260725T1022Z
```

Do not delete the preserved root.

- [ ] **Step 3: Verify GPU and disk state**

Require sufficient memory on GPUs 0–3 and over 300 GiB free:

```bash
nvidia-smi -i 0,1,2,3 \
  --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
df -h /root/dev/ByteSize/Reconstruction
```

- [ ] **Step 4: Start the corrected run**

```bash
FULL_GPUS=0,1,2,3 bash scripts/run_upper_full.sh
```

Poll registry and worker logs at intervals shorter than 60 seconds. Diagnose
any failure before changing code or resuming.

---

### Task 5: Verify, Analyze, Visualize, and Interpret the Complete Experiment

**Files:**
- Consume:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full`
- Produce beneath the full root:
  `metrics/`, `tables/`, `figures/`, `pca/`, `reports/`

**Interfaces:**
- Consumes: all completed metric parts and fixed/precomputed DA3 geometry.
- Produces: deterministic aggregates, statistics, visualizations, report, and
  View Reliability Module implications.

- [ ] **Step 1: Require complete registry and exact metric counts**

Require geometry 344 completed and feature 1,740 completed, with zero failed
or unexplained jobs. Require:

```text
controlled_fidelity.parquet: 92,880 rows
geometry_calibration.parquet: 73,440 rows
real_deformation_stress.parquet: 2,160 rows
```

- [ ] **Step 2: Verify artifact integrity and storage policy**

Revalidate every completed output hash. Require no `features.npz`,
`features.npy`, `.pt`, `.pth`, or NPZ marked `dino_features`. Confirm only
retained native/lowest and highest PCA PNGs and sufficient disk headroom.

- [ ] **Step 3: Verify deterministic aggregation and analysis**

Hash all aggregate tables, figures, and report; rerun aggregate and analyze;
require identical metric/table content hashes. Record any rendering metadata
differences separately from numeric determinism.

- [ ] **Step 4: Inspect representative visualizations**

Open native/lowest and highest PCA PNGs for all four shoes and all backbones.
Inspect clean, translation, identity-swap, and real-deformed conditions.
Verify the fixed PCA basis per backbone and compare bilinear, NAF normal,
blurred-guide, and shuffled-guide outputs where defined.

- [ ] **Step 5: Analyze scientific effects**

Quantify and visualize:

- backbone discrimination: DINO vs DINOv2 vs DINOv3-B vs DINOv3-L;
- native/37, 74, 148, 296, 592 resolution behavior;
- bilinear vs NAF normal vs blurred/shuffled guide controls;
- controlled severity monotonicity and identity-swap separation;
- fixed-geometry, adaptive-geometry, feature, and interaction effects;
- geometry calibration sensitivity envelopes;
- overlap/abstention and pair/view reliability;
- real-deformation stress and human quality-gate strata;
- leave-one-shoe-out stability and bootstrap uncertainty.

- [ ] **Step 6: Produce the final research interpretation**

The final report must distinguish visibility failure from feature
inconsistency and DA3 geometry error. It must state which signals can support:

- per-view reliability weights;
- semantic-only low-resolution conditioning;
- high-resolution pixel-aligned conditioning;
- abstention or downweighting for unreliable views;
- supervision or initialization targets for a learnable View Reliability
  Module.

State limitations: four shoes, upper views only, fixed/precomputed DA3 cameras
instead of calibrated GT, proxy metric rather than generation quality, and
DINOv2 native 42 versus other native 37.

- [ ] **Step 7: Final verification and handoff**

Run the complete test suite and `git diff --check`. Report all artifact paths,
row counts, hashes, key figures, quantitative findings, interpretation, and
remaining research risks. Do not claim completion without fresh evidence.
