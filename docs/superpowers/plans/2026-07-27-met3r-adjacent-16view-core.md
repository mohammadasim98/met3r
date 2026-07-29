# MEt3R Adjacent 16-View Core Implementation Plan

> **For implementation:** Use the `superpowers:executing-plans` skill and
> complete the checkboxes in order. Do not begin the full feature run until
> both the geometry preflight and the timed runtime gate pass.

**Goal:** Implement and execute the approved adjacent-view MEt3R experiment,
then publish a validated `final_research_report.md` containing the five frozen
figure groups and a research interpretation for the View Reliability Module.

**Design source:**
`docs/superpowers/specs/2026-07-27-met3r-adjacent-16view-core-design.md`

**Experiment ID:** `20260727_adjacent16_core_v1`

**Run root:**
`/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1`

**Camera wording:** Every artifact and report must use
`fixed/precomputed DA3 cameras (not calibrated GT)`.

**Implementation strategy:** Keep the existing four-image execution unit, but
replace the cardinal-view identity with a dynamic `shoe × quartet` identity.
Every materialized state, camera artifact, DA3 job, feature batch, and clean
feature cache therefore contains exactly four adjacent views. The protocol
uses 16 such clean quartet units and never loads all 16 views of a shoe into
one feature batch. Existing `20260724_upper_v2` behavior remains supported and
its artifacts remain immutable.

**Frozen scientific totals:**

| Quantity | Exact count |
| --- | ---: |
| Shoes | 4 |
| Quartets per shoe | 4 |
| Clean quartet units | 16 |
| Pair units | 96 |
| Feature variants | 12 |
| Controlled states | 848 |
| Stress states | 16 |
| Geometry jobs | 32 |
| Feature jobs | 3,472 |
| Controlled rows | 31,104 |
| Geometry-calibration rows | 6,528 |
| Stress rows | 1,152 |
| All numeric metric rows | 38,784 |

The 3,472 feature jobs are:

- 3,392 controlled jobs: `848 states × 4 backbones`;
- 64 stress jobs: `16 states × 4 backbones`;
- 16 targeted DINOv3-L calibration jobs.

The 32 geometry jobs are the 16 clean controlled states plus the 16 adaptive
stress states. Controlled corruptions do not receive adaptive DA3 jobs.

---

## Task 1: Freeze and validate the adjacent-16 protocol contract

**Files:**

- Create: `configs/adjacent16_core.json`
- Create: `met3r/adjacent16_contract.py`
- Create: `tests/test_adjacent16_contract.py`

- [ ] **1.1 Write failing tests for the exact contract**

Define tests that load the new configuration through:

```python
contract = load_adjacent16_contract(CONFIG)
```

The tests must assert:

```python
assert contract.experiment_id == "20260727_adjacent16_core_v1"
assert contract.physical_gpus == (0, 1, 2, 3)
assert tuple(q.quartet_id for q in contract.quartets) == ("q0", "q1", "q2", "q3")
assert tuple(q.view_ids for q in contract.quartets) == (
    ("1", "2", "3", "4"),
    ("14", "15", "16", "17"),
    ("27", "28", "29", "30"),
    ("40", "41", "42", "0"),
)
assert len(contract.feature_variants) == 12
assert contract.expected_rows == {
    "controlled": 31_104,
    "geometry_calibration": 6_528,
    "stress": 1_152,
    "total": 38_784,
}
```

Also test rejection of:

- a sorted-numeric camera order;
- any extra feature variant;
- NAF at resolution 37;
- exposure or blur in the corruption map;
- any GPU set other than four unique non-negative indices;
- a camera label other than the frozen wording;
- a geometry or stress count inconsistent with the frozen formulas.

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_contract.py -q
```

Expected: tests fail because the module and config do not exist.

- [ ] **1.2 Add the exact JSON contract**

`configs/adjacent16_core.json` must contain:

- the exact experiment and output roots;
- the exact 48-view circular camera order from the approved design;
- `q0` through `q3` with their four view IDs;
- source paths based on `sample_result`;
- stress paths under
  `20260727_adjacent16_core_v1/inputs/stress`;
- stage size 504 and source size 1080;
- metric thresholds `0.03`, `0.08`, and `10.0`;
- physical GPUs `[0, 1, 2, 3]`;
- four controlled corruption maps with strengths
  `4 + 4 + 4 + 1`;
- the pinned four backbone revisions, NAF revision, and DA3 revision;
- an explicit 12-record `feature_variants` allowlist;
- an explicit two-key `geometry_calibration_variant_keys` allowlist;
- an explicit two-key `visualization_variant_keys` allowlist;
- the 17 frozen geometry perturbation magnitudes;
- the preflight, runtime, and analyzability thresholds.

The 12 allowlisted variants must be:

```text
dino_vitb16             bilinear          37
dinov2_vitb14           bilinear          37
dinov3_vitb16           bilinear          37
dinov3_vitl16_pixal3d   bilinear          37
dinov3_vitl16_pixal3d   bilinear          74
dinov3_vitl16_pixal3d   naf_normal_guide  74
dinov3_vitl16_pixal3d   bilinear          148
dinov3_vitl16_pixal3d   naf_normal_guide  148
dinov3_vitl16_pixal3d   bilinear          296
dinov3_vitl16_pixal3d   naf_normal_guide  296
dinov3_vitl16_pixal3d   bilinear          592
dinov3_vitl16_pixal3d   naf_normal_guide  592
```

All records use `user_37_grid_track`; the three ViT-B inputs and DINOv3-L
input are 592, while DINOv2's native 42 grid is explicitly resampled to 37.

- [ ] **1.3 Implement typed validation**

Add these public records and functions:

```python
@dataclass(frozen=True)
class QuartetSpec:
    quartet_id: str
    view_ids: tuple[str, str, str, str]

@dataclass(frozen=True)
class Adjacent16Contract:
    experiment_id: str
    output_root: Path
    reconstruction_root: Path
    stress_root: Path
    physical_gpus: tuple[int, int, int, int]
    camera_order: tuple[str, ...]
    quartets: tuple[QuartetSpec, ...]
    feature_variants: tuple[FeatureVariant, ...]
    geometry_calibration_variant_keys: tuple[str, str]
    visualization_variant_keys: tuple[str, str]
    expected_rows: Mapping[str, int]
    raw: Mapping[str, object]

def load_adjacent16_contract(path: str | Path) -> Adjacent16Contract:
    """Load and fail closed on any deviation from the frozen protocol."""
```

Reuse `FeatureVariant` from `met3r.upper_design`, but do not call the legacy
cartesian `design.feature_variants()` enumerator.

- [ ] **1.4 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_contract.py tests/test_design.py \
  tests/test_upper_design.py -q
git add configs/adjacent16_core.json \
  met3r/adjacent16_contract.py tests/test_adjacent16_contract.py
git commit -m "feat: freeze adjacent 16-view experiment contract"
```

Expected: new contract tests pass and the legacy 90-variant tests remain
unchanged.

---

## Task 2: Build the dense clean/stress input manifest

**Files:**

- Create: `met3r/adjacent16_data.py`
- Create: `tests/test_adjacent16_data.py`
- Create: `tests/helpers_adjacent16.py`

- [ ] **2.1 Write fixture-based manifest tests**

Add compact synthetic 48-view fixtures with:

- RGB files `masked_inputs/0.jpg` through `47.jpg`;
- masks `mask/1.jpg` through `48.jpg`;
- 48 intrinsics, extrinsics, and processed images;
- a known circular camera order;
- four selected quartets.

Tests must prove:

- RGB/camera index `i` maps only to mask filename `i + 1`;
- selected `view_id` is a string but array indexing uses its integer value;
- numeric filename sorting is never used as camera azimuth order;
- every selected source is RGB 1080-by-1080;
- staged clean RGB correlates with indexed DA3 processed RGB at least 0.99;
- all camera values are finite;
- each adjacent orbit edge lies within 5 to 10 degrees;
- all four shoes share the frozen order;
- missing or extra image/mask/camera entries block the manifest;
- a stress view without complete audit metadata blocks `require_stress=True`.

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_data.py -q
```

Expected: tests fail because the manifest builder does not exist.

- [ ] **2.2 Implement dense records and source discovery**

Add:

```python
@dataclass(frozen=True)
class AdjacentViewRecord:
    shoe_id: str
    camera_dataset_id: str
    quartet_id: str
    view_id: str
    orbit_position: int
    source_path: Path
    source_mask_path: Path
    stress_path: Path | None
    stress_mask_path: Path | None
    deformation_json_path: Path | None
    metrics_json_path: Path | None
    task_json_path: Path | None
    processed_image_correlation: float
    camera_azimuth_deg: float
    source_sha256: str
    source_mask_sha256: str
    stress_sha256: str | None
    stress_mask_sha256: str | None
    quality: DeformationQuality | None

@dataclass(frozen=True)
class AdjacentShoeRecord:
    shoe_id: str
    camera_dataset_id: str
    views: tuple[AdjacentViewRecord, ...]

@dataclass(frozen=True)
class Adjacent16Manifest:
    reconstruction_root: Path
    stress_root: Path
    shoes: tuple[AdjacentShoeRecord, ...]

def build_adjacent16_manifest(
    contract: Adjacent16Contract,
    *,
    require_stress: bool,
) -> Adjacent16Manifest:
    """Validate all 48 source records, then retain the frozen selected 16."""
```

For each display-name shoe, load arrays from:

```text
sample_result/{DISPLAY}/upper/glb_masked_white_bg/outputs/
```

The manifest must first validate all 48 inputs and only then select the 16
frozen views. Reuse the strict deformation JSON comparisons from
`upper_data.py`, but factor shared audit parsing into non-protocol-specific
helpers rather than duplicating permissive JSON logic.

- [ ] **2.3 Compute and persist camera-baseline metadata**

Implement:

```python
def camera_azimuths(extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Return circular camera-center azimuths in degrees."""

def pair_baseline(
    first: AdjacentViewRecord,
    second: AdjacentViewRecord,
) -> tuple[int, float]:
    """Return orbit-step distance and absolute angular separation."""
```

The derived camera order must equal the frozen 48-view sequence exactly.
Each quartet pair receives `baseline_steps` in `{1, 2, 3}` and its measured
`angular_separation_deg`; do not replace the measured angle with `7.5 × step`.

- [ ] **2.4 Run real read-only validation**

Before stress images exist, validate the clean source tree:

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_data \
  configs/adjacent16_core.json --clean-only
```

Expected JSON:

```json
{"all_source_views": 192, "selected_views": 64, "selected_pair_units": 96}
```

- [ ] **2.5 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_data.py tests/test_upper_data.py \
  tests/test_bytesize_data.py -q
git add met3r/adjacent16_data.py tests/test_adjacent16_data.py \
  tests/helpers_adjacent16.py met3r/upper_data.py
git commit -m "feat: validate dense adjacent-view input manifest"
```

---

## Task 3: Expand the exact quartet-aware experiment design

**Files:**

- Create: `met3r/adjacent16_design.py`
- Create: `tests/test_adjacent16_design.py`

- [ ] **3.1 Write exact enumeration tests**

Tests must assert:

```python
design = build_adjacent16_design(manifest, contract)

assert len(design.controlled_states) == 848
assert len(design.stress_states) == 16
assert len(design.controlled_conditions) == 2_592
assert len(design.stress_conditions) == 96
assert len(design.feature_variants) == 12
```

`2,592` above is the number of controlled pair conditions before feature
expansion. Also assert:

- 16 clean states, each with six pairs;
- 832 changed states;
- every changed state has exactly three attached pair conditions;
- every pair condition has `quartet_id`, `baseline_steps`, and measured angle;
- each quartet contributes three one-step, two two-step, and one three-step
  pair;
- identity donors use the same numeric view from the cyclic next shoe;
- transformation direction/seed depends on shoe, quartet, view, corruption,
  and strength, not on pair;
- pair IDs include quartet identity and cannot collide;
- controlled rows expand to 31,104;
- calibration rows expand to 6,528;
- stress rows expand to 1,152.

- [ ] **3.2 Implement protocol-specific immutable records**

Add:

```python
@dataclass(frozen=True)
class AdjacentImageState:
    state_id: str
    track: str
    shoe_id: str
    quartet_id: str
    view_ids: tuple[str, str, str, str]
    changed_view: str | None
    corruption: str
    severity: float
    direction: int
    seed: int
    donor_shoe_id: str | None = None
    donor_view: str | None = None

@dataclass(frozen=True)
class AdjacentPairCondition:
    design_id: str
    track: str
    shoe_id: str
    quartet_id: str
    pair_id: str
    first_view: str
    second_view: str
    baseline_steps: int
    angular_separation_deg: float
    changed_view: str | None
    image_state_id: str
    corruption: str
    severity: float
```

Use IDs that make quartet ownership visible, for example:

```text
controlled__air_huarache_run_ultra_111449__q0__clean
controlled__air_huarache_run_ultra_111449__q0__view_1__translation__0.200000__+1
stress__air_huarache_run_ultra_111449__q0__deformed
q0__1__2
```

- [ ] **3.3 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_design.py tests/test_upper_design.py -q
git add met3r/adjacent16_design.py tests/test_adjacent16_design.py
git commit -m "feat: enumerate quartet-aware adjacent experiment"
```

---

## Task 4: Extract feature-independent geometry support scoring

**Files:**

- Modify: `met3r/known_camera.py`
- Modify: `tests/test_known_camera.py`
- Create: `tests/test_adjacent16_preflight.py`

- [ ] **4.1 Add failing parity and support tests**

Add public result types:

```python
@dataclass(frozen=True)
class DirectionalGeometrySupport:
    valid_mask: Tensor
    overlap_fraction: Tensor
    abstained: Tensor
    projected_xy: Tensor
    relative_depth_error: Tensor
    cycle_error_px: Tensor

@dataclass(frozen=True)
class SymmetricGeometrySupport:
    first_to_second: DirectionalGeometrySupport
    second_to_first: DirectionalGeometrySupport
    abstained: Tensor
```

Test that:

- geometry support is identical to the validity fields returned by
  `score_symmetric_known_camera_pair`;
- changing feature tensors does not change support;
- overlap denominator is the directional target foreground;
- symmetric abstention is true if either direction is below 8%;
- cycle and relative-depth diagnostics are finite only on the valid mask.

- [ ] **4.2 Refactor projection support from feature comparison**

Implement:

```python
def score_symmetric_geometry_support(
    first_depth: Tensor,
    second_depth: Tensor,
    first_intrinsics: Tensor,
    second_intrinsics: Tensor,
    first_extrinsics_w2c: Tensor,
    second_extrinsics_w2c: Tensor,
    *,
    first_mask: Tensor | None = None,
    second_mask: Tensor | None = None,
    config: KnownCameraConfig | None = None,
) -> SymmetricGeometrySupport:
    """Measure reprojection support without constructing dummy features."""
```

Move the common projection, depth-consistency, cycle, and validity code into
one private helper used by both geometry support and feature scoring. Preserve
the exact legacy feature-score outputs.

- [ ] **4.3 Run regression tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_known_camera.py tests/test_multiscale_metric.py \
  tests/test_pipeline_integration.py tests/test_adjacent16_preflight.py -q
git add met3r/known_camera.py tests/test_known_camera.py \
  tests/test_adjacent16_preflight.py
git commit -m "refactor: expose feature-independent geometry support"
```

---

## Task 5: Prepare 64 independent stress deformations

**Files:**

- Create: `met3r/adjacent16_stress.py`
- Create: `tests/test_adjacent16_stress.py`

This task uses the existing ByteSize `last_view_match.render` and
`shoe_deform` implementations without modifying the dirty ByteSize source
tree. It records the exact ByteSize source fingerprint in every task record.

- [ ] **5.1 Write tests for camera conversion and atomic tasks**

Tests must cover:

- conversion of DA3 w2c extrinsics plus the saved
  `world_to_last_rotation` into a `CameraPose`;
- FOV recovery from the indexed 504 intrinsics;
- pose agreement with the four existing cardinal
  `precomputed_icp/*/camera.json` records;
- target bbox recovery from the one-based source mask;
- sibling staging and atomic promotion;
- one view per deformation subprocess;
- round-robin GPU assignment 0 through 3;
- no reuse when source, last-render, prompt version, or implementation hash
  changes;
- only `camera_preserved`, `background_preserved`, and crop/preservation gate
  determine input acceptance;
- rejected attempts remain archived and are never selected by MEt3R score.

- [ ] **5.2 Implement adjacent-view last-reference rendering**

Add:

```python
@dataclass(frozen=True)
class StressTask:
    shoe_id: str
    quartet_id: str
    view_id: str
    source_path: Path
    mask_path: Path
    last_render_path: Path
    output_dir: Path
    physical_gpu: int

def render_stress_references(
    contract: Adjacent16Contract,
    manifest: Adjacent16Manifest,
) -> tuple[Path, ...]:
    """Render the last in each selected fixed/precomputed DA3 view."""

def run_stress_tasks(
    tasks: Sequence[StressTask],
    *,
    max_parallel: int = 4,
    max_outer_attempts: int = 3,
) -> Mapping[str, object]:
    """Run independent one-view shoe_deform tasks with atomic promotion."""
```

Use each shoe's existing:

```text
six_view_experiment_result_precomputed/multiview/{shoe_id}/
  precomputed_icp/multiview_camera.json
```

only for the previously computed DA3-world-to-last rotation and last mesh.
Use the selected view's own DA3 extrinsic/intrinsic and own source mask for
pose, FOV, and bbox. Validate the derived pose against existing cardinal
records before rendering any new view.

- [ ] **5.3 Implement accepted input layout and audit**

Each accepted view must end at:

```text
{RUN_ROOT}/inputs/stress/{shoe_id}/{quartet_id}/view_{view_id}/deformation/
  source.png
  last_render.png
  source_mask.png
  deformed.png
  deformed_mask.png
  deformation.json
  metrics.json
  task.json
```

`task.json` records source and output hashes, command, prompt version, outer
attempt, ByteSize repository state, camera label, and acceptance fields.
Never create a synthetic replacement for a rejected deformation.

- [ ] **5.4 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_stress.py tests/test_upper_data.py -q
git add met3r/adjacent16_stress.py tests/test_adjacent16_stress.py
git commit -m "feat: prepare independent adjacent-view stress inputs"
```

---

## Task 6: Materialize quartet states, cameras, and exact jobs

**Files:**

- Create: `met3r/adjacent16_prepare.py`
- Create: `met3r/adjacent16_materialize.py`
- Create: `tests/test_adjacent16_prepare.py`
- Create: `tests/test_adjacent16_materialize.py`

- [ ] **6.1 Write preparation and idempotency tests**

Tests must assert:

```python
prepared = prepare_adjacent16_run(CONFIG)

assert prepared.root == RUN_ROOT
assert prepared.registry.counts_by_phase() == {
    "feature": 3_472,
    "geometry": 32,
}
```

Also verify:

- exactly 848 controlled and 16 stress state records;
- exactly 2,688 pair-condition records;
- every state directory contains exactly its quartet's four views;
- every camera artifact contains the same four view IDs in the same order;
- 64 clean PCA-fit source paths are staged only as inputs for later in-memory
  visualization;
- controlled changed states have no geometry jobs;
- stress states have adaptive geometry jobs;
- all 16 clean geometry jobs request calibration artifact generation;
- calibration feature jobs contain only the two targeted `variant_keys`;
- feature payloads contain `geometry_contract`;
- prepare is byte-idempotent and preserves running/completed registry rows;
- any stress audit failure blocks preparation before job registration.

- [ ] **6.2 Implement quartet camera artifacts**

Write one artifact to:

```text
cameras/{shoe_id}/{quartet_id}/cameras.npz
```

It contains:

```text
extrinsics       (4, 4, 4)
intrinsics       (4, 3, 3)
mask             (4, 504, 504)
view_ids         quartet order
camera_groups    upper, upper, upper, upper
```

Write `metadata.json` and `images.json` beside it. Validate exact content on
resume before reuse.

- [ ] **6.3 Implement quartet-only state materialization**

Implement:

```python
def materialize_adjacent_state(
    state: AdjacentImageState,
    manifest: Adjacent16Manifest,
    run_root: Path,
    *,
    stage_size: int = 504,
) -> MaterializedState:
    """Atomically stage exactly one four-view quartet state."""
```

Use Lanczos for RGB, nearest-neighbor for masks, and white outside foreground.
For controlled states, change only `changed_view`. For stress states, use all
four independently deformed images without an additional corruption.

- [ ] **6.4 Register the exact job graph**

Implement:

```python
def register_adjacent16_jobs(
    registry: JobRegistry,
    run_root: Path,
    design: Adjacent16Design,
    config_sha256: str,
) -> None:
    """Register 32 geometry and 3,472 feature jobs exactly once."""
```

Payload rules:

- controlled jobs: `geometry_contract="fixed_base_only"`;
- stress jobs: `geometry_contract="base_plus_adaptive"`;
- calibration jobs: `variant_keys=(L37 bilinear, L592 NAF)`;
- all jobs: shoe, quartet, state, camera path, config hash, and output path;
- job IDs include analysis track, state, backbone, and feature track.

- [ ] **6.5 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_prepare.py tests/test_adjacent16_materialize.py \
  tests/test_upper_prepare.py tests/test_upper_materialize.py -q
git add met3r/adjacent16_prepare.py met3r/adjacent16_materialize.py \
  tests/test_adjacent16_prepare.py tests/test_adjacent16_materialize.py
git commit -m "feat: prepare resumable adjacent quartet jobs"
```

---

## Task 7: Generalize the DA3 geometry worker to dynamic quartets

**Files:**

- Modify: `met3r/upper_geometry.py`
- Modify: `tests/test_upper_geometry.py`
- Create: `tests/test_adjacent16_geometry.py`

- [ ] **7.1 Write dynamic-view regression tests**

Use a quartet `("40", "41", "42", "0")` to prove that:

- state `images.json`, not lexical sorting, determines image order;
- camera, state, depth metadata, and calibration records have the same order;
- depth/confidence/mask remain `(4, 504, 504)`;
- four views produce `4 × 17 = 68` calibration artifacts;
- old `("left", "back", "right", "front")` tests still pass;
- a camera/state order mismatch fails without retry.

- [ ] **7.2 Replace global view IDs with artifact-owned IDs**

Change internal signatures to carry the view order:

```python
@dataclass(frozen=True)
class _CameraSet:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    view_ids: tuple[str, str, str, str]

def _read_materialized_state(state_dir: Path) -> MaterializedState:
    """Read the exact ordered four-view images.json contract."""

def _load_cameras(
    path: Path,
    *,
    expected_view_ids: tuple[str, str, str, str],
) -> _CameraSet:
    """Validate four upper cameras against the state-owned order."""
```

Pass `camera.view_ids` into `run_fixed_camera_depth`, fixed-camera validation,
and calibration generation. Preserve the four-view shape constraint to keep
the existing VRAM envelope.

- [ ] **7.3 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_geometry.py tests/test_upper_geometry.py \
  tests/test_da3_depth.py -q
git add met3r/upper_geometry.py tests/test_upper_geometry.py \
  tests/test_adjacent16_geometry.py
git commit -m "refactor: run DA3 geometry on dynamic four-view quartets"
```

---

## Task 8: Generalize feature scoring and enforce the storage policy

**Files:**

- Modify: `met3r/upper_metric_worker.py`
- Modify: `met3r/pca_features.py`
- Modify: `tests/test_upper_metric_worker.py`
- Modify: `tests/test_pca_features.py`
- Create: `tests/test_adjacent16_metric_worker.py`

- [ ] **8.1 Write failing worker-contract tests**

Cover:

- `_StateRecord` carries `quartet_id` and ordered `view_ids`;
- clean state lookup key is `(shoe_id, quartet_id)`;
- clean feature cache key includes quartet;
- pair indexes resolve through the current state's view order;
- controlled scoring loads only clean/base geometry;
- controlled `s01 == s00`, `s11 == s10`, geometry effect is zero, and
  `geometry_contract == "fixed_base_only"`;
- stress scoring loads clean/base plus stress/adaptive geometry;
- calibration jobs score only payload `variant_keys`;
- DINOv3-L controlled jobs score nine variants;
- B-backbone controlled jobs score one variant;
- condition/provenance columns preserve quartet, baseline step, and angle;
- no raw feature tensor or per-job PCA artifact is written in adjacent mode;
- legacy upper worker behavior remains unchanged.

- [ ] **8.2 Make state and geometry view order dynamic**

Extend `_StateRecord` and replace every `UPPER_VIEWS.index()` use in runtime
scoring with:

```python
view_index = {view_id: index for index, view_id in enumerate(state.view_ids)}
```

Change:

```python
def _load_geometry(
    path: Path,
    device: torch.device,
    *,
    expected_view_ids: tuple[str, str, str, str],
    validate_metadata: bool,
) -> ViewGeometry:
```

The worker still requires exactly four upper views, but no longer requires
cardinal names.

- [ ] **8.3 Add fixed-base controlled scoring**

For `geometry_contract="fixed_base_only"`, pass the clean geometry as both
base and adaptive inputs to the existing common-support factorial scorer.
This preserves the established output schema while making:

```text
s01 = s00
s11 = s10
geometry_effect = 0
interaction = 0
```

Do not run DA3 on controlled corrupted states.

- [ ] **8.4 Add exact variant filtering and targeted calibration**

If a feature payload includes `variant_keys`, require a non-empty exact subset
of its backbone/track variants and preserve manifest ordering. Calibration
jobs must reject any key other than the frozen L37-bilinear and L592-NAF keys.

- [ ] **8.5 Disable worker-time PCA for the adjacent protocol**

Add a frozen config field `visualization.policy="posthoc_audit_only"`.
When active:

- do not call `_fit_or_load_pca`;
- do not persist PCA bases;
- write only an empty, hashed output manifest for the per-job visualization
  directories;
- keep features in GPU/CPU memory only for the current streaming job;
- retain `_assert_no_persistent_features`.

Allow `basis: PCABasis | None` internally so the legacy protocol can retain its
existing behavior.

- [ ] **8.6 Add exact-job execution for the timed sentinel**

Extend the registry and worker with:

```python
def claim_exact(
    self,
    worker_id: str,
    phase: str,
    job_id: str,
) -> ClaimedJob | None:
    """Transactionally claim one named pending job."""

def run_feature_worker(
    run_root: str | Path,
    backbone: str,
    device: str,
    *,
    job_ids: Sequence[str] | None = None,
) -> WorkerSummary:
    """Run all matching jobs, or exactly the named production jobs."""
```

An exact-job run must never claim a job outside the supplied list.

- [ ] **8.7 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_metric_worker.py tests/test_upper_metric_worker.py \
  tests/test_job_registry.py tests/test_pca_features.py \
  tests/test_streaming_features.py -q
git add met3r/upper_metric_worker.py met3r/pca_features.py \
  met3r/job_registry.py tests/test_adjacent16_metric_worker.py \
  tests/test_upper_metric_worker.py tests/test_job_registry.py \
  tests/test_pca_features.py
git commit -m "feat: score adjacent quartets with fixed controlled geometry"
```

---

## Task 9: Implement preflight, timed sentinel, and final hard gates

**Files:**

- Create: `met3r/adjacent16_gate.py`
- Extend: `tests/test_adjacent16_preflight.py`
- Create: `tests/test_adjacent16_gate.py`

- [ ] **9.1 Write geometry-preflight gate tests**

Define:

```python
@dataclass(frozen=True)
class GeometryPreflightReport:
    passed: bool
    pair_units: int
    symmetric_pass_rate: float
    median_min_directional_overlap: float
    per_shoe_pass_rate: Mapping[str, float]
    failed_checks: tuple[str, ...]

def validate_adjacent16_geometry_preflight(
    run_root: str | Path,
) -> GeometryPreflightReport:
    """Score and validate all 96 clean geometry pair units."""
```

The persisted table `reports/geometry_preflight.parquet` must have exactly 96
rows and include:

```text
shoe_id, quartet_id, pair_id, first_view, second_view,
baseline_steps, angular_separation_deg,
first_to_second_overlap, second_to_first_overlap,
minimum_directional_overlap, overlap_asymmetry,
common_valid_fraction, cycle_error_px_median,
relative_depth_error_median, symmetric_pass
```

Gate thresholds:

- at least 90% symmetric pass;
- median minimum directional overlap at least 20%;
- every shoe at least 80% pair coverage;
- zero input/camera/hash validation failures.

- [ ] **9.2 Write timed-sentinel projection tests**

Create deterministic production-job sampling:

- for each backbone: the first shoe/q0 clean job;
- for each backbone: the first shoe/q0 view-1 translation-0.20 job;
- for each backbone: the first shoe/q0 stress job;
- for DINOv3-L: the first shoe/q0 calibration job.

This is exactly 13 already-registered production jobs, not a separate score
design.

Add:

```python
@dataclass(frozen=True)
class RuntimeProjectionReport:
    passed: bool
    sentinel_job_ids: tuple[str, ...]
    projected_geometry_hours: float
    projected_feature_hours: float
    fixed_finalize_hours: float
    projected_total_hours: float
    failed_checks: tuple[str, ...]

def write_timed_sentinel_manifest(run_root: str | Path) -> Path:
    """Write exact per-backbone job-ID files for the 13 sentinel jobs."""

def validate_timed_runtime_projection(
    run_root: str | Path,
) -> RuntimeProjectionReport:
    """Project four-GPU wall time from completed sentinel durations."""
```

Compute elapsed job time from registry `started_at` and `finished_at`.
Classify feature jobs as `clean`, `changed`, `stress`, or `calibration`.
For each backbone/class, use the maximum measured sentinel duration. Project:

```text
geometry wall =
  latest geometry finish - earliest geometry start

feature wall =
  max over backbone-assigned GPUs of
    sum(full class count × sampled class seconds)

projected total =
  geometry wall + 1.25 × feature wall + 2 fixed finalize hours
```

Full class counts:

| Backbone group | Clean | Changed | Stress | Calibration |
| --- | ---: | ---: | ---: | ---: |
| Each B backbone | 16 | 832 | 16 | 0 |
| DINOv3-L | 16 | 832 | 16 | 16 |

The gate passes only at or below 24 hours.

- [ ] **9.3 Write final analyzability gate tests**

Add:

```python
@dataclass(frozen=True)
class Adjacent16FinalGateReport:
    passed: bool
    expected_rows: Mapping[str, int]
    actual_rows: Mapping[str, int]
    controlled_complete_series_coverage: float
    all_four_shoe_primary_cell_rate: float
    minimum_common_support_retention: float
    stress_symmetric_pass_rate: float
    raw_feature_artifacts: tuple[str, ...]
    unexplained_jobs: tuple[str, ...]
    failed_checks: tuple[str, ...]
```

Define complete controlled series as a
`shoe × quartet × pair × placement × corruption × variant` unit with its clean
baseline and every frozen non-zero strength finite. Identity swap requires its
single non-zero endpoint. Preserve abstentions as missing evidence.

Gate thresholds are 80%, 80%, 70%, and 70%, respectively, plus exact registry
and row counts and no raw feature artifacts.

- [ ] **9.4 Implement and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_preflight.py tests/test_adjacent16_gate.py \
  tests/test_pilot_gate.py -q
git add met3r/adjacent16_gate.py tests/test_adjacent16_preflight.py \
  tests/test_adjacent16_gate.py
git commit -m "feat: gate adjacent geometry runtime and analyzability"
```

---

## Task 10: Make aggregation ownership-aware for the new protocol

**Files:**

- Modify: `met3r/pilot_gate.py`
- Create: `tests/test_adjacent16_aggregation.py`

- [ ] **10.1 Write exact aggregation tests**

Tests must verify:

- controlled primary key remains `(design_id, variant_key)`;
- stress primary key remains `(design_id, variant_key)`;
- calibration primary key includes perturbed view, kind, and magnitude;
- quartet and baseline columns survive Parquet aggregation unchanged;
- every completed feature job owns exactly one part at its registered path;
- variant-filtered calibration jobs own only their two expected variants;
- duplicate, missing, unregistered, or hash-mutated parts block publication;
- repeated aggregation produces the same content-addressed generation hash;
- the legacy pilot aggregation tests still pass.

- [ ] **10.2 Add protocol-aware ownership validation**

Keep `aggregate_all_metric_parts()` as the atomic publisher. Factor generic
registry/part validation from the old pilot-specific validation, then add:

```python
def validate_adjacent16_metric_ownership(
    run_root: str | Path,
    design: Adjacent16Design,
) -> Mapping[str, int]:
    """Validate every registered condition/variant row against its owner."""
```

This function returns the three exact track counts and is called before the
final adjacent gate.

- [ ] **10.3 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_aggregation.py tests/test_pilot_gate.py -q
git add met3r/pilot_gate.py tests/test_adjacent16_aggregation.py
git commit -m "feat: validate adjacent metric part ownership"
```

---

## Task 11: Implement the focused scientific analysis and five figures

**Files:**

- Create: `met3r/adjacent16_analysis.py`
- Create: `tests/test_adjacent16_analysis.py`

- [ ] **11.1 Write endpoint and common-support tests**

Use synthetic four-shoe data with deliberate abstentions to test:

- identity-swap clean/corrupt AUROC;
- translation/TPS severity Spearman;
- translation/TPS pairwise rank accuracy;
- JPEG signed and absolute nuisance delta;
- exact four-backbone common-support intersection at bilinear 37;
- paired five-resolution bilinear comparisons;
- paired NAF-minus-bilinear comparisons at 74/148/296/592;
- one-, two-, and three-step baseline stratification;
- low-finite/high-abstained reliability rate;
- low/high finite disagreement;
- shoe-clustered bootstrap that resamples shoes only;
- four-shoe-valid counts beside every endpoint;
- common-support retention and complete-series coverage.

The common-support key is:

```text
shoe_id × quartet_id × pair_id × changed_view ×
corruption × severity
```

No endpoint may compare unrelated marginal means.

- [ ] **11.2 Implement analysis tables**

Add:

```python
@dataclass(frozen=True)
class Adjacent16AnalysisOutputs:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    report_path: Path

def analyze_adjacent16_run(
    run_root: str | Path,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> Adjacent16AnalysisOutputs:
    """Publish the frozen adjacent-view analysis atomically."""
```

Publish CSV and Parquet versions of:

```text
support_coverage
backbone_common37
resolution_bilinear
naf_vs_bilinear
geometry_low_high
stress_reliability
common_support
analyzability
```

Bootstrap intervals are descriptive because there are four shoe clusters.
Quartets and pairs are never resampled as independent subjects.

- [ ] **11.3 Implement exactly five figure groups**

Write:

```text
reports/figures/01_adjacent_support.png
reports/figures/02_common37_backbones.png
reports/figures/03_dinov3l_resolution_naf.png
reports/figures/04_geometry_calibration.png
reports/figures/05_real_deformation_reliability.png
```

Required panel contents:

1. symmetric coverage, directional overlap, and baseline-step support;
2. four common-37 backbone endpoints on common support;
3. bilinear resolution curves plus paired NAF-minus-bilinear differences;
4. L37-bilinear and L592-NAF geometry-response envelopes;
5. stress pass rates, low/high validity states, disagreement, and baseline
   step.

Plot NaN/abstention as coverage, not as zero score. Every panel displays
valid-shoe counts.

- [ ] **11.4 Write the final report**

Publish:

```text
reports/final_research_report.md
```

Its sections are:

1. frozen protocol and exact completed counts;
2. adjacent-view support/preflight result;
3. common-37 backbone fidelity;
4. DINOv3-L resolution fidelity;
5. NAF versus bilinear;
6. local geometry sensitivity;
7. real independent-deformation reliability boundary;
8. View Reliability Module implications;
9. limitations and non-claims;
10. artifact index and reproducibility hashes.

The interpretation must keep conditional feature fidelity separate from
geometry/visibility reliability and must not call the DA3 cameras GT.

- [ ] **11.5 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_analysis.py tests/test_upper_analysis.py \
  tests/test_analysis.py -q
git add met3r/adjacent16_analysis.py tests/test_adjacent16_analysis.py
git commit -m "feat: analyze adjacent-view fidelity and reliability"
```

---

## Task 12: Render only post-hoc audit PCA and diagnostics

**Files:**

- Create: `met3r/adjacent16_visualization.py`
- Create: `tests/test_adjacent16_visualization.py`

- [ ] **12.1 Write deterministic audit-selection tests**

Build an audit manifest from aggregate numeric metrics only. With
lexicographic `design_id` tie-breaking, select:

- minimum, median-nearest, and maximum clean overlap per baseline step;
- for each maximum-severity corruption, one representative jointly valid case
  and one low-valid/high-invalid case when it exists;
- otherwise the largest finite low/high disagreement for that corruption;
- for each baseline step in stress, the largest finite low/high disagreement
  and one low-valid/high-invalid case when it exists.

The audit set must be stable under input row shuffling.

- [ ] **12.2 Implement in-memory shared PCA rendering**

Add:

```python
def build_adjacent16_audit_manifest(run_root: str | Path) -> Path:
    """Select deterministic numeric audit cases after aggregation."""

def render_adjacent16_audit(
    run_root: str | Path,
    *,
    device: str,
) -> tuple[Path, ...]:
    """Replay selected L37/L592 cases and persist PNG diagnostics only."""
```

Fit one shared DINOv3-L PCA basis in memory using the 64 clean selected
images. Use deterministic token subsampling, sign canonicalization, and global
display bounds. Render only:

- DINOv3-L bilinear 37;
- DINOv3-L NAF-normal 592.

Persist:

```text
reports/audit_visualizations/audit_manifest.json
reports/audit_visualizations/pca/*.png
reports/audit_visualizations/directional/*.png
reports/audit_visualizations/pca_render_manifest.json
```

`pca_render_manifest.json` contains fit-image hashes, PCA fingerprint, global
bounds, model/NAF revisions, and output PNG hashes. It must not contain PCA
components or per-image features. Delete all tensors after each replay case.

- [ ] **12.3 Enforce no raw feature/basis persistence**

Tests recursively scan the run fixture and reject:

- `.pt` or `.pth`;
- `features.npy` or `features.npz`;
- an NPZ marked `artifact_type=dino_features`;
- persisted PCA component arrays or `basis.npz`.

Only PNG visualization products and JSON metadata are allowed for this
protocol.

- [ ] **12.4 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_visualization.py tests/test_pca_features.py \
  tests/test_visualization.py -q
git add met3r/adjacent16_visualization.py \
  tests/test_adjacent16_visualization.py
git commit -m "feat: render posthoc adjacent PCA audit only"
```

---

## Task 13: Add the adjacent CLI and gated launcher

**Files:**

- Create: `met3r/adjacent16_cli.py`
- Create: `scripts/run_adjacent16_core.sh`
- Modify: `setup.py`
- Create: `tests/test_adjacent16_cli.py`

- [ ] **13.1 Write CLI and launcher tests**

The CLI must expose:

```text
validate-inputs
prepare-stress
prepare
geometry-worker
geometry-preflight
sentinel-manifest
feature-worker
runtime-gate
aggregate
final-validate
analyze
visualize
status
```

Test that the launcher:

- defaults to physical GPUs `0,1,2,3`;
- assigns shoes to geometry GPUs and backbones to feature GPUs;
- supports phases `stress`, `geometry`, `sentinel`, `full`, and `finalize`;
- waits for all parallel children and propagates any failure;
- refuses `full` unless both gate JSON files say `passed=true`;
- resumes completed jobs rather than deleting the run;
- never touches `20260724_upper_v2`.

- [ ] **13.2 Implement the CLI**

Use the protocol-specific modules for preparation, validation, analysis, and
visualization. Reuse the generalized upper geometry/feature workers and
atomic aggregator.

`feature-worker` accepts:

```text
--backbone
--device
--job-id-file
```

When `--job-id-file` is absent, it processes all pending jobs for the
backbone. Add:

```text
met3r-adjacent16=met3r.adjacent16_cli:main
```

to `setup.py`.

- [ ] **13.3 Implement the phase launcher**

The launcher uses:

```bash
PYTHON=${PYTHON:-/opt/conda/envs/pixal3d/bin/python}
CONFIG=${CONFIG:-/root/dev/met3r-worktrees/known-camera-met3r/configs/adjacent16_core.json}
RUN_ROOT=${RUN_ROOT:-/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1,2,3}
```

Each phase writes logs under `RUN_ROOT/logs` and uses `set -euo pipefail`.
No launcher phase automatically revises thresholds or experiment scope.

- [ ] **13.4 Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_cli.py tests/test_upper_cli.py -q
bash -n scripts/run_adjacent16_core.sh
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli --help
git add met3r/adjacent16_cli.py scripts/run_adjacent16_core.sh \
  tests/test_adjacent16_cli.py setup.py
git commit -m "feat: orchestrate gated adjacent 16-view experiment"
```

---

## Task 14: Complete code-level verification before real compute

**Files:**

- Modify only files already listed if a verification failure requires a fix.

- [ ] **14.1 Run focused protocol tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_contract.py \
  tests/test_adjacent16_data.py \
  tests/test_adjacent16_design.py \
  tests/test_adjacent16_stress.py \
  tests/test_adjacent16_prepare.py \
  tests/test_adjacent16_materialize.py \
  tests/test_adjacent16_geometry.py \
  tests/test_adjacent16_metric_worker.py \
  tests/test_adjacent16_preflight.py \
  tests/test_adjacent16_gate.py \
  tests/test_adjacent16_aggregation.py \
  tests/test_adjacent16_analysis.py \
  tests/test_adjacent16_visualization.py \
  tests/test_adjacent16_cli.py -q
```

- [ ] **14.2 Run the full regression suite**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q
git diff --check
```

- [ ] **14.3 Perform plan/spec consistency checks**

Run read-only assertions that:

- config variants equal the approved 12 rows;
- quartets equal the approved four rows;
- registry dry-run counts equal 32 and 3,472;
- design row arithmetic equals 31,104, 6,528, 1,152, and 38,784;
- all report wording tests use the exact camera label;
- no code path writes raw adjacent DINO tensors.

- [ ] **14.4 Commit verification fixes**

```bash
git status --short
git diff --check
```

If verification required a source fix, rerun the affected task's focused
tests and commit only the exact file paths changed for that fix with:

```bash
git commit -m "test: verify adjacent 16-view experiment end to end"
```

Do not stage `configs/environment_pixal3d.json`. If no verification fix was
needed, do not create an empty commit.

---

## Task 15: Generate inputs, run geometry preflight, and pass the timed gate

This task creates experiment artifacts outside Git. It does not commit run
outputs.

- [ ] **15.1 Confirm hardware and free space**

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader
df -h /root/dev/ByteSize/Reconstruction
```

Proceed only when GPUs 0 through 3 are available for quartet-size batches and
the run root has sufficient space. Record both outputs in
`RUN_ROOT/reports/environment_preflight.txt`.

- [ ] **15.2 Render references and generate the 64 stress inputs**

```bash
cd /root/dev/met3r-worktrees/known-camera-met3r
bash scripts/run_adjacent16_core.sh stress
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  validate-inputs configs/adjacent16_core.json --require-stress
```

Expected:

- 64 accepted deformation tasks;
- 64 unique source hashes and 64 deformed hashes;
- every view has camera/background/crop preservation;
- each generation task used one source view and one matching last render;
- failed attempts remain archived but are not scheduled;
- deformation-generation elapsed time is reported separately from the metric
  pipeline.

- [ ] **15.3 Prepare and run the 32 geometry jobs**

```bash
bash scripts/run_adjacent16_core.sh geometry
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  status /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
```

Expected registry geometry state: 32 completed, zero pending/running/failed.

- [ ] **15.4 Run and inspect geometry preflight**

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  geometry-preflight \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
```

Inspect:

```text
reports/geometry_preflight.json
reports/geometry_preflight.parquet
```

Do not continue unless all frozen preflight checks pass.

- [ ] **15.5 Run the 13 timed production sentinel jobs**

```bash
bash scripts/run_adjacent16_core.sh sentinel
```

Inspect:

```text
reports/timed_sentinel_manifest.json
reports/runtime_projection.json
```

Do not continue unless `projected_total_hours <= 24` and all 13 jobs completed
with valid hashes. If it fails, stop and request a numbered, user-approved
protocol revision; do not shrink scope automatically.

---

## Task 16: Run all metrics, validate, visualize, and interpret

- [ ] **16.1 Resume the remaining production feature jobs**

```bash
cd /root/dev/met3r-worktrees/known-camera-met3r
bash scripts/run_adjacent16_core.sh full
```

The 13 completed sentinel jobs are reused. Monitor:

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  status /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
```

Expected final registry:

```text
geometry completed: 32
feature completed: 3,472
pending/running/failed/unexplained abstained: 0
```

- [ ] **16.2 Aggregate and run the final hard gate**

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  aggregate \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  final-validate \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
```

Confirm exact aggregate counts:

```text
controlled_fidelity.parquet       31,104
geometry_calibration.parquet       6,528
real_deformation_stress.parquet    1,152
total                             38,784
```

If an analyzability threshold fails, continue to the report only to document
the reliability boundary; do not impute, drop shoes, change quartets, or lower
the overlap threshold. Do not describe such a run as confirmatory.

- [ ] **16.3 Publish analysis and audit visualizations**

```bash
bash scripts/run_adjacent16_core.sh finalize
```

Expected:

- eight analysis tables in CSV and Parquet;
- exactly five core figure groups;
- audit PCA PNGs only for L37-bilinear and L592-NAF;
- selected directional diagnostics;
- no raw DINO feature or PCA-basis artifact;
- `reports/final_research_report.md`.

- [ ] **16.4 Verify deterministic republishing**

Record hashes, then aggregate/analyze a second time:

```bash
sha256sum \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/final_research_report.md \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/figures/*.png
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  aggregate \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  analyze \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
```

The aggregate generation and analysis artifact hashes must remain unchanged.

- [ ] **16.5 Perform the final research interpretation**

Read the final tables and figures together and explicitly answer:

1. Which common-37 backbone most reliably separates identity inconsistency,
   tracks translation/TPS severity, and ignores JPEG nuisance?
2. Is resolution fidelity monotonic for DINOv3-L, and how does it change by
   angular baseline?
3. At equal target grids, when does NAF improve or degrade fidelity relative
   to bilinear?
4. How large is local injected-geometry sensitivity at L37 versus L592?
5. How often is low-resolution semantic evidence finite while
   high-resolution pixel-aligned evidence is invalid or strongly discordant?
6. Which observable inputs should a learnable View Reliability Module use:
   overlap, overlap asymmetry, cycle error, baseline angle, low/high score
   disagreement, or feature-resolution identity?
7. Which findings are conditional on the fixed/precomputed DA3 cameras and
   which cannot be generalized from four shoes?

The final handoff links the report, five figures, gate reports, aggregate
tables, config, and source commit. It states the actual metric-pipeline and
deformation-generation elapsed times separately.

---

## Final completion checklist

- [ ] Approved quartets and 12 variants are unchanged.
- [ ] All 64 stress inputs passed the frozen camera/crop contract.
- [ ] Geometry preflight passed before the full feature run.
- [ ] Timed projection was at most 24 hours.
- [ ] Registry contains exactly 32 geometry and 3,472 feature jobs.
- [ ] Aggregates contain exactly 38,784 numeric condition rows.
- [ ] NaN abstentions were preserved.
- [ ] No raw DINO feature or PCA basis was persisted.
- [ ] Only L37-bilinear and L592-NAF PCA PNGs were saved.
- [ ] Exactly five core figure groups were published.
- [ ] `final_research_report.md` separates conditional fidelity from the
  reliability boundary.
- [ ] DA3 cameras are never described as calibrated GT.
- [ ] The prior `20260724_upper_v2` run is unchanged.
- [ ] The untracked user file `configs/environment_pixal3d.json` was neither
  edited nor staged.
