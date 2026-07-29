# MEt3R Upper-View DINO Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable upper-view experiment that measures DINO-family multi-view fidelity across resolution, separates feature and geometry effects, and validates the findings on the 16 real deformations without persisting raw DINO features.

**Architecture:** A deterministic upper-view manifest expands into reusable four-view image states and pair-condition rows. GPU geometry workers persist fixed-camera DA3 depth artifacts, while one long-lived feature worker per backbone streams one feature variant at a time into the known-camera factorial metric, fixed-basis PCA renderer, and atomic Parquet result parts. A SQLite registry owns resumability; pilot gates precede the four-GPU full run; a separate analysis layer produces scale-safe backbone comparisons, geometry calibration, stress-test summaries, and the final report.

**Tech Stack:** Python 3.11, PyTorch 2.8, torchvision, transformers, NAF/NATTEN, Depth Anything 3, NumPy, SciPy, Pillow, OpenCV, pandas, PyArrow, SQLite, matplotlib, pytest, Bash, NVIDIA GPU 1 for the pilot and GPUs 4–7 for the full run.

## Global Constraints

- Work in `/root/dev/met3r-worktrees/known-camera-met3r` on branch `feature/known-camera-met3r`.
- Use `/opt/conda/envs/pixal3d/bin/python` for tests and experiment processes.
- Use only the four upper views `left`, `back`, `right`, and `front`; exclude bottom and cross-group pairs.
- Label cameras exactly `fixed/precomputed DA3 cameras (not calibrated GT)` in generated reports.
- Stage all 1080x1080 source/deformed RGB images and masks at 504x504 before DA3 inference; never pair 1080x1080 images with 504x504 intrinsics.
- Use physical GPU 1 for the pilot. Use physical GPUs 4, 5, 6, and 7 for the full run.
- Pin the backbone, NAF, and DA3 revisions recorded in the approved design.
- Compute metrics at all 90 valid feature variants.
- Persist PCA PNGs only at native/lowest and highest resolutions; never persist raw DINO feature arrays.
- Use seed 42, a minimum overlap fraction of 0.08, relative depth tolerance 0.03, and confidence percentile 10.
- Keep controlled corruption, geometry calibration, and real-deformation stress results in separate analysis tracks.
- Do not interpret raw metric magnitudes across backbones; compare rank endpoints or robust clean-normalized values.
- Treat identity swap as a single full-foreground replacement condition, not a severity curve.
- Hard-stop before the full run if projected persistent storage exceeds 300 GB.
- The base design is `docs/superpowers/specs/2026-07-24-met3r-upper-dino-fidelity-design.md`; the approved pilot revision is `docs/superpowers/specs/2026-07-24-met3r-upper-pilot-v2-design.md`.

---

## File and Responsibility Map

| File | Responsibility |
| --- | --- |
| `met3r/upper_data.py` | Discover and validate the 16 source/deformed upper-view records and deformation-quality metadata. |
| `met3r/upper_design.py` | Enumerate reusable image states, 1,032 pair-condition cells, pilot subsets, and exact feature variants. |
| `met3r/upper_materialize.py` | Stage 504x504 clean, corrupted, and deformed four-view image sets with aligned masks. |
| `met3r/job_registry.py` | Transactional SQLite job registration, claiming, completion, retry, and resume validation. |
| `met3r/upper_geometry.py` | Run one-load-per-worker fixed-camera DA3 depth and geometry perturbation jobs. |
| `met3r/pca_features.py` | Fit/save/load fixed PCA bases and render comparable PCA PNGs. |
| `met3r/streaming_features.py` | Yield native, bilinear, and one NAF variant at a time without feature serialization. |
| `met3r/upper_metric_worker.py` | Stream factorial and geometry-calibration metric rows into atomic Parquet parts. |
| `met3r/pilot_gate.py` | Enforce operational pilot gates and project persistent storage. |
| `met3r/upper_analysis.py` | Compute endpoint tables, leave-one-shoe-out ranges, figures, stress strata, and report text. |
| `met3r/upper_cli.py` | Dedicated command-line surface for preparation, workers, aggregation, validation, and analysis. |
| `configs/upper_dino_fidelity.json` | Frozen experiment, pilot, path, GPU, corruption, and model configuration. |
| `scripts/run_upper_pilot.sh` | Exact single-GPU pilot sequence. |
| `scripts/run_upper_full.sh` | Exact four-GPU full geometry and backbone sharding sequence. |
| `tests/test_upper_*.py` | Unit and integration coverage for each new boundary. |

---

### Task 1: Verify and Commit the Existing Known-Camera Foundation

**Files:**
- Verify: `met3r/known_camera.py`
- Verify: `met3r/experiment.py`
- Verify: `met3r/features.py`
- Verify: `met3r/da3_depth.py`
- Verify: `met3r/runner.py`
- Verify: `met3r/analysis.py`
- Verify: `tests/`
- Commit existing scoped changes: `README.md`, `KNOWN_CAMERA_EXPERIMENT.md`, `configs/known_camera_bytesize.json`, `met3r/*.py`, `requirements-known-camera.txt`, `setup.py`, `tests/`

**Interfaces:**
- Consumes: the uncommitted known-camera implementation already present in the worktree.
- Produces: a clean, tested baseline commit containing `score_multiscale_factorial`, `run_fixed_camera_depth`, pinned feature loaders, geometry perturbations, and existing tests.

- [ ] **Step 1: Audit the dirty worktree without changing it**

Run:

```bash
cd /root/dev/met3r-worktrees/known-camera-met3r
git status --short
git diff --check
git diff --stat
```

Expected: only the known-camera files listed above are dirty/untracked; `git diff --check` exits 0.

- [ ] **Step 2: Run the existing focused test suite**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_known_camera.py \
  tests/test_multiscale_metric.py \
  tests/test_features.py \
  tests/test_da3_depth.py \
  tests/test_geometry.py \
  tests/test_geometry_perturbations.py \
  tests/test_corruptions.py \
  tests/test_design.py \
  tests/test_runner.py \
  tests/test_results.py \
  tests/test_pipeline_integration.py \
  tests/test_analysis.py \
  tests/test_visualization.py \
  tests/test_bytesize_data.py \
  tests/test_human_rating.py \
  tests/test_cli.py \
  tests/test_packaging.py \
  tests/test_provenance.py \
  tests/test_public_api.py -q
```

Expected: 61 tests pass.

- [ ] **Step 3: Commit only the verified known-camera foundation**

Run:

```bash
git add \
  README.md KNOWN_CAMERA_EXPERIMENT.md setup.py requirements-known-camera.txt \
  configs/known_camera_bytesize.json \
  met3r/__init__.py met3r/analysis.py met3r/bytesize_data.py met3r/cli.py \
  met3r/corruptions.py met3r/da3_depth.py met3r/design.py met3r/experiment.py \
  met3r/features.py met3r/geometry.py met3r/geometry_perturbations.py \
  met3r/human_rating.py met3r/identity.py met3r/known_camera.py \
  met3r/provenance.py met3r/results.py met3r/runner.py met3r/visualization.py \
  tests
git commit -m "feat: add known-camera DINO fidelity foundation"
```

Expected: the commit contains no experiment output or DINO feature tensor.

---

### Task 2: Add the Upper-View Data Contract and Frozen Configuration

**Files:**
- Create: `met3r/upper_data.py`
- Create: `configs/upper_dino_fidelity.json`
- Create: `tests/test_upper_data.py`
- Modify: `requirements-known-camera.txt`
- Modify: `setup.py`

**Interfaces:**
- Consumes: reconstruction root `Path` and the fixed four shoe/view IDs.
- Produces:
  - `build_upper_manifest(reconstruction_root: str | Path) -> UpperManifest`
  - `UpperManifest.write_json(path: str | Path) -> Path`
  - `DeformationQuality` fields used by the stress analysis.

- [ ] **Step 1: Write failing manifest and quality-metadata tests**

Add tests that use the real read-only dataset:

```python
from pathlib import Path

from met3r.upper_data import EXPECTED_SHOES, UPPER_VIEWS, build_upper_manifest

ROOT = Path("/root/dev/ByteSize/Reconstruction")


def test_real_upper_manifest_has_exactly_sixteen_views() -> None:
    manifest = build_upper_manifest(ROOT)
    assert tuple(shoe.shoe_id for shoe in manifest.shoes) == EXPECTED_SHOES
    assert all(tuple(view.view_id for view in shoe.views) == UPPER_VIEWS for shoe in manifest.shoes)
    assert sum(len(shoe.views) for shoe in manifest.shoes) == 16
    assert all(view.source_image.size == (1080, 1080) for shoe in manifest.shoes for view in shoe.views)
    assert min(view.processed_image_correlation for shoe in manifest.shoes for view in shoe.views) >= 0.99


def test_real_deformation_quality_matches_selected_attempt() -> None:
    manifest = build_upper_manifest(ROOT)
    for shoe in manifest.shoes:
        for view in shoe.views:
            quality = view.quality
            assert quality.status == "low_quality"
            assert quality.selected_attempt_id
            assert quality.fit_relation in {"improved", "equivalent", "worsened"}
            assert isinstance(quality.semantic_gate, bool)
            assert isinstance(quality.preservation_gate, bool)
            assert isinstance(quality.camera_preserved, bool)
            assert isinstance(quality.outsole_preserved, bool)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest tests/test_upper_data.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'met3r.upper_data'`.

- [ ] **Step 3: Implement strict upper-view discovery**

Create the following public records and parser in `met3r/upper_data.py`:

```python
EXPECTED_SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_nrg_dc6083_500",
    "air_max_90_surplus_cq7743_300",
    "air_tuned_max_cv6984_001",
)
UPPER_VIEWS = ("left", "back", "right", "front")
CAMERA_DATASET_BY_SHOE = {
    "air_huarache_run_ultra_111449": "AIR HUARACHE RUN ULTRA(111449)",
    "air_max_90_nrg_dc6083_500": "AIR MAX 90 NRG(DC6083-500)",
    "air_max_90_surplus_cq7743_300": "AIR MAX 90 SURPLUS(CQ7743-300)",
    "air_tuned_max_cv6984_001": "AIR TUNED MAX(CV6984-001)",
}


@dataclass(frozen=True)
class ImageInfo:
    size: tuple[int, int]
    mode: str
    sha256: str


@dataclass(frozen=True)
class DeformationQuality:
    status: str
    selected_attempt_id: str
    semantic_gate: bool
    preservation_gate: bool
    camera_preserved: bool
    outsole_preserved: bool
    fit_relation: str
    task_status: str


@dataclass(frozen=True)
class UpperViewRecord:
    shoe_id: str
    camera_dataset_id: str
    view_id: str
    source_path: Path
    source_mask_path: Path
    deformed_path: Path
    deformed_mask_path: Path
    deformation_json_path: Path
    metrics_json_path: Path
    task_json_path: Path
    source_image: ImageInfo
    deformed_image: ImageInfo
    processed_image_correlation: float
    quality: DeformationQuality


@dataclass(frozen=True)
class UpperShoeRecord:
    shoe_id: str
    views: tuple[UpperViewRecord, ...]


@dataclass(frozen=True)
class UpperManifest:
    reconstruction_root: Path
    shoes: tuple[UpperShoeRecord, ...]

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict_paths(self), indent=2),
            encoding="utf-8",
        )
        return destination


def _selected_attempt(payload: Mapping[str, object]) -> Mapping[str, object]:
    selected = str(payload["selected_attempt_id"])
    attempts = payload["attempts"]
    if not isinstance(attempts, list):
        raise ValueError("deformation attempts must be a list")
    matches = [
        item for item in attempts
        if isinstance(item, dict) and item.get("attempt_id") == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"selected attempt {selected!r} is not unique")
    return matches[0]


def _quality(deformation: Mapping[str, object], task: Mapping[str, object]) -> DeformationQuality:
    selected = _selected_attempt(deformation)
    semantic = selected["semantic"]
    preservation = selected["preservation"]
    comparison = selected["comparison"]
    if not all(isinstance(value, dict) for value in (semantic, preservation, comparison)):
        raise ValueError("selected deformation audit is malformed")
    return DeformationQuality(
        status=str(deformation["status"]),
        selected_attempt_id=str(deformation["selected_attempt_id"]),
        semantic_gate=bool(semantic["passes_gate"]),
        preservation_gate=bool(preservation["passes_gate"]),
        camera_preserved=bool(semantic["camera_preserved"]),
        outsole_preserved=bool(semantic["outsole_preserved"]),
        fit_relation=str(comparison["relation"]),
        task_status=str(task["status"]),
    )
```

`build_upper_manifest` must resolve
`six_view_experiment_result_precomputed/multiview/{shoe}/{view}/deformation`,
require the seven input files named in `UpperViewRecord`, verify RGB and mask
dimensions, reject unexpected/missing shoe IDs, and serialize only absolute
paths and scalar metadata. It must use `CAMERA_DATASET_BY_SHOE` to find the
camera record and call `validate_processed_image` on every deformation source
with `min_correlation=0.99`; never infer the camera directory from a slug.

- [ ] **Step 4: Add the frozen JSON configuration**

Create `configs/upper_dino_fidelity.json` with these exact top-level values:

```json
{
  "experiment_id": "20260724_upper_v2",
  "reconstruction_root": "/root/dev/ByteSize/Reconstruction",
  "output_root": "/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2",
  "da3_repository": "/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3",
  "camera_label": "fixed/precomputed DA3 cameras (not calibrated GT)",
  "views": ["left", "back", "right", "front"],
  "seed": 42,
  "stage_size": 504,
  "metric": {
    "relative_depth_tolerance": 0.03,
    "min_overlap_fraction": 0.08,
    "confidence_percentile": 10.0
  },
  "persistent_storage_limit_gb": 300,
  "physical_gpus": [4, 5, 6, 7],
  "corruptions": {
    "translation": [0.03, 0.06, 0.12, 0.20],
    "thin_plate_warp": [0.25, 0.50, 0.75, 1.00],
    "exposure": [0.15, 0.30, 0.50, 0.75],
    "blur": [0.20, 0.40, 0.70, 1.00],
    "jpeg": [0.25, 0.50, 0.75, 1.00],
    "identity_swap": [1.00]
  },
  "pilot": {
    "shoes": [
      "air_huarache_run_ultra_111449",
      "air_max_90_surplus_cq7743_300"
    ],
    "pairs": ["right__front", "left__right"],
    "corruptions": {
      "translation": [0.12, 0.20],
      "exposure": [0.50, 0.75],
      "identity_swap": [1.00]
    },
    "backbones": ["dinov3_vitl16_pixal3d"]
  }
}
```

Add these exact model, feature, and geometry sections to the same object:

```json
{
  "models": {
    "dino_vitb16": {
      "revision": "7c446df5b9f45747937fb0d72314eb9f7b66930a"
    },
    "dinov2_vitb14": {
      "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    },
    "dinov3_vitb16": {
      "revision": "5931719e67bbdb9737e363e781fb0c67687896bc"
    },
    "dinov3_vitl16_pixal3d": {
      "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881"
    },
    "naf_revision": "37f2dfc180f2de53d98bd601109c0da0dd6b0f43",
    "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
    "da3_revision": "8615eefb62f2db4f8d6ebaa59160086981672829"
  },
  "feature_tracks": {
    "user_37_grid_track": {
      "dino_vitb16": {"input_size": 592, "native_grid": 37},
      "dinov2_vitb14": {"input_size": 588, "native_grid": 42},
      "dinov3_vitb16": {"input_size": 592, "native_grid": 37},
      "dinov3_vitl16_pixal3d": {"input_size": 592, "native_grid": 37},
      "common_evaluation_grids": [37, 74, 148, 296, 592]
    },
    "current_pixal3d_code_track": {
      "dinov3_vitl16_pixal3d": {
        "input_size": 512,
        "native_grid": 32,
        "naf_targets": [128, 512]
      }
    }
  },
  "upsampling_modes": [
    "bilinear",
    "naf_normal_guide",
    "naf_blurred_guide",
    "naf_shuffled_guide"
  ],
  "geometry_calibration": {
    "rotation_degrees": [0.25, 0.5, 1.0, 2.0, 5.0],
    "relative_translation": [0.0025, 0.005, 0.01, 0.02],
    "depth_scale": [1.01, 1.02, 1.05, 1.1],
    "relative_depth_noise_std": [0.005, 0.01, 0.02, 0.05]
  }
}
```

- [ ] **Step 5: Add the Parquet dependency and run tests**

Add `pyarrow>=14` to `requirements-known-camera.txt` and the `known-camera`
extra in `setup.py`.

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest tests/test_upper_data.py tests/test_packaging.py -q
```

Expected: all tests pass and the real manifest reports four shoes/sixteen views.
The current real-data audit minimum is above 0.9998, comfortably exceeding the
registered 0.99 gate.

- [ ] **Step 6: Commit**

```bash
git add met3r/upper_data.py configs/upper_dino_fidelity.json \
  tests/test_upper_data.py requirements-known-camera.txt setup.py
git commit -m "feat: define upper-view experiment inputs"
```

---

### Task 3: Enumerate Controlled Conditions and Materialize 504x504 Image States

**Files:**
- Create: `met3r/upper_design.py`
- Create: `met3r/upper_materialize.py`
- Create: `tests/test_upper_design.py`
- Create: `tests/test_upper_materialize.py`
- Modify: `met3r/corruptions.py`
- Modify: `met3r/design.py`
- Modify: `tests/test_corruptions.py`

**Interfaces:**
- Consumes: `UpperManifest`, frozen configuration, and camera-artifact stage size.
- Produces:
  - `feature_variants(config) -> tuple[FeatureVariant, ...]`
  - `build_upper_design(manifest, config, scope: Literal["pilot", "full"]) -> UpperDesign`
  - `materialize_image_state(state, manifest, output_dir, stage_size=504) -> MaterializedState`
  - exactly 336 changed full states, 1,032 full pair cells, 44 pilot pair cells, and 90 full feature variants.

- [ ] **Step 1: Write failing count, balance, donor, and materialization tests**

```python
def test_full_design_counts_and_clean_dedup(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    assert len(design.changed_states) == 336
    assert len(design.pair_conditions) == 1032
    assert len(design.stress_states) == 4
    assert len(design.stress_pair_conditions) == 24
    assert len(design.feature_variants) == 93
    assert len({row.design_id for row in design.pair_conditions}) == 1032
    assert sum(row.corruption == "clean" for row in design.pair_conditions) == 24


def test_pilot_design_is_exact(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="pilot")
    assert len(design.pair_conditions) == 44
    assert {row.pair_id for row in design.pair_conditions} == {"left__back", "left__right"}
    assert {row.backbone for row in design.feature_variants} == {"dinov3_vitl16_pixal3d"}
    assert len(design.feature_variants) == 30


def test_direction_is_balanced_and_pair_independent(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    states = [
        state for state in design.changed_states
        if state.corruption == "translation" and state.severity == 0.12
    ]
    assert sorted(state.direction for state in states).count(-1) == 8
    assert sorted(state.direction for state in states).count(1) == 8
    for state in states:
        rows = [row for row in design.pair_conditions if row.image_state_id == state.state_id]
        assert len({row.image_state_id for row in rows}) == 1


def test_identity_swap_uses_same_view_cyclic_donor(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    state = next(
        item for item in design.changed_states
        if item.shoe_id == "air_huarache_run_ultra_111449"
        and item.changed_view == "left"
        and item.corruption == "identity_swap"
    )
    assert state.donor_shoe_id == "air_max_90_nrg_dc6083_500"
    assert state.donor_view == "left"
```

The materialization test must assert RGB/mask shape `(504, 504)`, a white
background, nearest-neighbor binary mask values, and identity-swap output equal
to the donor foreground on white.

- [ ] **Step 2: Run the tests and verify missing interfaces**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_design.py tests/test_upper_materialize.py tests/test_corruptions.py -q
```

Expected: import or assertion failures for the new upper interfaces.

- [ ] **Step 3: Make feature-variant enumeration public**

In `met3r/design.py`, rename `_feature_variants` to `feature_variants` and keep
the returned records stable. Update `build_design_rows` to call the public name.
Define the typed adapter in `met3r/upper_design.py`:

```python
@dataclass(frozen=True)
class FeatureVariant:
    feature_track: str
    backbone: str
    input_size: int
    feature_mode: str
    resolution: int

    @property
    def key(self) -> str:
        return (
            f"{self.feature_track}__{self.backbone}__{self.input_size}__"
            f"{self.feature_mode}__{self.resolution}"
        )
```

- [ ] **Step 4: Implement reusable state and pair enumeration**

Use these exact identities:

```python
@dataclass(frozen=True)
class ImageState:
    state_id: str
    track: str
    shoe_id: str
    changed_view: str | None
    corruption: str
    severity: float
    direction: int
    seed: int
    donor_shoe_id: str | None = None
    donor_view: str | None = None


@dataclass(frozen=True)
class PairCondition:
    design_id: str
    track: str
    shoe_id: str
    pair_id: str
    first_view: str
    second_view: str
    changed_view: str | None
    image_state_id: str
    corruption: str
    severity: float


@dataclass(frozen=True)
class UpperDesign:
    image_states: tuple[ImageState, ...]
    pair_conditions: tuple[PairCondition, ...]
    stress_states: tuple[ImageState, ...]
    stress_pair_conditions: tuple[PairCondition, ...]
    feature_variants: tuple[FeatureVariant, ...]

    @property
    def changed_states(self) -> tuple[ImageState, ...]:
        return tuple(
            state for state in self.image_states
            if state.corruption != "clean"
        )


@dataclass(frozen=True)
class MaterializedState:
    state_id: str
    state_dir: Path
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    masks: np.ndarray
    metadata_path: Path


def _state_seed(shoe_id: str, view_id: str, corruption: str, severity: float) -> int:
    payload = f"42|{shoe_id}|{view_id}|{corruption}|{severity:.6f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
```

For each severity, assign directions by alternating `-1, +1` over the sorted
16 `(shoe_id, view_id)` states. A changed state participates in the three
unordered pairs containing its changed view. A clean state produces six pairs
once. Pilot filtering happens before pair expansion and produces 44 pair cells.

- [ ] **Step 5: Add signed translation/exposure and full identity swap**

Extend `CorruptionSpec` without breaking existing callers:

```python
@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    strength: float
    direction: int = 1

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_CORRUPTIONS:
            raise ValueError(f"unknown corruption {self.kind!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("corruption strength must be in [0, 1]")
        if self.direction not in (-1, 1):
            raise ValueError("corruption direction must be -1 or 1")
```

Translation uses `direction * round(width * strength)`. Exposure uses
`factor = 2.0 ** (direction * strength)` so positive and negative exposure
have reciprocal magnitude. Add `identity_swap` to `SUPPORTED_CORRUPTIONS`.
For `identity_swap`, require `strength == 1.0`, `donor`, and
`donor_foreground_mask`; return a white RGB canvas with donor foreground and
return the donor mask from `apply_corruption_with_mask`.

Keep legacy `part_transfer` support for the upstream CLI/tests, but remove it
from the upper experiment configuration and design.

- [ ] **Step 6: Implement atomic 504x504 state materialization**

`materialize_image_state` must:

```python
def _stage(rgb_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(rgb_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        )
    with Image.open(mask_path) as image:
        mask = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        ) > 127
    return np.where(mask[..., None], rgb, 255).astype(np.uint8), mask
```

Stage all four views, modify only `changed_view`, write
`images/{view}.png`, `masks/{view}.png`, `images.json`, and `state.json` into a
temporary sibling directory, then rename it to
`states/{state_id}`. The state JSON records source hashes, concrete signed
transform, seed, donor, and stage size.

- [ ] **Step 7: Run focused tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_design.py tests/test_corruptions.py \
  tests/test_upper_design.py tests/test_upper_materialize.py -q
```

Expected: all tests pass; the exact counts are 336, 1,032, 44, and 90.

- [ ] **Step 8: Commit**

```bash
git add met3r/design.py met3r/corruptions.py met3r/upper_design.py \
  met3r/upper_materialize.py tests/test_design.py tests/test_corruptions.py \
  tests/test_upper_design.py tests/test_upper_materialize.py
git commit -m "feat: materialize deterministic upper-view conditions"
```

---

### Task 4: Add the Transactional Job Registry and Run Preparation

**Files:**
- Create: `met3r/job_registry.py`
- Create: `met3r/upper_prepare.py`
- Create: `tests/test_job_registry.py`
- Create: `tests/test_upper_prepare.py`
- Modify: `met3r/bytesize_data.py`
- Modify: `tests/test_bytesize_data.py`

**Interfaces:**
- Consumes: `UpperDesign`, materialized state paths, phase names, and job payloads.
- Produces:
  - `JobRegistry(path: str | Path)`
  - `register(job: JobDefinition) -> None`
  - `claim_next(worker_id: str, phase: str, filters: Mapping[str, str]) -> ClaimedJob | None`
  - `complete(job_id: str, output_manifest: Mapping[str, object]) -> None`
  - `fail(job_id: str, error: str, retryable: bool) -> None`
  - `requeue_stale(phase: str, max_age_seconds: int, now: datetime) -> int`
  - `prepare_run(config_path, scope) -> PreparedRun`.

- [ ] **Step 1: Write failing concurrency, resume, and preparation tests**

```python
def test_claim_is_transactional_and_unique(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "geometry", {"shoe_id": "shoe-a"}))
    first = registry.claim_next("worker-1", "geometry", {})
    second = registry.claim_next("worker-2", "geometry", {})
    assert first is not None and first.job_id == "a"
    assert second is None


def test_completed_job_requires_valid_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "part.parquet"
    output.write_bytes(b"metric")
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "feature", {}))
    registry.claim_next("worker", "feature", {})
    registry.complete("a", {"files": [{"path": str(output), "sha256": sha256_file(output)}]})
    assert registry.validated_state("a") == "completed"
    output.write_bytes(b"changed")
    assert registry.validated_state("a") == "pending"


def test_stale_running_job_is_requeued_after_worker_crash(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "feature", {}))
    registry.claim_next("dead-worker", "feature", {})
    count = registry.requeue_stale(
        phase="feature",
        max_age_seconds=21_600,
        now=datetime.now(timezone.utc) + timedelta(hours=7),
    )
    assert count == 1
    assert registry.claim_next("replacement", "feature", {}).job_id == "a"


def test_prepare_registers_expected_pilot_jobs(tmp_path, real_config) -> None:
    prepared = prepare_run(real_config, scope="pilot", output_root=tmp_path)
    counts = prepared.registry.counts_by_phase()
    assert counts["geometry"] == 32
    assert counts["feature"] == 64


def test_upper_camera_artifacts_have_only_four_504_views(tmp_path, bytesize_manifest) -> None:
    write_upper_camera_artifacts(
        bytesize_manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )
    with np.load(tmp_path / "air_huarache_run_ultra_111449" / "cameras.npz") as data:
        assert data["intrinsics"].shape == (4, 3, 3)
        assert data["extrinsics"].shape == (4, 4, 4)
        assert data["mask"].shape == (4, 504, 504)
        assert data["view_ids"].tolist() == list(UPPER_VIEWS)
```

The 64 pilot feature jobs are 32 image states times the two Pixal3D tracks.

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_job_registry.py tests/test_upper_prepare.py -q
```

Expected: imports fail for the two new modules.

- [ ] **Step 3: Implement the SQLite state machine**

Create this schema:

```sql
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  phase TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending','running','completed','abstained','failed')),
  payload_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  worker_id TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  started_at TEXT,
  finished_at TEXT,
  output_manifest_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_phase_state ON jobs(phase, state);
```

Use these public immutable records:

```python
@dataclass(frozen=True)
class JobDefinition:
    job_id: str
    phase: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    phase: str
    payload: Mapping[str, object]
    attempts: int


@dataclass(frozen=True)
class WorkerSummary:
    completed: int
    abstained: int
    failed: int


@dataclass(frozen=True)
class PreparedRun:
    root: Path
    scope: str
    config_sha256: str
    registry: JobRegistry
```

Open SQLite with `timeout=60`, enable WAL and foreign keys, and claim with
`BEGIN IMMEDIATE`. A claim updates exactly one `pending` row ordered by
`job_id`. `fail(..., retryable=True)` returns the job to `pending` only while
`attempts < 3`; otherwise it sets `failed`. `validated_state` hashes every
declared output and resets an invalid completed row to pending.
`requeue_stale` resets `running` jobs older than its explicit UTC cutoff, which
allows recovery after a killed GPU process without touching fresh workers.

- [ ] **Step 4: Implement run preparation**

`prepare_run` must create:

```text
{output_root}/{scope}/
  config/frozen_config.json
  manifests/upper_manifest.json
  manifests/image_states.jsonl
  manifests/pair_conditions.parquet
  manifests/feature_variants.json
  registry/jobs.sqlite
  cameras/
  pca_fit_images/
  states/
  geometry/
  metrics/parts/
  pca/
  visualizations/
  logs/
  reports/
```

Copy the config by content hash, build the selected design, materialize each
unique state, and register:

- one geometry job per image state
- one feature job per image state and applicable `(backbone, feature_track)`
- geometry-calibration jobs for the full scope
- one stress image state per shoe in the full scope

Add
`write_upper_camera_artifacts(manifest, output_dir, shoe_aliases)` to
`met3r/bytesize_data.py`. It must call `_write_selected_camera_set` only for
the four upper records and write them under the deformation slug supplied by
`CAMERA_DATASET_BY_SHOE`; it must not stage bottom or cross-group records.
Materialize all 16 clean source views at 504x504 under `pca_fit_images/` even
for the pilot, so every backbone PCA basis uses the approved 16-view fit set.
These PCA-only images do not create geometry or metric jobs.

Use a `run_manifest.json` containing config SHA-256, git commit, dirty-tree
fingerprint, Python/package versions, and the exact physical GPU policy.

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_job_registry.py tests/test_upper_prepare.py tests/test_bytesize_data.py -q
```

Expected: all tests pass; calling `prepare_run` twice is idempotent.

- [ ] **Step 6: Commit**

```bash
git add met3r/job_registry.py met3r/upper_prepare.py met3r/bytesize_data.py \
  tests/test_job_registry.py tests/test_upper_prepare.py tests/test_bytesize_data.py
git commit -m "feat: add resumable upper experiment registry"
```

---

### Task 5: Add the Long-Lived DA3 Geometry Worker

**Files:**
- Create: `met3r/upper_geometry.py`
- Create: `tests/test_upper_geometry.py`
- Modify: `met3r/da3_depth.py`
- Modify: `tests/test_da3_depth.py`

**Interfaces:**
- Consumes: geometry jobs, staged images/masks, selected `cameras.npz`, and one loaded DA3 model.
- Produces:
  - `run_geometry_worker(run_root, shoe_id, device, da3_repository) -> WorkerSummary`
  - compact `depth_geometry.npz` without `processed_images`
  - one-factor-at-a-time calibration artifacts for every shoe/view.

- [ ] **Step 1: Write failing compact-artifact and model-reuse tests**

```python
def test_depth_artifact_can_omit_processed_images(tmp_path: Path) -> None:
    artifact = run_fixed_camera_depth(
        FakeDepthModel(),
        images,
        extrinsics,
        intrinsics,
        tmp_path,
        model_id="fake",
        camera_provenance="fixed_precomputed_da3",
        store_processed_images=False,
    )
    with np.load(artifact.arrays_path) as data:
        assert "processed_images" not in data.files
        assert {"depth", "confidence", "mask", "extrinsics", "intrinsics"} <= set(data.files)


def test_geometry_worker_loads_model_once_for_multiple_jobs(fake_prepared_run, monkeypatch) -> None:
    loads = []
    monkeypatch.setattr("met3r.upper_geometry.load_da3_model", lambda *a, **k: loads.append(1) or FakeDepthModel())
    summary = run_geometry_worker(fake_prepared_run, "shoe-a", "cpu", "/fake/da3")
    assert summary.completed == 2
    assert loads == [1]
```

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_da3_depth.py tests/test_upper_geometry.py -q
```

Expected: `store_processed_images` and `upper_geometry` are missing.

- [ ] **Step 3: Make processed-image persistence optional**

Add `store_processed_images: bool = True` to `run_fixed_camera_depth`. Build the
NPZ dictionary without `processed_images` when false, and record
`"store_processed_images": false` in JSON metadata. Preserve the default for
existing callers.

- [ ] **Step 4: Implement the geometry worker loop**

The worker must load DA3 once, then:

```python
registry.requeue_stale(
    phase="geometry",
    max_age_seconds=21_600,
    now=datetime.now(timezone.utc),
)
while job := registry.claim_next(
    worker_id=f"geometry:{shoe_id}",
    phase="geometry",
    filters={"shoe_id": shoe_id},
):
    try:
        state = read_materialized_state(job.payload["state_dir"])
        artifact = run_fixed_camera_depth(
            model,
            state.image_paths,
            cameras.extrinsics,
            cameras.intrinsics,
            output_dir=job_output_dir,
            model_id=DEFAULT_DA3_MODEL_ID,
            model_revision=DEFAULT_DA3_REVISION,
            camera_provenance="fixed_precomputed_da3",
            process_res=504,
            foreground_mask=state.masks,
            view_ids=UPPER_VIEWS,
            camera_groups=("upper",) * 4,
            analysis_tier="primary",
            source_state=repository_state(da3_repository),
            store_processed_images=False,
        )
        validate_fixed_camera_contract(artifact, cameras)
        registry.complete(job.job_id, output_manifest(artifact))
    except (OSError, RuntimeError) as error:
        registry.fail(job.job_id, str(error), retryable=is_transient(error))
    except Exception as error:
        registry.fail(job.job_id, traceback.format_exc(), retryable=False)
        raise
```

Do not catch contract, schema, or hash mismatches as transient failures.

- [ ] **Step 5: Generate full-scope calibration geometry**

After each shoe's clean geometry completes, generate rotation, relative
translation, depth-scale, and relative-depth-noise variants for all four views
using the exact magnitudes in the frozen config. Register every output hash in
the calibration job manifest.

- [ ] **Step 6: Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_da3_depth.py tests/test_geometry_perturbations.py \
  tests/test_upper_geometry.py -q
git add met3r/da3_depth.py met3r/upper_geometry.py \
  tests/test_da3_depth.py tests/test_upper_geometry.py
git commit -m "feat: stream fixed-camera DA3 geometry jobs"
```

Expected: all tests pass and the fake worker initializes its model once.

---

### Task 6: Add Fixed-Basis PCA and One-Variant-at-a-Time Features

**Files:**
- Create: `met3r/pca_features.py`
- Create: `met3r/streaming_features.py`
- Create: `tests/test_pca_features.py`
- Create: `tests/test_streaming_features.py`
- Modify: `met3r/features.py`

**Interfaces:**
- Consumes: loaded backbone/NAF modules, four-view image tensors, feature variants, and clean native features.
- Produces:
  - `fit_pca_basis(batches, max_tokens=200_000, seed=42) -> PCABasis`
  - `PCABasis.save(path)` / `PCABasis.load(path)`
  - `project_pca(feature: Tensor, basis: PCABasis) -> np.ndarray`
  - `iter_feature_variants(native, images, variants, naf) -> Iterator[(FeatureVariant, Tensor)]`.

- [ ] **Step 1: Write failing deterministic PCA and streaming tests**

```python
def test_fixed_pca_colors_are_shared_across_images(tmp_path: Path) -> None:
    first = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    second = first + 10
    basis = fit_pca_basis([first, second], max_tokens=100, seed=42)
    path = tmp_path / "basis.npz"
    basis.save(path)
    loaded = PCABasis.load(path)
    np.testing.assert_allclose(loaded.components, basis.components)
    assert loaded.fingerprint == basis.fingerprint
    assert not np.array_equal(project_pca(first, basis), project_pca(first, fit_pca_basis([first], seed=42)))


def test_streaming_generator_holds_only_one_upsampled_variant(fake_naf) -> None:
    native = torch.ones(4, 8, 2, 2)
    images = torch.ones(4, 3, 8, 8)
    variants = (
        FeatureVariant("t", "b", 8, "native", 2),
        FeatureVariant("t", "b", 8, "bilinear", 4),
        FeatureVariant("t", "b", 8, "naf_normal_guide", 8),
    )
    iterator = iter_feature_variants(native, images, variants, fake_naf)
    assert [variant.key for variant, _ in iterator] == [item.key for item in variants]
```

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pca_features.py tests/test_streaming_features.py -q
```

Expected: imports fail for the new modules.

- [ ] **Step 3: Implement deterministic PCA without scikit-learn**

Flatten BCHW features to rows of channels, deterministically subsample at most
200,000 rows, compute the mean and `torch.pca_lowrank(..., q=3, center=False)`,
fix component signs by making the largest-absolute loading positive, and
calculate component-wise 1st and 99th percentiles on the fit tokens.

Persist only:

```python
@dataclass(frozen=True)
class PCABasis:
    mean: np.ndarray
    components: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fingerprint: str

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        np.savez_compressed(
            destination,
            mean=self.mean,
            components=self.components,
            lower=self.lower,
            upper=self.upper,
            fingerprint=np.asarray(self.fingerprint),
            seed=np.asarray(42),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PCABasis":
        with np.load(path) as artifact:
            basis = cls(
                mean=np.asarray(artifact["mean"]),
                components=np.asarray(artifact["components"]),
                lower=np.asarray(artifact["lower"]),
                upper=np.asarray(artifact["upper"]),
                fingerprint=str(np.asarray(artifact["fingerprint"]).item()),
            )
        if basis.fingerprint != pca_fingerprint(basis):
            raise ValueError("PCA basis fingerprint mismatch")
        return basis
```

Projection always uses these stored bounds and never per-image normalization.

- [ ] **Step 4: Implement feature streaming**

Add `MultiScaleFeatureExtractor.extract_native(images) -> Tensor` in
`met3r/features.py`. In `met3r/streaming_features.py`, yield:

```python
if variant.feature_mode == "native":
    yield variant, native
elif variant.feature_mode == "bilinear":
    yield variant, F.interpolate(
        native, (variant.resolution, variant.resolution),
        mode="bilinear", align_corners=False,
    )
else:
    guide_name = {
        "naf_normal_guide": "normal",
        "naf_blurred_guide": "blurred",
        "naf_shuffled_guide": "shuffled",
    }[variant.feature_mode]
    guides = build_naf_guides(images)
    yield variant, naf(
        guides[guide_name],
        native,
        (variant.resolution, variant.resolution),
    )
```

Create guides lazily once per image batch, delete each yielded upsampled tensor
after the caller advances, and call `torch.cuda.empty_cache()` only after an
OOM retry, not in the normal inner loop.

- [ ] **Step 5: Implement lowest/highest-only rendering**

`render_pca_png` writes a PNG only when the variant is native or its resolution
equals the track maximum. Add a test that a 74/148/296 variant produces no file
while native and 592 do. Fit every basis from all 16 staged files under
`pca_fit_images/`, including during the two-shoe pilot. For Pixal3D DINOv3-L,
fit one basis from the combined clean 592-input and 512-input native batches
before rendering either track.

- [ ] **Step 6: Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_features.py tests/test_pca_features.py tests/test_streaming_features.py -q
git add met3r/features.py met3r/pca_features.py met3r/streaming_features.py \
  tests/test_pca_features.py tests/test_streaming_features.py
git commit -m "feat: stream multiscale features with fixed PCA"
```

Expected: all tests pass and no test creates `features.npz`.

---

### Task 7: Stream Factorial Metrics and Geometry Calibration to Parquet

**Files:**
- Create: `met3r/upper_metric_worker.py`
- Create: `tests/test_upper_metric_worker.py`
- Modify: `met3r/results.py`
- Modify: `met3r/visualization.py`

**Interfaces:**
- Consumes: feature jobs, clean/changed geometry, clean/changed image tensors, fixed PCA basis, and pair-condition rows.
- Produces:
  - `run_feature_worker(run_root, backbone, device) -> WorkerSummary`
  - atomic `metrics/parts/{track}/{job_id}.parquet`
  - PCA/diagnostic PNG manifests
  - geometry-calibration Parquet parts.

- [ ] **Step 1: Write failing fake-backbone integration tests**

```python
def test_worker_writes_factorial_rows_without_feature_artifact(fake_run, monkeypatch) -> None:
    monkeypatch.setattr("met3r.upper_metric_worker.load_feature_backbone", lambda name: FakeBackbone())
    monkeypatch.setattr("met3r.upper_metric_worker.load_naf_upsampler", lambda **kw: FakeNAF())
    summary = run_feature_worker(fake_run, "fake_backbone", "cpu")
    assert summary.failed == 0
    parts = list((fake_run / "metrics" / "parts").rglob("*.parquet"))
    frame = pd.concat(pd.read_parquet(path) for path in parts)
    assert {"s00", "s10", "s01", "s11", "feature_effect", "geometry_effect", "interaction"} <= set(frame)
    assert not list(fake_run.rglob("features.npz"))
    assert not any("feature__" in path.name for path in fake_run.rglob("*"))


def test_worker_persists_directional_support_and_provenance(fake_run, monkeypatch) -> None:
    run_feature_worker(fake_run, "fake_backbone", "cpu")
    frame = read_all_parts(fake_run)
    assert {
        "s11_first_to_second_score", "s11_second_to_first_score",
        "s11_first_to_second_overlap", "s11_second_to_first_overlap",
        "cycle_error_px_median", "relative_depth_error_median",
        "config_sha256", "camera_provenance", "backbone_revision"
    } <= set(frame)


def test_stress_rows_keep_both_view_quality_covariates(fake_stress_run, monkeypatch) -> None:
    run_feature_worker(fake_stress_run, "fake_backbone", "cpu")
    frame = read_all_parts(fake_stress_run)
    assert {
        "first_semantic_gate", "second_semantic_gate",
        "first_preservation_gate", "second_preservation_gate",
        "first_camera_preserved", "second_camera_preserved",
        "first_outsole_preserved", "second_outsole_preserved",
        "first_fit_relation", "second_fit_relation"
    } <= set(frame)
```

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest tests/test_upper_metric_worker.py -q
```

Expected: module import fails.

- [ ] **Step 3: Preserve directional metric outputs**

Extend `flatten_factorial_result` with explicit first-to-second and
second-to-first score/overlap values for `s00`, `s10`, `s01`, and `s11`. Keep
the existing symmetric fields for compatibility. Use names such as
`s11_first_to_second_score` and `s11_second_to_first_overlap`; do not overwrite
the symmetric `s11` value.

- [ ] **Step 4: Implement one-load-per-backbone feature worker**

The worker sequence is:

1. Load the assigned backbone and NAF once.
2. Load all clean staged images required to fit the backbone PCA basis.
3. Fit or validate the basis fingerprint.
4. Cache only clean native four-view tensors for the active shoe/track.
5. Claim one image-state/track job.
6. Extract the changed native tensor.
7. Zip clean and changed variant iterators by exact `FeatureVariant.key`.
8. Score every pair condition attached to the state with
   `score_multiscale_factorial`.
9. Attach both views' deformation-quality covariates for stress rows.
10. Render allowed PCA/diagnostic PNGs.
11. Atomically write one Parquet part and complete the registry job.
12. Delete changed/native/upsampled tensors before the next job.

At worker startup, requeue only stale jobs from the same phase. Wrap each claim
in `try/except`: retry a classified CUDA OOM or transient I/O error at most
twice through `registry.fail(..., retryable=True)`; record every schema,
provenance, or contract error with `retryable=False` and re-raise it so the
shell phase fails.

```python
registry.requeue_stale(
    phase="feature",
    max_age_seconds=21_600,
    now=datetime.now(timezone.utc),
)
try:
    run_claimed_feature_job(job)
except (torch.cuda.OutOfMemoryError, OSError) as error:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        release_job_tensors()
        torch.cuda.empty_cache()
    registry.fail(job.job_id, traceback.format_exc(), retryable=True)
except Exception:
    registry.fail(job.job_id, traceback.format_exc(), retryable=False)
    raise
```

The scoring call for one variant is:

```python
def select_geometry(geometry: ViewGeometry, index: int) -> ViewGeometry:
    return ViewGeometry(
        depth=geometry.depth[index:index + 1],
        mask=geometry.mask[index:index + 1],
        confidence=(
            geometry.confidence[index:index + 1]
            if geometry.confidence is not None else None
        ),
        intrinsics=geometry.intrinsics[index:index + 1],
        extrinsics_w2c=geometry.extrinsics_w2c[index:index + 1],
    )


result = score_multiscale_factorial(
    {variant.key: clean_features[first:first + 1]},
    {variant.key: clean_features[second:second + 1]},
    {variant.key: changed_features[first:first + 1]},
    {variant.key: changed_features[second:second + 1]},
    select_geometry(base_geometry, first),
    select_geometry(base_geometry, second),
    select_geometry(adaptive_geometry, first),
    select_geometry(adaptive_geometry, second),
    config=KnownCameraConfig(
        relative_depth_tolerance=0.03,
        min_overlap_fraction=0.08,
    ),
    confidence_percentile=10.0,
)[variant.key]
```

Use the intersection support already enforced by
`score_multiscale_factorial`.

- [ ] **Step 5: Add atomic Parquet output and persistent-feature guard**

Write to `{job_id}.parquet.tmp`, read it back with PyArrow, verify schema and
row count, rename to `{job_id}.parquet`, then hash it. Before completing a job,
walk the persistent run root and reject files whose NPZ metadata declares
`artifact_type=dino_features` or whose basename is `features.npz`.

- [ ] **Step 6: Stream geometry-calibration scoring**

For full-scope clean states, reuse each in-memory feature variant to score all
17 non-zero perturbations for each view and every pair containing that view.
Write calibration rows under
`metrics/parts/geometry_calibration/{job_id}.parquet` with perturbation kind,
magnitude, directed changed view, base score, perturbed score, geometry effect,
overlap, and abstention.

- [ ] **Step 7: Implement diagnostic retention**

Save diagnostics for:

- every pilot cell
- clean cells
- middle severity (`0.12`, `0.50`, `0.70`, or `0.75` as applicable)
- maximum severity
- every abstained cell

Use `save_direction_diagnostics` for both directions and include the condition,
variant key, and direction in the filename.

- [ ] **Step 8: Run tests and commit**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_results.py tests/test_visualization.py \
  tests/test_upper_metric_worker.py tests/test_pipeline_integration.py -q
git add met3r/results.py met3r/visualization.py met3r/upper_metric_worker.py \
  tests/test_upper_metric_worker.py
git commit -m "feat: stream upper-view metrics to parquet"
```

Expected: all tests pass; the fake run has Parquet/PCA/diagnostics but no raw
DINO feature artifact.

---

### Task 8: Add Pilot Gates, Aggregation, CLI, and GPU Scripts

**Files:**
- Create: `met3r/pilot_gate.py`
- Create: `met3r/upper_cli.py`
- Create: `scripts/run_upper_pilot.sh`
- Create: `tests/test_pilot_gate.py`
- Create: `tests/test_upper_cli.py`
- Modify: `setup.py`

**Interfaces:**
- Consumes: prepared run tree, registry, Parquet parts, PCA manifests, and disk usage.
- Produces:
  - `validate_pilot(run_root) -> PilotGateReport`
  - `aggregate_metric_parts(run_root, track) -> Path`
  - CLI commands `prepare`, `geometry-worker`, `feature-worker`, `aggregate`, `validate-pilot`, and `status`.

- [ ] **Step 1: Write failing pilot-gate tests**

```python
def test_gate_passes_complete_operational_pilot(valid_pilot_run: Path) -> None:
    report = validate_pilot(valid_pilot_run)
    assert report.passed
    assert report.expected_metric_rows == 1320
    assert report.clean_primary_directional_coverage >= 0.90
    assert report.clean_primary_pair_coverage >= 0.90
    assert report.clean_negative_control_abstention >= 0.90
    assert report.raw_feature_artifacts == []


def test_gate_fails_raw_features_and_storage_overage(valid_pilot_run: Path) -> None:
    (valid_pilot_run / "features.npz").write_bytes(b"forbidden")
    report = validate_pilot(valid_pilot_run, projected_bytes=301 * 1024**3)
    assert not report.passed
    assert "raw_feature_artifact" in report.failed_checks
    assert "storage_limit" in report.failed_checks
```

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q
```

Expected: imports fail.

- [ ] **Step 3: Implement hard gates and storage projection**

Use:

```python
@dataclass(frozen=True)
class PilotGateReport:
    passed: bool
    failed_checks: tuple[str, ...]
    expected_metric_rows: int
    actual_metric_rows: int
    clean_directional_coverage: float
    clean_primary_directional_coverage: float
    clean_primary_pair_coverage: float
    clean_negative_control_abstention: float
    raw_feature_artifacts: tuple[str, ...]
    projected_total_bytes: int
```

The report must check:

- all registered pilot jobs completed or explicitly abstained
- exactly 1,320 metric rows
- no mismatched config/input/model/provenance hashes
- finite scores for all non-abstained rows
- clean `right__front` directional and symmetric-pair coverage at least 0.90
- clean `left__right` expected-abstention rate at least 0.90
- one PCA basis fingerprint per backbone
- a forced sentinel rerun matches design ID and metrics within
  `rtol=1e-5, atol=1e-6`
- no persistent raw DINO feature artifact
- projected persistent storage at most 300 GB

Project storage by artifact category, using measured bytes per pilot image
state/job and the full design counts rather than multiplying the whole pilot
directory by one scalar. Write `reports/pilot_gate.json` and
`reports/pilot_storage_projection.json`.

- [ ] **Step 4: Implement metric aggregation**

Read all validated part paths from the registry, verify hashes, concatenate with
`pyarrow.dataset`, sort by `design_id` and feature key, reject duplicate primary
keys, and atomically write:

```text
metrics/controlled_fidelity.parquet
metrics/geometry_calibration.parquet
metrics/real_deformation_stress.parquet
metrics/summary_preview.csv
```

- [ ] **Step 5: Implement the dedicated CLI**

Expose:

```text
python -m met3r.upper_cli prepare CONFIG --scope pilot|full
python -m met3r.upper_cli geometry-worker RUN_ROOT --shoe-id ID --device cuda:0
python -m met3r.upper_cli feature-worker RUN_ROOT --backbone NAME --device cuda:0
python -m met3r.upper_cli aggregate RUN_ROOT
python -m met3r.upper_cli validate-pilot RUN_ROOT --rerun-sentinel
python -m met3r.upper_cli status RUN_ROOT [--require-complete]
```

Add `met3r-upper-experiment=met3r.upper_cli:main` to `setup.py`.

- [ ] **Step 6: Write the pilot script**

Create `scripts/run_upper_pilot.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/conda/envs/pixal3d/bin/python
CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot
DA3_REPO=/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3

"$PYTHON" -m met3r.upper_cli prepare "$CONFIG" --scope pilot

for shoe in \
  air_huarache_run_ultra_111449 \
  air_max_90_surplus_cq7743_300
do
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m met3r.upper_cli geometry-worker \
    "$RUN_ROOT" --shoe-id "$shoe" --device cuda:0 --da3-repository "$DA3_REPO"
done

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m met3r.upper_cli feature-worker \
  "$RUN_ROOT" --backbone dinov3_vitl16_pixal3d --device cuda:0

"$PYTHON" -m met3r.upper_cli aggregate "$RUN_ROOT"
"$PYTHON" -m met3r.upper_cli validate-pilot "$RUN_ROOT" --rerun-sentinel
```

- [ ] **Step 7: Run tests and commit**

```bash
chmod +x scripts/run_upper_pilot.sh
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_pilot_gate.py tests/test_upper_cli.py -q
git add met3r/pilot_gate.py met3r/upper_cli.py \
  scripts/run_upper_pilot.sh \
  tests/test_pilot_gate.py tests/test_upper_cli.py setup.py
git commit -m "feat: orchestrate resumable pilot experiment"
```

Expected: tests pass and `python -m met3r.upper_cli --help` lists all six
commands.

---

### Task 9: Add Scale-Safe Analysis, Stress Strata, Figures, and Report

**Files:**
- Create: `met3r/upper_analysis.py`
- Create: `tests/test_upper_analysis.py`
- Create: `scripts/run_upper_full.sh`
- Modify: `met3r/upper_cli.py`
- Modify: `KNOWN_CAMERA_EXPERIMENT.md`

**Interfaces:**
- Consumes: the three aggregated Parquet tracks and frozen run manifest.
- Produces:
  - `analyze_upper_run(run_root) -> AnalysisOutputs`
  - endpoint CSV/Parquet tables
  - leave-one-shoe-out tables
  - primary figures
  - `reports/final_report.md`.

- [ ] **Step 1: Write failing endpoint and stratification tests**

```python
def test_primary_fidelity_uses_fixed_geometry_feature_effect() -> None:
    frame = synthetic_factorial_frame()
    prepared = prepare_primary_metric(frame)
    np.testing.assert_allclose(prepared["metric_delta"], frame["s10"] - frame["s00"])
    assert not np.array_equal(prepared["metric_delta"], frame["s11"] - frame["s00"])


def test_identity_swap_has_auc_but_no_severity_spearman() -> None:
    summary = summarize_controlled_fidelity(synthetic_identity_frame())
    row = summary.iloc[0]
    assert row["clean_corrupt_auc"] == 1.0
    assert np.isnan(row["severity_spearman"])


def test_clean_normalization_is_within_backbone_variant() -> None:
    normalized = robust_clean_normalize(synthetic_two_backbone_frame())
    groups = normalized.groupby(["backbone", "feature_track", "feature_mode", "resolution"])
    assert all(abs(group.loc[group["corruption"] == "clean", "normalized_score"].median()) < 1e-8 for _, group in groups)


def test_stress_report_keeps_all_and_camera_preserved_strata() -> None:
    outputs = summarize_stress(synthetic_stress_frame())
    assert set(outputs["stratum"]) >= {"all", "both_camera_preserved"}
```

- [ ] **Step 2: Run and confirm failures**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest tests/test_upper_analysis.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement robust within-variant normalization**

Use:

```python
@dataclass(frozen=True)
class AnalysisOutputs:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    report_path: Path
```

For each `(backbone, feature_track, feature_mode, resolution)` group:

```python
clean = group.loc[group["corruption"] == "clean", "s00"].to_numpy(float)
center = np.median(clean)
scale = 1.4826 * np.median(np.abs(clean - center))
if not np.isfinite(scale) or scale < 1e-6:
    q25, q75 = np.quantile(clean, [0.25, 0.75])
    scale = max((q75 - q25) / 1.349, 1e-6)
group["normalized_score"] = (group["s10"] - center) / scale
group["normalized_delta"] = (group["s10"] - group["s00"]) / scale
group["metric_delta"] = group["s10"] - group["s00"]
```

Use `metric_delta`/`s10` for primary fixed-geometry fidelity, `s11` only for
the combined adaptive secondary analysis, and report geometry/interaction
separately. For each controlled series identified by
`(shoe_id, pair_id, changed_view, backbone, feature_track, feature_mode,
resolution, corruption)`, add one in-memory severity-zero observation with
`metric_delta=0` and `score=s00`; do not write duplicated clean baselines back
to the metric Parquet. Average changed-view placements within pair only after
computing the directional endpoint, then average pairs within shoe.

- [ ] **Step 4: Implement endpoint families**

Compute:

- identity swap: clean/corrupt AUROC and pairwise rank accuracy
- translation/TPS: within-pair severity Spearman and rank accuracy
- exposure/blur/JPEG: clean-normalized signed and absolute delta
- resolution utility: endpoint difference from native within backbone/mode
- NAF controls: normal minus bilinear, blurred, and shuffled guide endpoints
- attribution: feature, geometry, interaction, and absolute feature share
- geometry envelope: perturbation magnitude required to equal each observed
  absolute feature effect
- coverage: directional overlap and abstention
- stress: source/deformed delta for `all`, `both_camera_preserved`,
  semantic-gate, preservation-gate, and fit-relation strata

Average pair endpoints within shoe, then average shoes. Add 1,000 seed-42
shoe-cluster bootstrap repeats and leave-one-shoe-out estimates. Do not compute
confirmatory p-values.

- [ ] **Step 5: Implement figures and report**

Generate:

```text
reports/tables/controlled_fidelity.csv
reports/tables/backbone_comparison.csv
reports/tables/resolution_utility.csv
reports/tables/naf_guide_ablation.csv
reports/tables/attribution.csv
reports/tables/geometry_calibration.csv
reports/tables/real_deformation_stress.csv
reports/tables/leave_one_shoe_out.csv
reports/figures/resolution_fidelity.png
reports/figures/severity_curves.png
reports/figures/backbone_discrimination.png
reports/figures/naf_guide_ablation.png
reports/figures/attribution.png
reports/figures/geometry_calibration.png
reports/figures/real_deformation_stress.png
reports/figures/coverage.png
reports/final_report.md
```

The report opens with the fixed/precomputed DA3 limitation, states that only
upper views/four shoes were tested, separates all three tracks, and treats the
four-shoe intervals as descriptive.

- [ ] **Step 6: Write the four-GPU full script**

Create `scripts/run_upper_full.sh` with physical GPU mapping:

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/conda/envs/pixal3d/bin/python
CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/configs/upper_dino_fidelity.json
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full
DA3_REPO=/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3

"$PYTHON" -m met3r.upper_cli prepare "$CONFIG" --scope full

shoes=(
  air_huarache_run_ultra_111449
  air_max_90_nrg_dc6083_500
  air_max_90_surplus_cq7743_300
  air_tuned_max_cv6984_001
)
gpus=(4 5 6 7)
pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" -m met3r.upper_cli geometry-worker \
    "$RUN_ROOT" --shoe-id "${shoes[$index]}" --device cuda:0 \
    --da3-repository "$DA3_REPO" \
    >"$RUN_ROOT/logs/geometry_${shoes[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0

backbones=(
  dino_vitb16
  dinov2_vitb14
  dinov3_vitb16
  dinov3_vitl16_pixal3d
)
pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON" -m met3r.upper_cli feature-worker \
    "$RUN_ROOT" --backbone "${backbones[$index]}" --device cuda:0 \
    >"$RUN_ROOT/logs/features_${backbones[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0

"$PYTHON" -m met3r.upper_cli aggregate "$RUN_ROOT"
"$PYTHON" -m met3r.upper_cli analyze "$RUN_ROOT"
```

- [ ] **Step 7: Wire the CLI, document commands, run tests, and commit**

```bash
chmod +x scripts/run_upper_full.sh
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_analysis.py tests/test_upper_analysis.py tests/test_upper_cli.py -q
git add met3r/upper_analysis.py met3r/upper_cli.py \
  tests/test_upper_analysis.py KNOWN_CAMERA_EXPERIMENT.md \
  scripts/run_upper_full.sh
git commit -m "feat: analyze upper-view DINO fidelity experiment"
```

Expected: all tests pass, `upper_cli --help` lists `analyze`, and a synthetic
run produces every named table, figure, and report.

---

### Task 10: Run Full Verification and the Real GPU-1 Pilot

**Files:**
- Verify: all source and tests
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot`

**Interfaces:**
- Consumes: Tasks 1–9 and cached/downloadable pinned model weights.
- Produces: a complete real-model pilot, `pilot_gate.json`, storage projection,
  timings, PCA images, diagnostics, and 1,320 metric rows.

V1 is retained at
`/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v1/pilot`
as a failed protocol: all 1,320 rows were produced, but mixed primary/opposite
clean directional coverage was `0.22916666666666666`. V2 never overwrites it.

- [ ] **Step 1: Run the full software test suite**

```bash
cd /root/dev/met3r-worktrees/known-camera-met3r
/opt/conda/envs/pixal3d/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` exits 0.

- [ ] **Step 2: Verify GPUs and disk**

```bash
nvidia-smi -i 1 --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
df -h /root/dev/ByteSize/Reconstruction
```

Expected: GPU 1 is addressable and the target filesystem has more than
300 GB free.

- [ ] **Step 3: Run the pilot**

```bash
PILOT_GPU=1 bash scripts/run_upper_pilot.sh
```

Expected: exit 0. Long-running processes are monitored by polling their logs;
do not leave the user without a progress update for more than 60 seconds.

- [ ] **Step 4: Inspect hard gates and artifacts**

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot")
gate = json.loads((root / "reports/pilot_gate.json").read_text())
projection = json.loads((root / "reports/pilot_storage_projection.json").read_text())
assert gate["passed"], gate
assert gate["expected_metric_rows"] == 1320
assert gate["clean_primary_directional_coverage"] >= 0.90
assert gate["clean_primary_pair_coverage"] >= 0.90
assert gate["clean_negative_control_abstention"] >= 0.90
assert not gate["raw_feature_artifacts"]
assert projection["projected_total_bytes"] <= 300 * 1024**3
print(json.dumps({"gate": gate, "projection": projection}, indent=2))
PY
```

Expected: every hard gate passes and projected storage is at most 300 GB.

- [ ] **Step 5: Review visual comparability**

Open a representative native/highest-resolution PCA montage with the local
image viewer and verify:

- the same backbone basis fingerprint is printed in every panel metadata
- colors are not independently stretched per image
- bilinear/NAF normal/blurred/shuffled panels exist at the highest resolution
- source and changed panels use identical display bounds

If only a scientific sanity check is unexpected, record it in
`reports/pilot_gate.json` and continue. If code/config/provenance changes, clear
only the invalidated pilot jobs through the registry and rerun the complete
pilot under the new config hash.

- [ ] **Step 6: Create an annotated tag for the frozen full-run code**

```bash
git status --short
git tag -a upper-dino-fidelity-v2-pilot-passed \
  -m "Pilot passed for 20260724_upper_v2 on physical GPU 1"
```

Expected: source files are clean; experiment artifacts remain outside git.

---

### Task 11: Run the Four-GPU Full Experiment and Generate the Report

**Files:**
- Produce outside git:
  `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full`
- Read: `scripts/run_upper_full.sh`

**Interfaces:**
- Consumes: the passing frozen pilot tag and GPUs 4–7.
- Produces: complete controlled, geometry-calibration, stress-test, PCA,
  diagnostic, statistical, and report artifacts.

- [ ] **Step 1: Confirm the pilot tag and gate**

```bash
cd /root/dev/met3r-worktrees/known-camera-met3r
git describe --exact-match --tags HEAD
/opt/conda/envs/pixal3d/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/pilot/reports/pilot_gate.json")
assert json.loads(path.read_text())["passed"]
PY
```

Expected: tag `upper-dino-fidelity-v2-pilot-passed` and a passing gate.

- [ ] **Step 2: Run all full phases**

```bash
bash scripts/run_upper_full.sh
```

Expected: exit 0. DA3 finishes before feature workers start. Geometry is sharded
one shoe per GPU; features are sharded one backbone per GPU.

- [ ] **Step 3: Validate registry and row coverage**

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.upper_cli status \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full \
  --require-complete
```

Expected:

- controlled fidelity: 92,880 rows
- geometry calibration: 73,440 rows
- real-deformation stress: 2,160 rows
- failed jobs: 0
- invalid completed output hashes: 0
- raw DINO feature artifacts: 0
- 16 real-deformation inputs represented
- geometry-calibration rows present for every shoe/view/perturbation/variant

- [ ] **Step 4: Run final aggregation and analysis idempotently**

```bash
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full
/opt/conda/envs/pixal3d/bin/python -m met3r.upper_cli aggregate "$RUN_ROOT"
/opt/conda/envs/pixal3d/bin/python -m met3r.upper_cli analyze "$RUN_ROOT"
```

Expected: rerunning either command produces the same content hashes.

- [ ] **Step 5: Verify final deliverables**

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
from pathlib import Path
root = Path("/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2/full")
required = [
    "metrics/controlled_fidelity.parquet",
    "metrics/geometry_calibration.parquet",
    "metrics/real_deformation_stress.parquet",
    "reports/tables/controlled_fidelity.csv",
    "reports/tables/backbone_comparison.csv",
    "reports/tables/resolution_utility.csv",
    "reports/tables/naf_guide_ablation.csv",
    "reports/tables/attribution.csv",
    "reports/tables/geometry_calibration.csv",
    "reports/tables/real_deformation_stress.csv",
    "reports/tables/leave_one_shoe_out.csv",
    "reports/final_report.md",
]
missing = [item for item in required if not (root / item).is_file()]
assert not missing, missing
assert not list(root.rglob("features.npz"))
print("final artifact verification passed")
PY
```

Expected: `final artifact verification passed`.

- [ ] **Step 6: Review the scientific report**

Confirm that `reports/final_report.md`:

- calls cameras fixed/precomputed DA3 estimates rather than GT
- separates feature, adaptive-depth geometry, interaction, and pose/depth
  perturbation evidence
- compares backbone discrimination rather than raw score magnitudes
- distinguishes semantic identity, spatial fidelity, and nuisance sensitivity
- treats real deformations as a separate stress test with all/quality strata
- states the four-shoe and upper-only generalization limitations
- reports negative or null findings without changing the frozen protocol

Experiment artifacts stay outside git. Commit only a concise run pointer and
content hashes if the repository convention requires one; otherwise leave the
tagged source commit unchanged.

---

## Spec Coverage Check

| Approved requirement | Implemented by |
| --- | --- |
| Four shoes, upper four views, no bottom | Tasks 2–3 |
| 1,032 pair conditions and both changed-view placements | Task 3 |
| DINO/DINOv2/DINOv3/Pixal3D-L and 90 variants | Tasks 2–3, 6–7 |
| Fixed/precomputed DA3 camera wording and camera contract | Tasks 2, 5, 9 |
| Four-cell feature/depth attribution | Task 7 |
| Controlled pose/depth calibration | Tasks 5, 7, 9 |
| All 16 real deformations as a separate stress test | Tasks 2–4, 7, 9 |
| No persistent raw DINO features | Tasks 6–8, 10–11 |
| Fixed PCA basis; lowest/highest PNG only | Tasks 6–7 |
| GPU 1 pilot and GPU 4–7 full sharding | Tasks 8, 10–11 |
| SQLite resumability and atomic outputs | Tasks 4, 7 |
| Pilot hard gates and 300 GB cap | Tasks 8, 10 |
| Shoe-equal weighting, bootstrap, leave-one-out | Task 9 |
| Human rating deferred | No implementation in this plan, as approved |
