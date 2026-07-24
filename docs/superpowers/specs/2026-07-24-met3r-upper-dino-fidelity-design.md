# MEt3R Upper-View DINO Fidelity Experiment Design

**Date:** 2026-07-24

**Status:** User-approved design

**Experiment ID:** `20260724_upper_v1`

## 1. Objective

This experiment evaluates whether DINO-family features can serve as a proxy for
multi-view inconsistency in the Pixal3D conditioning setting. It tests three
questions:

1. Whether low-resolution features primarily capture semantic consistency and
   higher-resolution features add reliable pixel/local-alignment sensitivity.
2. How DINO, DINOv2, DINOv3, and the Pixal3D DINOv3-L backbone differ in their
   ability to distinguish consistent and inconsistent view pairs.
3. How much of a metric-score change is attributable to image features rather
   than errors in the fixed/precomputed DA3 geometry.

The saved cameras are DA3 estimates, not calibrated ground truth. Every output
and report must call them **fixed/precomputed DA3 cameras**. The experiment can
measure sensitivity to controlled geometry errors and decompose observable
feature/depth effects, but it cannot identify the unknown true camera error of
an individual view.

## 2. Scope

### Included

- Four shoes:
  - `air_huarache_run_ultra_111449`
  - `air_max_90_nrg_dc6083_500`
  - `air_max_90_surplus_cq7743_300`
  - `air_tuned_max_cv6984_001`
- Four upper-orbit views per shoe, in the canonical order
  `left`, `back`, `right`, `front`.
- All six unordered upper-view pairs per shoe.
- Controlled corruption, geometry calibration, and a separate real-deformation
  stress test.
- Full metric computation at every registered feature resolution.
- PCA visualizations at the lowest/native and highest resolutions only.

### Excluded

- Bottom images.
- Upper-to-bottom cross-group pairs.
- Claims about outsole/bottom-view generalization.
- Persisting raw DINO feature tensors.
- Human ratings during the initial metric run. Human validation is a separate
  later stage.

## 3. Input Data

### 3.1 Clean controlled-experiment inputs

For each selected upper view, use the clean image and foreground mask associated
with the deformation task:

```text
/root/dev/ByteSize/Reconstruction/
  six_view_experiment_result_precomputed/
  multiview/{shoe_id}/{view}/deformation/source.png
/root/dev/ByteSize/Reconstruction/
  six_view_experiment_result_precomputed/
  multiview/{shoe_id}/{view}/deformation/source_mask.png
```

The data loader must verify that all 16 image/mask pairs exist, share the
expected dimensions, and map to the correct shoe and view.

The deformation assets are 1080x1080, while the selected fixed-camera
intrinsics correspond to the existing 504x504 DA3 processed-image artifacts.
Materialize every clean, deformed, and controlled-corruption image state at
504x504 before DA3 inference: use Lanczos for RGB, nearest-neighbor for masks,
and force non-foreground pixels to white. Feature backbones then resize these
staged 504x504 images independently to their registered 592, 588, or 512 input
size. Never pass the 1080x1080 deformation asset directly with the 504x504
intrinsics.

### 3.2 Real-deformation stress inputs

Use all 16 upper-view deformation results:

```text
/root/dev/ByteSize/Reconstruction/
  six_view_experiment_result_precomputed/
  multiview/{shoe_id}/{view}/deformation/deformed.png
```

Load `deformed_mask.png`, `deformation.json`, `metrics.json`, and `task.json`
from the same directory. Retain the following metadata as covariates:

- deformation status and selected attempt
- `semantic_gate`
- `preservation_gate`
- `camera_preserved`
- outsole-preservation result
- last-render fit relation and whether it improved or worsened

All 16 deformations remain in the stress test even when their status is
`low_quality`. Results are reported both for the complete set and for
quality-stratified subsets. They are never pooled with controlled-corruption
results.

### 3.3 Camera and depth provenance

Use the already audited upper-view camera mapping from the ByteSize
reconstruction artifacts. Camera poses and intrinsics remain fixed for the
primary metric and for adaptive DA3 depth inference. The pipeline must verify
that DA3 returns the supplied pose/intrinsic contract unchanged.

The clean/base geometry is inferred once per clean four-view shoe set. For each
changed image state, adaptive geometry re-estimates DA3 depth, confidence, and
validity masks while retaining the same supplied fixed/precomputed camera pose
and intrinsics.

## 4. Experimental Tracks

### 4.1 Controlled fidelity

For each shoe, evaluate all six unordered pairs. For every non-clean condition,
apply the change to each member of the pair in turn:

- A clean / B changed
- A changed / B clean

This is required even though the final MEt3R score is bidirectional, because a
view's content, visibility, and overlap can make corruption placement
asymmetric.

Controlled corruptions are:

| Corruption | Non-zero normalized strengths | Intended evidence |
| --- | --- | --- |
| Translation | `0.03, 0.06, 0.12, 0.20` | spatial alignment |
| Thin-plate warp | `0.25, 0.50, 0.75, 1.00` | non-rigid spatial alignment |
| Exposure | `0.15, 0.30, 0.50, 0.75` | appearance invariance |
| Blur | `0.20, 0.40, 0.70, 1.00` | local-detail sensitivity |
| JPEG | `0.25, 0.50, 0.75, 1.00` | compression invariance |
| Identity swap | `1.00` | semantic identity |

Translation direction, exposure sign, and TPS displacement seed are assigned
once per shoe, changed view, corruption, and strength. Signs/directions are
balanced across the 16 changed-view states at each strength while remaining
deterministic under seed 42. The assignment deliberately excludes pair ID, so
the same changed view state can be reused in all three pairs that contain that
view. Spatial corruptions transform the foreground mask with the image.

Identity swap replaces the complete foreground with another shoe's foreground
from the same view ID. Donors follow this fixed cyclic mapping:

```text
air_huarache_run_ultra_111449
  -> air_max_90_nrg_dc6083_500
air_max_90_nrg_dc6083_500
  -> air_max_90_surplus_cq7743_300
air_max_90_surplus_cq7743_300
  -> air_tuned_max_cv6984_001
air_tuned_max_cv6984_001
  -> air_huarache_run_ultra_111449
```

The donor foreground mask becomes the changed-view foreground mask. This is an
upper-bound semantic corruption rather than a pure identity-only intervention:
shoe shape can also change. That limitation must be stated in the analysis.
Identity swap has no artificial intermediate severities; its endpoints are
clean and full replacement.

Clean results are computed once per unordered pair rather than repeated for
each corruption.

### 4.2 Geometry attribution

For every controlled pair-condition and real-deformation pair, compute four
scores on the intersection of their valid pixels:

| Cell | Image features | Geometry |
| --- | --- | --- |
| `s00` | clean/source | clean base |
| `s10` | changed/deformed | clean base |
| `s01` | clean/source | adaptive geometry from changed/deformed images |
| `s11` | changed/deformed | adaptive geometry from changed/deformed images |

Record:

```text
feature_effect  = s10 - s00
geometry_effect = s01 - s00
interaction     = s11 - s10 - s01 + s00
```

The fixed camera pose/intrinsics are the same in all four cells. Here,
`geometry_effect` measures adaptive depth/mask coupling, not camera-pose error.
Camera-pose error is handled by the separate controlled perturbation sweep.

For all clean upper pairs, perturb each view in turn using:

- rotation in degrees: `0.25, 0.5, 1.0, 2.0, 5.0`
- translation as a fraction of median camera radius:
  `0.0025, 0.005, 0.01, 0.02`
- foreground depth scale: `1.01, 1.02, 1.05, 1.10`
- depth-noise standard deviation relative to median foreground depth:
  `0.005, 0.01, 0.02, 0.05`

Only one geometry factor changes at a time. The resulting calibration envelope
answers how large a known geometry perturbation is needed to produce an
observed score change. It does not convert DA3 estimates into ground truth.

### 4.3 Real-deformation stress test

For each shoe, evaluate the six deformed-deformed upper pairs and compare them
with their source-source baselines. Both views are changed in this track.
Compute the same four-cell attribution using clean/base geometry and adaptive
depth inferred from the four deformed views.

Report:

- all 24 pairs
- pairs where both views have `camera_preserved=true`
- semantic-gate and preservation-gate strata
- score change by last-render fit improvement/worsening
- failure cases with framing or preservation violations

This track tests external realism of the controlled findings. It is not used to
fit thresholds or redefine the primary endpoints.

## 5. Feature Backbones and Resolution Tracks

Pin and record these exact backbones:

| Name | Model | Revision |
| --- | --- | --- |
| `dino_vitb16` | `facebookresearch/dino::dino_vitb16` | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| `dinov2_vitb14` | `facebook/dinov2-base` | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| `dinov3_vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| `dinov3_vitl16_pixal3d` | `camenduru/dinov3-vitl16-pretrain-lvd1689m` | `3c276edd87d6f6e569ff0c4400e086807d0f3881` |

Pin NAF to revision
`37f2dfc180f2de53d98bd601109c0da0dd6b0f43` and DA3 to
`depth-anything/DA3NESTED-GIANT-LARGE` revision
`8615eefb62f2db4f8d6ebaa59160086981672829`.

Two resolution tracks remain separate:

1. `user_37_grid_track`
   - DINO, DINOv3-B, and DINOv3-L: 592-pixel input, 37x37 native grid
   - DINOv2: 588-pixel input, 42x42 native grid
   - explicit evaluation grids: 37, 74, 148, 296, 592
2. `current_pixal3d_code_track`
   - Pixal3D DINOv3-L: 512-pixel input, 32x32 native grid
   - explicit targets: 128 and 512

For each explicit target, compute:

- bilinear interpolation
- NAF with the normal RGB guide
- NAF with a blurred guide
- NAF with a spatially shuffled guide

Never label an interpolated or NAF output as native. The design contains 93
feature variants in total.

## 6. Metric and Data Contracts

Use bidirectional known-camera back-projection with:

- relative depth tolerance: `0.03`
- minimum overlap fraction: `0.10`
- confidence percentile: `10`

Persist directional scores and their symmetric average. Each analysis row must
contain:

- stable `design_id`
- shoe, pair, source view, changed view, and direction
- corruption kind, normalized strength, concrete transform parameters, seed,
  and donor where applicable
- feature track, backbone/model revision, input size, feature mode, resolution,
  NAF revision, and guide mode
- `s00`, `s10`, `s01`, `s11`
- feature, geometry, and interaction effects
- overlap and abstention for every cell
- cycle-reprojection error and relative-depth disagreement
- camera/depth/input hashes and DA3 provenance
- real-deformation quality covariates when applicable

If overlap is below `0.10`, the score abstains. Abstentions are never imputed;
they reduce reported coverage.

## 7. PCA Visualization

Raw DINO tensors may exist only in GPU memory, CPU memory, or process-scoped
ephemeral scratch. They must be deleted after scoring and rendering and may not
be copied into the persistent experiment tree.

Fit one deterministic PCA basis per backbone using tokens from all 16 clean
native upper-view features. For Pixal3D DINOv3-L, fit one shared basis across
the 592-input and 512-input clean native features. Use seed 42 for any token
subsampling. Persist only the small PCA parameters: mean, three components, and
fixed component display bounds.

Use the 1st and 99th percentiles of the PCA-fit tokens as fixed RGB bounds.
Never normalize PCA colors per image.

Render PNGs for every evaluated image state at:

- the backbone/track's lowest native resolution
- the track's highest resolution

At the highest resolution, render bilinear, NAF normal, NAF blurred, and NAF
shuffled results. For the real-deformation track, render both source and
deformed inputs with the same backbone basis and display bounds.

Diagnostic score/overlap/cycle/depth maps are saved for all pilot cells. During
the full run they are saved for clean, middle-severity, maximum-severity, and
all abstained/failure cells. Numeric metrics are retained for every cell.

## 8. Pilot

Run the pilot on GPU 4 before the full experiment.

Pilot shoes:

- `air_huarache_run_ultra_111449`
- `air_max_90_surplus_cq7743_300`

Pilot pairs:

- adjacent: `left__back`
- wide baseline: `left__right`

Evaluate both corruption placements for:

- translation: clean, `0.12`, `0.20`
- exposure: clean, `0.50`, `0.75`
- identity swap: clean and `1.00`

Run only the Pixal3D DINOv3-L backbone, but run both the 37/592 and current
32/512 tracks and every planned interpolation/NAF mode in those tracks.

Hard pilot gates:

- zero missing planned jobs
- zero input/checkpoint/provenance mismatch
- zero non-finite metric values on non-abstained cells
- at least 90% clean directional coverage at the `0.10` overlap threshold
- identical PCA basis hashes across all images for a backbone
- deterministic rerun of a sentinel job produces equal IDs and numerically
  equivalent metrics
- no persistent raw DINO feature artifact

Translation and identity swap increasing inconsistency relative to clean, and
exposure being less sensitive, are scientific sanity checks rather than hard
pass/fail criteria. Unexpected scientific results trigger inspection and are
reported; they are not discarded.

Any configuration or code change after the pilot invalidates its use as final
data. Rerun the pilot under the frozen configuration before starting the full
run.

## 9. Full Execution

Use
`/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v1`
as the persistent experiment root.

The run is divided into sequential phases:

1. Manifest, camera mapping, input, and checkpoint audit.
2. GPU-4 pilot and hard-gate validation.
3. DA3 phase with one shoe on each of GPUs 4, 5, 6, and 7.
4. Feature/metric phase:
   - GPU 4: DINO
   - GPU 5: DINOv2
   - GPU 6: DINOv3-B
   - GPU 7: Pixal3D DINOv3-L, both resolution tracks
5. Geometry-only perturbation calibration.
6. Real-deformation stress test.
7. Aggregation, statistical analysis, and report generation.

Do not overlap the DA3 and feature phases. Each feature worker loads its model
once and streams image states, target resolutions, scoring, and PCA rendering.

Expected controlled-fidelity scale:

- 24 unordered clean pairs
- 48 directed corruption placements
- 21 non-clean changed conditions per placement
- 1,032 pair-condition cells after clean deduplication
- 95,976 metric rows across 93 feature variants

Adaptive DA3 artifacts are deduplicated by shoe, changed view, corruption, and
strength. The expected controlled set is four clean four-view geometries plus
336 changed four-view geometries.

## 10. Resumability and Failure Handling

Use an SQLite registry as the authoritative job state. A job is identified by
the hashes of its inputs, frozen configuration, model revisions, and design
coordinates. States are `pending`, `running`, `completed`, `abstained`, and
`failed`.

- Claim jobs transactionally.
- Write result files to a temporary sibling path and atomically rename them.
- Validate file schema and hashes before marking a job complete.
- A resumed run skips only validated completed jobs.
- Retry transient load or I/O failures at most twice.
- On OOM, release tensors, record the error, and retry with lower worker
  concurrency without changing model, resolution, precision, or experiment
  semantics.
- Persistent failures remain explicit in the coverage report.

Store metric results in Parquet for analysis and export CSV summaries. Keep
DA3 geometry, manifests, registry, logs, PCA models, and visualization PNGs.
Do not store raw DINO features.

Measure pilot wall time and bytes by artifact class. If projected persistent
storage exceeds 300 GB, stop before the full run and revise only the
visualization retention policy; do not reduce metric coverage without a new
design decision.

## 11. Statistical Analysis

Backbone score scales are not directly comparable. Compare backbones using
within-backbone clean-normalized deltas and discrimination/ranking endpoints:

- semantic fidelity:
  identity-swap clean/corrupt AUROC and pairwise rank accuracy
- spatial fidelity:
  within-pair Spearman correlation for translation/TPS severity
- nuisance sensitivity:
  clean-normalized exposure/blur/JPEG score change
- resolution utility:
  endpoint change by resolution within backbone and feature mode
- geometry robustness:
  feature effect, geometry effect, interaction, and geometry-perturbation
  envelope
- validity:
  directional coverage, overlap, cycle error, and depth disagreement

Average pair-level endpoints within shoe before averaging shoes, so shoes
receive equal weight. Report:

- shoe-equal-weighted estimates
- shoe-clustered bootstrap intervals with seed 42 and 1,000 repeats
- leave-one-shoe-out ranges
- per-shoe curves and failure cases

With only four shoes, bootstrap intervals are descriptive and discrete.
Do not claim population-level statistical significance or report p-values as
confirmatory evidence.

Primary figures are:

- fidelity versus feature resolution
- corruption severity curves
- backbone discrimination comparison
- bilinear versus NAF guide ablations
- feature/depth-geometry/interaction decomposition
- controlled geometry-perturbation curves
- clean/source versus real-deformation stress deltas
- coverage and abstention plots
- fixed-basis PCA comparison montages

## 12. Later Human-Validation Stage

Human validation is not part of the initial run. When started, raters must not
see metric values, corruption labels, or condition names. They score four
separate dimensions:

1. **Geometric alignment:** silhouette, visible-surface position, occlusion
   boundary, and part placement after accounting for viewpoint. Ignore identity,
   color, sharpness, and realism.
2. **Semantic identity:** shoe model, parts, colorway, materials, logos, laces,
   and sole pattern. Ignore small displacements, blur/compression, and camera
   accuracy.
3. **Local-detail alignment:** seams, logo edges, eyelets, and texture
   boundaries at corresponding surface locations. Merely recognizing the same
   shoe is insufficient.
4. **Single-view plausibility:** artifacts and physical plausibility of each
   image independently. Ignore whether the two views agree.

With one rater, hide and repeat 15% of samples and report intra-rater weighted
kappa. A publishable follow-up should use at least two blinded raters and
report inter-rater agreement.

## 13. Verification

Before the pilot:

- Unit-test upper-only pair enumeration and bidirectional corruption placement.
- Unit-test deterministic balanced corruption parameters.
- Unit-test identity-swap donor mapping, foreground replacement, and mask
  propagation.
- Unit-test four-cell common-support attribution.
- Unit-test feature-variant enumeration and exact count 93.
- Unit-test shared fixed PCA basis and global display bounds.
- Unit-test raw-feature cleanup and persistent-output allowlist.
- Unit-test SQLite claiming, atomic completion, resume, and failed-job behavior.
- Run the existing known-camera suite.

The pilot is the real-model integration test. The full run may begin only after
the hard pilot gates pass under a frozen configuration.

## 14. Deliverables

The experiment root contains:

```text
20260724_upper_v1/
  config/
  manifests/
  registry/jobs.sqlite
  geometry/
  metrics/
  pca/
  visualizations/
  logs/
  reports/
```

Final deliverables are:

- frozen config and provenance manifest
- camera/input audit
- resumable job registry
- fixed/adaptive DA3 geometry artifacts
- complete metric Parquet and summary CSV files
- PCA basis parameters and lowest/highest-resolution PCA PNGs
- diagnostic maps and comparison montages
- statistical tables and figures
- a report stating the fixed/precomputed-camera limitation and separating
  controlled fidelity, geometry calibration, and real-deformation stress
  conclusions
