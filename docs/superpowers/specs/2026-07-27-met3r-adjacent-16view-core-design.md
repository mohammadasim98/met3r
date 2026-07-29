# MEt3R Adjacent 16-View Core Experiment Design

**Date:** 2026-07-27

**Status:** User-approved design, pending written-spec review

**Experiment ID:** `20260727_adjacent16_core_v1`

**Output root:**
`/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1`

## 1. Objective and Research Claims

This experiment replaces the sparse `left/back/right/front` four-view set with
four azimuth-balanced quartets of adjacent upper views. It is intentionally
limited to the evidence needed for these four claims:

1. At a common low feature resolution, DINO, DINOv2, DINOv3-B, and the
   Pixal3D DINOv3-L backbone differ in their ability to measure controlled
   multi-view inconsistency.
2. For the Pixal3D DINOv3-L backbone, feature resolution changes semantic and
   spatial inconsistency fidelity, and the relationship need not be monotonic.
3. NAF-normal upsampling and bilinear interpolation can be compared on the
   same backbone, image states, camera pairs, and target grids.
4. When adjacent views provide sufficient geometric support, overlap and
   low/high-resolution feature evidence can support a View Reliability
   Module that preserves semantic conditioning while gating unreliable
   high-resolution pixel-aligned conditioning.

The saved cameras are DA3 estimates, not calibrated ground truth. Every
artifact and report must describe them as:

> fixed/precomputed DA3 cameras (not calibrated GT)

The experiment measures feature fidelity conditional on this geometry and
local sensitivity to injected geometry errors. It does not identify the
unknown absolute DA3 camera error.

## 2. Scope

### 2.1 Included

- Four shoes:
  - `air_huarache_run_ultra_111449`
  - `air_max_90_nrg_dc6083_500`
  - `air_max_90_surplus_cq7743_300`
  - `air_tuned_max_cv6984_001`
- Sixteen upper views per shoe, organized as four disjoint adjacent quartets.
- All six unordered pairs within each quartet.
- Angular-baseline analysis at approximately `7.5`, `15`, and `22.5` degrees.
- Twelve feature variants.
- Four controlled corruptions.
- A targeted two-variant geometry calibration.
- An independent real-deformation stress track for the same 16 views.
- Numeric metrics for every registered condition.
- PCA and directional diagnostics only for the predeclared audit subset.

### 2.2 Excluded

- Bottom views and upper-to-bottom pairs.
- Pairs across different quartets.
- DINO native-grid and current-Pixal3D-code-track variants outside the frozen
  12-variant matrix.
- NAF blurred-guide and shuffled-guide ablations.
- Exposure and blur corruptions.
- Full geometry sweeps for every backbone and resolution.
- Human rating.
- Raw DINO feature persistence.
- Claims of confirmatory population-level significance from four shoes.

The prior `20260724_upper_v2` experiment remains immutable evidence for the
sparse-view failure regime and is not overwritten or pooled silently with this
experiment.

## 3. Input Data and Frozen View Selection

### 3.1 Source images

The clean upper views come from:

```text
/root/dev/ByteSize/Reconstruction/sample_result/
  {MODEL_DISPLAY_NAME}/upper/glb_masked_white_bg/masked_inputs/{VIEW_ID}.jpg
/root/dev/ByteSize/Reconstruction/sample_result/
  {MODEL_DISPLAY_NAME}/upper/mask/{VIEW_ID_PLUS_ONE}.jpg
```

RGB and camera array indices are zero-based `0`--`47`, while mask filenames
are one-based `1`--`48`. The only valid mapping is
`VIEW_ID_PLUS_ONE = VIEW_ID + 1`; filename equality must not be assumed.

The corresponding fixed/precomputed DA3 camera arrays are:

```text
.../upper/glb_masked_white_bg/outputs/intrinsics.npz
.../upper/glb_masked_white_bg/outputs/extrinsics.npz
.../upper/glb_masked_white_bg/outputs/processed_images.npz
```

The shoe slug to display-name mapping remains:

```text
air_huarache_run_ultra_111449
  -> AIR HUARACHE RUN ULTRA(111449)
air_max_90_nrg_dc6083_500
  -> AIR MAX 90 NRG(DC6083-500)
air_max_90_surplus_cq7743_300
  -> AIR MAX 90 SURPLUS(CQ7743-300)
air_tuned_max_cv6984_001
  -> AIR TUNED MAX(CV6984-001)
```

Every shoe has 48 source images, 48 one-based masks, and 48 matching camera
entries. The pipeline must validate exact image/mask/camera counts, the
zero-to-one-based index mapping, image decodability, camera finiteness, and
source-to-processed-image correspondence before creating any feature job.

The masked source images are 1080-by-1080, while the fixed/precomputed DA3
intrinsics and processed images use a 504-by-504 camera contract. Stage clean,
controlled, and deformed RGB at 504-by-504 with Lanczos resampling; stage masks
with nearest-neighbor resampling; and force non-foreground RGB pixels to white.
The preflight must require grayscale correlation of at least 0.99 between each
staged clean image and its indexed DA3 processed image. Feature backbones resize
the staged 504-by-504 RGB independently to their registered model input size.
Never combine the 1080-by-1080 source directly with 504-by-504 intrinsics.

### 3.2 Camera order

View IDs must not be sorted numerically. Camera azimuth is computed from the
extrinsics, and the circular order must match this frozen sequence for all
four shoes:

```text
0, 11, 22, 33, 43, 44, 45, 46, 47,
1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
34, 35, 36, 37, 38, 39, 40, 41, 42
```

The observed median adjacent-camera separation is approximately `7.5`
degrees. An order mismatch or an adjacent separation outside the audited
`5`--`10` degree orbit range blocks the experiment rather than silently
selecting replacement views.

### 3.3 Frozen quartets

Four starts spaced by 12 positions on the 48-view orbit define the quartets:

| Quartet | View IDs | Within-quartet offsets |
| --- | --- | --- |
| `q0` | `1, 2, 3, 4` | 1, 2, 3 steps |
| `q1` | `14, 15, 16, 17` | 1, 2, 3 steps |
| `q2` | `27, 28, 29, 30` | 1, 2, 3 steps |
| `q3` | `40, 41, 42, 0` | 1, 2, 3 steps |

Each quartet contributes six unordered pairs:

- three one-step pairs, approximately `7.5` degrees;
- two two-step pairs, approximately `15` degrees;
- one three-step pair, approximately `22.5` degrees.

Therefore each shoe contributes `4 × 6 = 24` pair units, and the four shoes
contribute 96 pair units. Quartet and baseline-step labels are mandatory
analysis columns.

The quartet choice is frozen from camera azimuth before inspecting new DINO
scores. It must not be replaced with the highest-overlap quartet after metric
inspection.

## 4. Geometry Preflight

Run geometry-only preflight for the selected 64 clean images before scheduling
the feature experiment. Recompute or materialize selected-view depth and
foreground validity masks because the source tree contains visualized depth
images but not a persistent raw depth tensor suitable for scoring.

Use:

- relative depth tolerance: `0.03`;
- minimum directional overlap: `0.08`;
- confidence percentile: `10`;
- symmetric validity: both directions must pass;
- camera label: fixed/precomputed DA3 cameras (not calibrated GT).

Record, at minimum:

- first-to-second and second-to-first overlap;
- `min(o12, o21)`;
- overlap asymmetry `abs(o12 - o21)`;
- common-valid fraction;
- median cycle-reprojection error;
- median relative-depth disagreement;
- quartet and angular-baseline step.

The full feature run is allowed only when:

1. at least 90% of the 96 clean pair units pass symmetric 8% overlap;
2. the median `min(o12, o21)` across the 96 pairs is at least 20%;
3. every shoe has at least 80% clean symmetric pair coverage;
4. no camera/image/hash validation fails.

These thresholds are frozen before new feature scores are inspected. Failure
requires a numbered protocol revision; it must not be repaired by post-hoc
overlap-threshold tuning.

## 5. Feature Matrix

Use the pinned model and code revisions from `20260724_upper_v2`:

| Name | Revision |
| --- | --- |
| DINO ViT-B/16 | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| DINOv2 ViT-B/14 | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| DINOv3 ViT-B/16 | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| Pixal3D DINOv3-L | `3c276edd87d6f6e569ff0c4400e086807d0f3881` |
| NAF | `37f2dfc180f2de53d98bd601109c0da0dd6b0f43` |
| DA3 | `8615eefb62f2db4f8d6ebaa59160086981672829` |

Only the `user_37_grid_track` is used.

### 5.1 Common low-resolution backbone comparison

Evaluate these four variants at the common 37-by-37 target grid:

| Backbone | Mode | Resolution |
| --- | --- | ---: |
| DINO ViT-B/16 | bilinear | 37 |
| DINOv2 ViT-B/14 | bilinear | 37 |
| DINOv3 ViT-B/16 | bilinear | 37 |
| Pixal3D DINOv3-L | bilinear | 37 |

DINOv2's native 42-by-42 grid is explicitly resampled to 37-by-37 and must
not be labeled native.

### 5.2 Pixal3D resolution and NAF comparison

For Pixal3D DINOv3-L:

| Resolution | Bilinear | NAF normal |
| ---: | :---: | :---: |
| 37 | yes | no |
| 74 | yes | yes |
| 148 | yes | yes |
| 296 | yes | yes |
| 592 | yes | yes |

This contributes nine DINOv3-L variants. Together with the three B-backbone
37-grid variants, the frozen total is 12 feature variants.

No other backbone, resolution, feature mode, guide mode, or input-size track
may be added without changing the experiment ID.

## 6. Controlled Fidelity Track

For each of the 96 pair units, apply every non-clean change to each member of
the pair in turn. A changed image state is deterministic under seed 42 and is
reused across the three within-quartet pairs that contain the view.

### 6.1 Frozen corruption matrix

| Corruption | Non-zero normalized strengths | Research role |
| --- | --- | --- |
| Translation | `0.03, 0.06, 0.12, 0.20` | global pixel misalignment |
| Thin-plate warp | `0.25, 0.50, 0.75, 1.00` | non-rigid spatial inconsistency |
| JPEG | `0.25, 0.50, 0.75, 1.00` | semantic-preserving nuisance |
| Identity swap | `1.00` | semantic identity inconsistency |

Translation direction and TPS seed are assigned by shoe, changed view,
corruption, and strength, not by pair. Translation and TPS transform the
foreground mask with the image.

Identity swap uses the same view ID from the cyclic donor shoe mapping already
defined by the prior experiment. It remains an upper-bound semantic
corruption because shoe shape can change with identity.

Exposure and blur are intentionally excluded.

### 6.2 Primary controlled geometry

The primary controlled endpoint uses clean/base fixed-precomputed geometry:

```text
feature effect = s10 - s00
```

This isolates the metric response to the registered image change under a
frozen projection contract. Adaptive depth/mask attribution remains secondary
and is run only in the targeted geometry track.

### 6.3 Exact controlled count

Per feature variant:

- clean: `96`;
- translation: `96 × 2 placements × 4 strengths = 768`;
- thin-plate warp: `768`;
- JPEG: `768`;
- identity swap: `96 × 2 placements = 192`;
- total: `2,592`.

Across 12 variants, the controlled track contains exactly 31,104 condition
rows.

## 7. Targeted Geometry Calibration

Run the existing geometry perturbation envelope only for:

1. Pixal3D DINOv3-L bilinear 37;
2. Pixal3D DINOv3-L NAF-normal 592.

Use all 96 clean pair units, perturb each member in turn, and retain the
existing frozen perturbation levels:

- camera rotation: `0.25, 0.5, 1.0, 2.0, 5.0` degrees;
- camera translation: `0.0025, 0.005, 0.01, 0.02` of median camera radius;
- foreground depth scale: `1.01, 1.02, 1.05, 1.10`;
- relative depth-noise standard deviation: `0.005, 0.01, 0.02, 0.05`.

There are 17 non-zero perturbation conditions. The exact geometry count is:

```text
96 pairs × 2 perturbed placements × 17 conditions × 2 variants = 6,528 rows
```

The geometry envelope measures local sensitivity around fixed/precomputed DA3
geometry. It does not estimate absolute camera error or convert DA3 cameras
into calibrated ground truth.

## 8. Real-Deformation Stress Track

Create one independently deformed image for each of the 64 selected clean
views. The deformation stage must preserve the registered view, crop, and
camera framing, while remaining independent across views. Camera preservation
is a pre-metric input requirement, not a post-hoc analysis stratum. A
deformation that fails the existing camera/crop-preservation audit is not
scheduled for metric scoring and must be regenerated under the same frozen
deformation contract. The same source/deformed audit metadata and hashes used
by the prior stress track must be retained.

For each feature variant, score the six deformed-deformed pairs within every
quartet against the corresponding source-source baselines. Both views are
changed. Persist:

- fixed/base score and overlap;
- adaptive depth/mask score and overlap;
- feature, adaptive-geometry, and interaction effects when common support
  exists;
- directional and symmetric abstention;
- quartet and angular-baseline step.

The exact stress count is:

```text
96 pair units × 12 feature variants = 1,152 rows
```

Primary stress evidence is:

1. symmetric pass rate at the frozen 8% threshold;
2. valid low-resolution and high-resolution score change;
3. overlap asymmetry and cycle error;
4. disagreement between 37-bilinear and 592-NAF evidence.

No post-hoc quality strata or human labels are added to the primary analysis.
After the deformation inputs pass their camera-preservation contract, any
metric abstention is evidence for the reliability gate and is never imputed.

The 12--20 hour compute budget below begins after these 64 deformed inputs and
their metadata exist. Deformation generation latency is reported separately
because it is not GPU metric-pipeline time.

## 9. Primary Endpoints and Analysis

### 9.1 Support

- clean symmetric pair coverage;
- controlled complete-series coverage;
- stress symmetric pass rate;
- directional overlap asymmetry;
- finite-shoe count per endpoint.

### 9.2 Backbone comparison

Compare the four common 37-grid variants only on the exact common-support
intersection of:

```text
shoe × quartet × pair × changed view × corruption × strength
```

Primary summaries:

- identity-swap clean/corrupt AUROC;
- translation and TPS severity Spearman;
- translation and TPS pairwise rank accuracy;
- JPEG signed and absolute nuisance delta.

### 9.3 Resolution and NAF comparison

For Pixal3D DINOv3-L:

- compare the five bilinear resolutions on paired common support;
- compare bilinear versus NAF-normal at 74, 148, 296, and 592;
- report paired endpoint differences rather than unrelated marginal means;
- stratify by one-, two-, and three-step angular baselines.

### 9.4 Geometry and stress

- compare the low/high local geometry-response envelopes;
- report whether real-deformation score evidence remains finite at low and
  high resolution;
- treat high-resolution abstention with finite low-resolution evidence as the
  direct View Reliability Module use case.

All bootstrap intervals are shoe-clustered. Quartet, pair, placement, and
severity are repeated measurements, not independent shoes. With four shoes,
all intervals and rankings remain descriptive.

## 10. Analyzability Criteria

These criteria determine whether the experiment supports the planned
comparisons; they do not force a desired scientific result.

1. Geometry preflight passes the criteria in Section 4.
2. Controlled complete-series coverage is at least 80%.
3. At least 80% of primary summary cells have finite endpoints for all four
   shoes.
4. The common-support intersection retains at least 70% of otherwise valid
   rows for each primary comparison.
5. The audited real-deformation inputs achieve at least 70% symmetric pass at
   the frozen 8% threshold.
6. Registry jobs have zero unexplained failures and exact expected row counts.

If criteria 2--5 fail, report the failure and the corresponding reliability
boundary. Do not lower the overlap threshold, remove difficult shoes, or
replace quartets after inspecting feature scores.

## 11. Visualization and Storage

Persist all numeric metric rows. Do not persist raw DINO feature tensors.

Save PCA only for:

- Pixal3D DINOv3-L bilinear 37;
- Pixal3D DINOv3-L NAF-normal 592.

Use one deterministic shared PCA basis and fixed display bounds per backbone
contract. Do not normalize colors per image.

Directional diagnostic PNGs are limited to a deterministic audit manifest:

- clean geometry preflight;
- representative valid and abstained cases;
- minimum, median, and maximum overlap cases per angular-baseline step;
- maximum-severity translation, TPS, JPEG, and identity cases;
- representative real-deformation low/high disagreement.

Do not save diagnostics for every condition row. The final report contains
only five core figure groups:

1. adjacent-view support and coverage;
2. common-37-grid backbone comparison;
3. DINOv3-L resolution and NAF comparison;
4. low/high geometry calibration;
5. real-deformation reliability stress.

## 12. Execution Budget and Reproducibility

Run on physical GPUs 0--3. Before the full run, a timed sentinel must project
the total metric-pipeline wall time at no more than 24 hours. The planned
budget is:

- geometry/input preflight: 1--2 hours;
- controlled feature track: 8--12 hours;
- targeted geometry and stress: 2--4 hours;
- aggregation, validation, and report generation: 1--2 hours;
- expected total after stress inputs exist: 12--20 hours.

The frozen exact primary row count is:

```text
controlled 31,104
geometry    6,528
stress      1,152
total      38,784
```

The run must:

- use content-addressed, resumable jobs;
- record source commit, configuration hash, model revisions, camera/input
  hashes, and timestamps;
- preserve NaN abstentions;
- emit no raw feature artifacts;
- validate aggregate hashes by a repeated aggregate/analyze pass;
- leave `20260724_upper_v2` unchanged.

If the timed sentinel projects more than 24 hours, the full run remains
blocked and requires a user-approved scope revision. The implementation must
not silently reduce shoes, quartets, corruptions, severities, or primary
feature variants.

## 13. Required Final Interpretation

The report must keep these two regimes separate:

1. **Conditional feature fidelity:** how backbone, resolution, and NAF behave
   when adjacent views provide sufficient fixed/precomputed DA3 support.
2. **Reliability boundary:** when geometry, visibility, or independent
   deformation makes high-resolution pixel-aligned evidence invalid while
   low-resolution semantic evidence remains usable.

The experiment can support View Reliability Module design, but it does not
train or validate the learnable module itself. A larger-shoe study is required
before claiming population-level backbone superiority.
