# Known-camera DINO fidelity experiment

This extension evaluates whether DINO-family feature differences are a useful
proxy for multi-view inconsistency, and whether feature resolution separates
semantic agreement from pixel-level agreement. It replaces DUSt3R/MASt3R
alignment with fixed, precomputed camera poses and explicitly audits the
remaining geometry uncertainty.

## What is implemented

- Pure PyTorch known-camera reprojection with depth visibility checks,
  bidirectional scoring, overlap-based abstention, feature-error maps, relative
  depth error, and cycle reprojection error.
- A common-support 2×2 decomposition:

  | score | image features | geometry |
  | --- | --- | --- |
  | `s00` | clean | base/fixed |
  | `s10` | changed | base/fixed |
  | `s01` | clean | adaptive/re-estimated |
  | `s11` | changed | adaptive/re-estimated |

  `feature_effect = s10 - s00`, `geometry_effect = s01 - s00`, and
  `interaction = s11 - s10 - s01 + s00`. All four cells are recomputed on the
  intersection of their valid pixels, so support changes cannot masquerade as
  a feature effect.
- DINO, DINOv2, DINOv3, and the exact Pixal3D DINOv3-L backbone adapter. Every
  backbone revision, the NAF repository revision, and the DA3 model revision
  are pinned and recorded in generated artifacts.
- Native, bilinear, and NAF-upsampled features. NAF is also run with normal,
  blurred, and shuffled RGB guides. These controls reveal whether apparent
  high-resolution fidelity comes from DINO features or from guide-image detail
  injected by NAF.
- Fixed-camera DA3 depth inference with saved model/camera provenance and a
  returned-camera contract check confirming that the aligned API output exposes
  the supplied poses/intrinsics.
- Controlled translation, thin-plate warp, exposure, blur, JPEG, and
  part-transfer corruptions.
- Known-magnitude camera/depth perturbations for a geometry-only sensitivity
  curve.
- A config-driven design-matrix generator, analysis-ready rows, shoe-clustered
  bootstrap summaries, score/severity plots, correspondence diagnostic
  figures, and a blinded human-rating packet.

## Camera scope and the GT limitation

The saved cameras are **fixed/precomputed DA3 cameras (not calibrated GT)**.
The report and metadata use exactly that wording.

The ByteSize upper and bottom image sets were reconstructed separately. The
notebook aligned their exported GLB point clouds, but GLB export first applies
the first-camera transform and a point-cloud-dependent centering translation.
The notebook saved the ICP transform but not both centering transforms.
Consequently:

- Pairs within `upper` or within `bottom` have a shared DA3 coordinate system
  and are the primary analysis.
- Cross-group rotation is recoverable from the documented coordinate
  conversion.
- Cross-group absolute translation is not exactly recoverable from the stored
  artifacts. The loader retains a translation proxy and labels all such pairs
  `sensitivity_only`; they must not be pooled into the primary result.

This is why simply calling the poses “known cameras” would overstate the
evidence. The 2×2 decomposition distinguishes two observable effects within a
fixed-pose setting: changing features while holding precomputed geometry fixed,
and changing adaptive depth while holding images fixed. The synthetic
pose/depth sweep then maps known perturbation magnitudes to score changes. It
does not identify the unknown DA3 error for an individual pair, but reveals
whether an observed feature effect is larger than plausible geometry-induced
changes. Report factorial effects, the perturbation envelope, overlap, and
cycle/depth diagnostics together. None of these turns DA3 estimates into GT.

Run the audit first:

```bash
python -m met3r.cli audit-cameras \
  /root/dev/ByteSize/Reconstruction \
  outputs/camera_audit.json
```

The current data audit should discover 4 shoes and 24 selected views. It checks
that each zero-based DA3 camera index matches the correct one-based source/mask
image using grayscale correlation, and that the notebook transform is constant
over all 48 bottom cameras.

## Resolution tracks

The requested 37×37 track uses 592×592 inputs for patch-16 DINO/DINOv3. DINOv2
uses patch size 14, so a 588×588 input gives its native 42×42 grid; all
backbones are then evaluated on common explicit grids. Do not call a bilinear
or NAF grid “native.”

There is an important repository discrepancy: the current Pixal3D code uses
`image_size=512`, so its DINOv3 native grid is 32×32, not 37×37, and its NAF
targets vary by stage (for example 128 or 512). The supplied config therefore
contains both:

- `user_37_grid_track`, which tests the stated 37×37 hypothesis; and
- `current_pixal3d_code_track`, which reproduces the checked-out implementation.

Report them separately. This avoids silently attributing an input-resolution
change to the backbone version.

## Recommended controlled protocol

For every shoe and eligible pair:

1. Keep one view clean and corrupt the other at five ordered severity levels.
   For translation and thin-plate warp, use the CLI foreground-mask options so
   the adaptive geometry artifact receives the transformed mask rather than the
   clean silhouette.
2. Use geometry-preserving corruptions (exposure, blur, JPEG) to measure
   invariance, spatial corruptions (translation, thin-plate warp) to measure
   localization sensitivity, and part transfer to measure semantic identity.
3. Extract each backbone once per image set and cache the NPZ artifact.
4. Compute base depth once from clean images with the fixed poses. Compute
   adaptive depth from the changed images with the *same* poses.
5. Record all four factorial scores, both effects, the interaction, minimum
   bidirectional overlap, cycle error, depth disagreement, and abstention.
6. Run the same score with known pose/depth perturbations. Match the observed
   geometry effect to this calibration curve instead of interpreting every
   score increase as a DINO response.
7. Keep `intra_group` and `cross_group` results in separate tables.
   Also retain `camera_group` so the four-view upper geometry is not silently
   pooled with the two-view bottom geometry.

The primary fidelity endpoints are:

- Spearman correlation between controlled severity and inconsistency score,
  computed within pair, averaged within shoe, then averaged across shoes;
- clean-versus-corrupt ROC AUC;
- within-shoe pairwise severity ranking accuracy;
- Spearman correlation with the negative human consistency rating;
- valid-pair coverage and overlap.

Use shoe-clustered bootstrap intervals for fidelity and feature/geometry
attribution, because views and severity levels from one shoe are not
independent. Select thresholds on a development split and report final
estimates on a held-out shoe/object split when the dataset is expanded. With
only four shoes, emphasize confidence intervals and individual shoe plots
rather than significance claims.

## Human-rating dimensions

`python -m met3r.cli make-ratings samples.json ratings/` creates randomized
tasks, a rubric, re-encoded metadata-free stimuli, and an unblinding key outside
the rater packet directory. The task sheet never contains metric values or
condition names.

The four scores deliberately target different evidence:

1. **Geometric alignment**: silhouette, visible surface, occlusion boundary,
   and part position after accounting for the camera change. Ignore color,
   logo identity, sharpness, and realism.
2. **Semantic identity**: model, part inventory, colorway, materials, logos,
   laces, and sole pattern. Ignore small displacement, blur/compression, and
   camera accuracy.
3. **Local detail alignment**: whether seams, logo edges, lace holes, and
   texture boundaries occur at corresponding surface locations. Recognizing
   the same shoe is insufficient.
4. **Single-view plausibility**: artifacts and physical plausibility of each
   image independently. Ignore whether the two views agree.

Each dimension has separate 1/3/5 anchors in the generated `RUBRIC.md`. With
one rater, repeat a hidden 10–15% subset to measure intra-rater weighted kappa.
For a publishable study, use at least two blinded raters and report
inter-rater agreement before averaging.

## CLI workflow

Install the lightweight extension dependencies without requiring the legacy
PyTorch3D/FeatUp stack. Run the module from this checkout; a full package
installation still includes the original MEt3R stack:

```bash
pip install -r requirements-known-camera.txt
```

The extension requires Python 3.10 or newer; package version 1.1.0 records that
compatibility boundary explicitly.

Discover the fixed six-view records:

```bash
python -m met3r.cli prepare-manifest \
  /root/dev/ByteSize/Reconstruction outputs/manifest.json
```

Expand the checked-in configuration into a deterministic audit-only table. This
enumerates every shoe/pair, primary versus sensitivity tier, backbone,
resolution/mode, corruption, and severity cell:

```bash
python -m met3r.cli build-design \
  configs/known_camera_bytesize.json outputs/manifest.json \
  outputs/experiment_design.csv
```

The table records view indices, changed-view index, corruption seed, and
planned state, but deliberately does not claim to be an executor: part-transfer
donors/masks and concrete artifact paths must be registered by the experiment
orchestrator before a row is run. Treat it as a coverage audit and join results
back by `design_id`.

Materialize ready-to-use `images.json` and `cameras.npz` files. Primary
artifacts are written separately for upper and bottom; the six-view
cross-group artifact is placed under `sensitivity/` with proxy provenance:

```bash
python -m met3r.cli prepare-camera-artifacts \
  /root/dev/ByteSize/Reconstruction outputs/cameras
```

This command stages the saved 504×504 DA3 processed RGB images next to their
504×504 intrinsics and alpha masks. It intentionally does not feed the original
1080×1080 images with already-resized intrinsics.

Extract a 37-grid DINOv3 track with interpolation and NAF controls:

```bash
python -m met3r.cli extract-features clean_images.json outputs/features/clean \
  --backbone dinov3_vitl16_pixal3d --input-size 592 \
  --bilinear-scales 37,74,148,296,592 \
  --naf-scales 74,148,296,592 --device cuda
```

Run fixed-pose DA3 depth. `cameras.npz` must contain `extrinsics` and
`intrinsics`; it should also contain a boolean `mask` at DA3 output resolution
so white shoe regions are not confused with the white background. Without it,
the adapter records that it fell back to positive finite depth. The provenance
string must describe whether the artifact is an intra-group pose set or the
cross-group proxy:

```bash
python -m met3r.cli infer-depth clean_images.json cameras.npz outputs/depth/base \
  --da3-repository /root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3 \
  --camera-provenance fixed_precomputed_da3 --device cuda
```

For spatial corruptions, transform the foreground mask with the image and pass
it to the corresponding adaptive-depth run:

```bash
python -m met3r.cli corrupt clean.png changed.png \
  --kind translation --strength 0.12 \
  --foreground-mask clean-mask.png --output-mask changed-mask.png

python -m met3r.cli infer-depth changed_images.json cameras.npz outputs/depth/adaptive \
  --mask-override 1=changed-mask.png \
  --da3-repository /root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3 \
  --camera-provenance fixed_precomputed_da3 --device cuda
```

`--mask-override` is repeatable and fails if the base camera artifact has no
mask. Feature and depth artifacts store a content hash for the ordered image
set; strict scoring refuses clean/base or changed/adaptive combinations whose
content identities do not match. They also store runtime package fingerprints;
local DA3 runs additionally record the checkout commit and dirty state. The
content hash of tracked and untracked changes disambiguates dirty checkouts at
the same commit. Runtime artifacts record the environment provenance observed
at execution time; any environment manifest is a runtime artifact rather than
a committed experiment input.

Score a pair across all feature keys:

```bash
python -m met3r.cli score-artifacts \
  outputs/features/clean/features.npz \
  outputs/features/changed/features.npz \
  outputs/depth/base/depth_geometry.npz \
  outputs/depth/adaptive/depth_geometry.npz \
  outputs/results/pair.csv \
  --first-index 0 --second-index 1 --metadata-json pair_metadata.json
```

Materialize geometry-only calibration variants. Relative translation is
normalized by the median camera radius, and camera rotations preserve the
original camera center:

```bash
python -m met3r.cli calibrate-geometry \
  outputs/depth/base/depth_geometry.npz outputs/depth/calibration \
  --view-index 1 --rotations 0.25,0.5,1,2,5 \
  --relative-translations 0.0025,0.005,0.01,0.02 \
  --depth-scales 1.01,1.02,1.05,1.1 \
  --relative-depth-noise 0.005,0.01,0.02,0.05
```

Score each generated geometry file through the dedicated path (pose changes are
intentionally rejected by the normal fixed-pose factorial scorer):

```bash
python -m met3r.cli score-geometry-sensitivity \
  outputs/features/clean/features.npz \
  outputs/depth/base/depth_geometry.npz \
  outputs/depth/calibration/rotation_5/geometry.npz \
  outputs/results/rotation_5.csv \
  --first-index 0 --second-index 1 --metadata-json rotation_5.json
```

This directly measures the score response attributable to each known geometry
error on common support.

Concatenate pair CSVs and retain `severity`, `corruption`, `backbone`,
`shoe_id`, `pair_id`, `pair_scope`, and `analysis_tier`. After unblinding,
join any of the four rating columns by task key; they are analyzed separately
rather than collapsed into a generic consistency score:

```bash
python -m met3r.cli analyze outputs/results/all.csv outputs/analysis \
  --bootstrap-repeats 1000
```

The output includes `fidelity_summary.csv`, `attribution_summary.csv`, and
`score_vs_severity.png`. Use `met3r.visualization.save_direction_diagnostics`
to save per-pair score/support/depth/cycle maps for qualitative failure review.

## Decision rule for the reliability module

The experiments support a learnable view-reliability module only if:

- the metric retains ordered sensitivity across held-out shoes;
- low-resolution features track semantic identity while being relatively
  invariant to small geometric/photometric changes;
- high-resolution/NAF features add local-alignment sensitivity, and that gain
  survives blurred/shuffled-guide controls;
- feature effects are larger than geometry effects at usable overlap; and
- abstention/overlap predicts known failure cases.

These results can supervise two distinct quantities per view: a view weight
and a resolution gate. Avoid collapsing them into one scalar: a view may be
semantically reliable but unsafe for pixel-level back-projection.
