# MEt3R DINOv2 Native-Grid Design

## Goal

Remove the invalid DINOv2 `42→37` NAF condition from the full upper-view
experiment. DINOv2 uses its native `42×42` feature grid as its lowest
resolution; it is not forced onto the `37×37` grid.

## Root Cause

The frozen user track lists common targets `37,74,148,296,592`. DINO, DINOv3,
and Pixal3D DINOv3-L have native `37×37` grids in that track, but DINOv2 has a
native `42×42` grid. Passing a `37×37` output target to NAF makes its spatial
dilation `37 // 42 = 0`, which NATTEN correctly rejects because dilation must
be positive.

This is a protocol-design error rather than a GPU, camera, or transient
runtime failure.

## Resolution Policy

Generated feature targets must not be smaller than the backbone's native
feature grid:

```text
retain target iff target_resolution >= native_grid
```

This rule removes all four generated DINOv2 `37×37` variants:

- `bilinear__37`;
- `naf_normal_guide__37`;
- `naf_blurred_guide__37`;
- `naf_shuffled_guide__37`.

DINOv2 retains:

- native `42×42`;
- bilinear and three NAF modes at `74,148,296,592`.

The other backbone tracks are unchanged because their common `37×37` target
is equal to, not smaller than, their native `37×37` grid. The Pixal3D current
track is unchanged.

## Scientific Interpretation

- Lowest-resolution semantic evidence is compared at each model's native
  grid: `37×37` for DINO/DINOv3 and `42×42` for DINOv2.
- Direct cross-backbone comparisons of generated modes use only the shared
  valid targets `74,148,296,592`.
- NAF comparisons never claim a DINOv2 `37×37` result.
- Native-grid comparisons must identify the small `37` versus `42` resolution
  difference as a limitation rather than attributing every difference to the
  backbone.
- Cameras remain labeled
  `fixed/precomputed DA3 cameras (not calibrated GT)`.

## Revised Counts

The full design changes from 93 to 89 feature variants. Job counts remain
unchanged because a feature job owns all variants for one state/backbone
track:

- geometry jobs: 344;
- feature jobs: 1,740.

Expected metric rows become:

- controlled fidelity: `89 × 1,032 = 91,848`;
- geometry calibration: `89 × 816 = 72,624`;
- real-deformation stress: `89 × 24 = 2,136`.

The GPU-1 pilot is unchanged because it contains only DINOv3-L tracks whose
targets are not below their native grids.

## Recovery and Provenance

The interrupted full root is preserved under a failure-specific sibling name;
it is not reused because its immutable variant manifest contains the invalid
93-variant design and its run manifest predates this correction.

A fresh full root is prepared from the corrected code. It records the new git
commit and the already-approved physical GPU runtime mapping `0,1,2,3`.

## Testing

Use test-driven development:

1. Feature enumeration must contain 89 full variants.
2. No DINOv2 generated variant may have resolution below 42.
3. DINOv2 must contain native 42 and generated targets
   `74,148,296,592` for all four modes.
4. Pilot design and its 30 variants must remain unchanged.
5. Streaming feature generation must reject a below-native NAF variant
   defensively if an invalid external manifest bypasses design enumeration.
6. Update exact report/runbook counts, run focused tests, run the complete
   test suite, and restart from a clean full root.
