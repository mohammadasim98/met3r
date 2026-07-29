# Task 9 Report: Scale-Safe Upper-View Analysis

## Scope and baseline

- Worktree: `/root/dev/met3r-worktrees/known-camera-met3r`
- Branch: `feature/known-camera-met3r`
- Required and verified baseline:
  `b06cec2268561e9712a3936e895356c13ef223bc`
- The protected pre-existing untracked
  `configs/environment_pixal3d.json` was not read, modified, staged, or
  committed.
- No real DA3/DINO GPU experiment was run in Task 9.

## TDD evidence

### RED

The Task 9 endpoint, output, CLI, and launcher tests were written before the
production analysis module:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_analysis.py tests/test_upper_cli.py -q

ERROR tests/test_upper_analysis.py
ModuleNotFoundError: No module named 'met3r.upper_analysis'
1 error in 2.69s
```

### First GREEN cycle and corrections

The first implementation run reached 13 passing tests and exposed three
contract errors:

```text
3 failed, 13 passed in 3.42s
```

- The in-memory severity-zero baseline incorrectly contributed to coverage,
  yielding `0.5` when the sole observed corruption row abstained.
- Two stress-only synthetic frames had no `corruption` column even though
  controlled clean normalization requires one.

Coverage and overlap now use observed cells only, while the synthetic zero
remains available exclusively to the endpoint calculation. Stress-only
subsets receive the explicit analysis label `deformed`.

The immediate rerun was:

```text
16 passed in 5.30s
```

The richer final Task 9 test adds native, bilinear, NAF normal-guide,
blurred-guide, and shuffled-guide variants and verifies paired contrast
intervals:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_analysis.py -q

7 passed in 9.95s
```

## Statistical implementation

- The primary fixed-geometry score is `s10`; its effect is exactly
  `s10 - s00`. `s11` is retained only as the adaptive-depth secondary score.
- Clean center and scale are calculated independently within exact
  `(backbone, feature_track, feature_mode, resolution)` groups. The scale is
  MAD-based, with an IQR fallback and a finite `1e-6` floor.
- One severity-zero observation is created in memory for each nonclean
  controlled `(shoe, pair, changed view, exact feature variant, corruption)`
  series. It is never written to an aggregate metric Parquet.
- Endpoints are first calculated per changed-view placement, then placements
  are averaged within pair, pairs within shoe, and shoes equally.
- Identity swap reports clean/corrupt AUROC and rank accuracy but no severity
  Spearman endpoint. Continuous corruptions report within-series Spearman and
  rank accuracy. Exposure, blur, and JPEG retain both signed and absolute
  clean-normalized effects.
- Directional `s10` overlap, abstention coverage, feature, geometry,
  interaction, adaptive total, and absolute feature share remain explicit.
- Resolution and NAF contrasts are paired within shoe before aggregation and
  bootstrap. The NAF normal guide is compared separately with bilinear,
  blurred-guide, and shuffled-guide endpoints.
- Geometry calibration reports the interpolated first crossing of the
  monotone absolute-effect envelope, with raw/effective units and explicit
  coverage and censoring.
- Real deformations remain a separate track with `all`,
  `both_camera_preserved`, `both_semantic_gate`,
  `both_preservation_gate`, and fit-relation strata.
- All estimable endpoint families use 1,000 seed-42 shoe-cluster bootstrap
  repeats by default. Leave-one-shoe-out output covers controlled fidelity,
  resolution, NAF, attribution, geometry calibration, and stress.
- No confirmatory p-values are calculated. Constant, nonfinite, abstained, and
  undersized samples produce safe NaN endpoints instead of exceptions.

## Aggregate and output safety

`analyze_upper_run` resolves the Task 8 immutable aggregate pointer exactly
once, then reads controlled, geometry-calibration, and stress Parquets from
that one generation. It stages the complete reports tree, including CSV and
Parquet table forms, validates every named CSV/PNG/report file, and swaps the
tree atomically while retaining the prior complete tree until publication
succeeds.

The synthetic output audit generated all named deliverables:

```text
reports/tables/controlled_fidelity.csv
reports/tables/severity_curves.csv
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

Every named file was nonempty. The report opens with
`fixed/precomputed DA3 cameras (not calibrated GT)`, states upper-only/four
shoes/descriptive intervals, and separates controlled fidelity, geometry
calibration, and real-deformation stress.

## CLI and full launcher

- The CLI now has exactly seven commands; `analyze RUN_ROOT` prints all
  generated paths as JSON.
- `scripts/run_upper_full.sh` uses the required Pixal3D Python and exact
  config/run/DA3 paths.
- Geometry maps the four exact shoes to physical GPUs 4, 5, 6, and 7.
- Features map the four exact backbones to physical GPUs 4, 5, 6, and 7.
- Both parallel phases wait for every process, preserve per-worker logs, and
  propagate any worker failure before aggregation or analysis.

## Verification evidence

Focused Task 9 plus legacy analysis:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_analysis.py tests/test_upper_analysis.py tests/test_upper_cli.py -q

20 passed in 16.71s
```

Relevant Task 7-9 integration:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_metric_worker.py tests/test_pipeline_integration.py \
  tests/test_pilot_gate.py tests/test_upper_cli.py \
  tests/test_upper_analysis.py -q

75 passed in 76.78s (0:01:16)
```

Fresh full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q

233 passed in 184.65s (0:03:04)
```

Fresh static surface verification:

```text
bash -n scripts/run_upper_full.sh
all seven `met3r.upper_cli COMMAND --help` invocations
/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
git diff --check

exit 0
```

Ruff is not installed in the required Pixal3D environment:

```text
/opt/conda/envs/pixal3d/bin/python: No module named ruff
```

## Self-review and concerns

- The four-shoe intervals are deliberately descriptive and cannot support
  broad generalization.
- The report's camera limitation is exactly
  `fixed/precomputed DA3 cameras (not calibrated GT)`.
- Only upper views are analyzed; bottom and cross-group pairs remain excluded.
- Controlled, calibration, and stress tracks remain separate.
- Full-run feasibility avoids quadratic scans by indexing both clean
  baselines and calibration curves.
- Task 9 validates software and synthetic statistical contracts only. The real
  GPU-4 pilot and four-GPU experiment remain Tasks 10 and 11.

---

## Review remediation: endpoint validity and hierarchical calibration

The first Task 9 review rejected commit `8770168` for five important
statistical issues: endpoint bootstrap intervals could be formed from only
one distinct finite shoe; the geometry envelope omitted missing kinds and
snapped to tested magnitudes; absolute feature share averaged rowwise ratios;
mixed fit relations duplicated stress rows; and resolution/NAF contrasts
omitted validity endpoints.

### Review RED evidence

Five independent regression tests were added first:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_analysis.py -q

5 failed, 6 passed in 10.76s
```

Each failure mapped directly to one review finding:

- missing `severity_spearman_n_shoes_valid`;
- missing hierarchy-weighted `feature_effect_abs_mean`;
- missing interpolated `matched_magnitude_raw`;
- absent mutually exclusive mixed fit-relation label;
- absent `coverage_difference_from_native`.

### Incremental GREEN evidence

```text
finite-shoe bootstrap/count regression: 1 passed in 3.07s
hierarchical absolute attribution ratio: 1 passed in 2.82s
joint two-view fit stratum: 1 passed in 3.19s
resolution/NAF validity contrasts: 1 passed in 4.03s
geometry interpolation/censoring: 1 passed in 3.15s
all Task 9 analysis regressions: 11 passed in 11.49s
```

### Statistical corrections

- Every endpoint now reports `<endpoint>_n_shoes_valid`. Bootstrap bounds are
  NaN unless at least two distinct shoes have finite values for that endpoint.
  Point estimates and leave-one-shoe-out rows remain available with their
  explicit valid-shoe counts.
- Every observed feature cell contributes one geometry-envelope record for
  each canonical kind: rotation, relative translation, depth scale, and
  relative depth noise. Missing, wholly abstained, and out-of-range curves are
  censored rather than removed from coverage.
- Geometry crossings are linearly interpolated at the first crossing of the
  monotone cumulative absolute-effect envelope. Neutral origins are explicit.
  Depth scale reports both raw scale factor and effective absolute distance
  from neutral 1; every kind records raw/effective units and total, covered,
  and censored cell counts.
- Absolute feature and geometry effects are separately averaged by changed
  placement, pair, and shoe. The reported feature share is the ratio of those
  equal-shoe weighted absolute effects. Its bootstrap repeats the ratio after
  shoe-cluster resampling. Numerator, denominator, and valid-shoe counts are
  explicit.
- Fit-relation strata are mutually exclusive. Equal relations use
  `both_fit_relation:<value>`; mixed pairs use one canonical first/second label
  such as `mixed_fit_relation:improved|equivalent`.
- Paired resolution and all three NAF comparator contrasts now include
  coverage, abstention rate, both directional overlaps, signed and absolute
  nuisance deltas, and discrimination endpoints, each with a paired
  valid-shoe count and descriptive interval.

### Review cumulative verification

Focused Task 9 plus legacy analysis:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_analysis.py tests/test_upper_analysis.py tests/test_upper_cli.py -q

24 passed in 18.46s
```

Relevant Task 7-9 integration:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_metric_worker.py tests/test_pipeline_integration.py \
  tests/test_pilot_gate.py tests/test_upper_cli.py \
  tests/test_upper_analysis.py -q

79 passed in 77.11s (0:01:17)
```

Fresh full suite:

```text
/opt/conda/envs/pixal3d/bin/python -m pytest -q

237 passed in 170.69s (0:02:50)
```

Fresh static surface verification:

```text
bash -n scripts/run_upper_full.sh
all seven `met3r.upper_cli COMMAND --help` invocations
/opt/conda/envs/pixal3d/bin/python -m compileall -q met3r tests
git diff --check

exit 0
```

The remediation preserves the exact limitation label
`fixed/precomputed DA3 cameras (not calibrated GT)`, upper-only/four-shoe
scope, descriptive-interval wording, and separate controlled, calibration,
and stress tracks.

---

## Figure review remediation: hierarchical curves and semantic grouping

The next cumulative review found that the severity figure still averaged raw
rows and that every line grouping was too coarse to prevent unrelated feature
variants from being connected. It also requested removal of the obsolete
grid-snapping description.

### Figure review RED and GREEN

Two focused regressions were written first. One constructs unequal numbers of
rows per pair and shoe; the other constructs two feature variants at the same
severities and inspects the semantic plot partitions.

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_analysis.py::test_severity_curve_weights_placements_pairs_and_shoes_equally \
  tests/test_upper_analysis.py::test_semantic_plot_groups_never_connect_distinct_variants -q

2 failed in 3.14s
```

The failures showed that the old severity table lacked `n_shoes` and that no
full-identity semantic grouping boundary existed.

After implementation:

```text
2 passed in 3.21s
all Task 9 analysis tests: 13 passed in 12.92s
```

### Figure corrections

- Added an explicit `severity_curves.csv` and matching Parquet table. It is
  keyed by full backbone/track/mode/resolution identity, corruption, and
  severity. Values average changed-view placements within pair, pairs within
  shoe, then shoes equally, and retain the valid-shoe count.
- Severity curves use the complete feature identity plus corruption.
- Resolution utility holds backbone, track, mode, corruption, and native
  resolution fixed while resolution is the x axis.
- NAF compares the categorical normal-guide comparator at fixed backbone,
  track, resolution, and corruption.
- Backbone discrimination compares categorical backbones only at fixed track,
  mode, resolution, and corruption.
- Attribution holds backbone, track, mode, and corruption fixed across
  resolution.
- Geometry calibration holds backbone, track, mode, corruption, and
  perturbation kind fixed across resolution, preventing mixed-unit lines.
- Stress and coverage likewise retain their complete compatible semantic
  identities.
- Figures with at most 20 semantic groups show the complete labels. Denser
  figures omit the unreadable legend and annotate the number of separately
  plotted groups.
- Report wording now describes an interpolated first crossing on the monotone
  absolute-effect envelope, with raw/effective units and censoring.

### Figure review cumulative verification

```text
focused legacy analysis, Task 9, and CLI:
  26 passed in 20.43s
relevant Task 7-9 integration:
  81 passed in 84.55s (0:01:24)
fresh full suite:
  239 passed in 170.96s (0:02:50)
bash syntax, all seven command helps, compileall, and git diff check:
  exit 0
```

All figures remained nonempty in the atomic synthetic-output audit. The camera
limitation, upper-only/four-shoe scope, descriptive intervals, primary
`s10 - s00`, adaptive `s11` secondary, and separate analysis tracks are
unchanged.

---

## Panel review remediation: incomparable geometry and backbone scales

The next cumulative review found that one geometry axis mixed four
perturbation units and that the stress figure displayed raw metric deltas from
different backbones on one axis.

### Panel review RED and GREEN

Two tests were written first against the actual Matplotlib figure object. They
inspect panel titles, y-axis labels, line labels, y-axis sharing, and scale
limits.

```text
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_upper_analysis.py::test_geometry_panel_figure_separates_kinds_and_effective_units \
  tests/test_upper_analysis.py::test_stress_panel_figure_separates_backbone_raw_scales -q

2 failed in 3.43s
```

Both failed because the panel-figure boundary did not yet exist.

After implementation:

```text
isolated panel tests: 2 passed in 3.60s
all Task 9 analysis tests: 15 passed in 13.70s
```

### Panel implementation

- `geometry_calibration.png` has four perturbation-kind panels:
  rotation, relative translation, depth scale, and relative depth noise.
  Panels never share y axes. Their effective y-axis units are respectively
  degrees, camera-radius fraction, absolute scale delta from 1, and relative
  depth standard deviation.
- Geometry lines retain backbone, feature track, feature mode, and corruption
  identity within each fixed-kind panel while resolution remains the x axis.
- `real_deformation_stress.png` has one independent-y panel per backbone.
  Lines retain feature track, feature mode, and quality stratum identity while
  resolution remains the x axis.
- The stress y label and report explicitly state that raw metric deltas are
  interpretable only within backbone. Cross-panel raw magnitudes must not be
  compared.
- Empty panels retain an explicit no-estimable-observations message. The
  synthetic output audit still requires both PNGs and every other primary
  figure to be nonempty.

### Panel review cumulative verification

```text
focused legacy analysis, Task 9, and CLI:
  28 passed in 20.18s
relevant Task 7-9 integration:
  83 passed in 79.24s (0:01:19)
fresh full suite:
  241 passed in 171.47s (0:02:51)
bash syntax, all seven command helps, compileall, and git diff check:
  exit 0
```

The exact camera limitation, upper-only/four-shoe scope, descriptive
intervals, primary/secondary distinction, and separate analysis tracks remain
unchanged.
