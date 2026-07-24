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
- Geometry calibration finds the minimum known perturbation magnitude whose
  absolute geometry effect reaches each observed absolute feature effect and
  reports envelope coverage.
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
