# Korean Visual Research Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the adjacent 16-view experiment's main research report as a Korean-first, figure-embedded, scientifically bounded document.

**Architecture:** Keep the existing analysis tables, plots, atomic report publisher, and experiment artifacts unchanged. Extend the existing `_write_report` function to derive representative numbers from the published tables and emit a Korean narrative with relative Markdown links to the five core figures; then republish the real run and rebuild its audit visualizations.

**Tech Stack:** Python 3.11, pandas, pytest, Markdown, existing `met3r.adjacent16_analysis` and `met3r.adjacent16_cli`

## Global Constraints

- The generated `final_research_report.md`, not a manually edited copy, is the source of truth.
- Write the narrative primarily in Korean while preserving exact English technical terms where needed.
- Always call the camera contract `fixed/precomputed DA3 camera (calibrated GT 아님)`.
- Embed all five existing core figures using report-relative Markdown paths.
- Separate direct observations, bounded interpretation, non-claims, and proposed module design.
- Preserve all existing aggregate Parquet/CSV data, row counts, bootstrap settings, and reproducibility hashes.
- Do not rerun feature extraction, recompute experiment metrics, save raw DINO features, or claim calibrated-camera ground truth.
- Preserve the protected untracked file `configs/environment_pixal3d.json`; never stage or modify it.

---

### Task 1: Lock the Korean visual report contract in a failing test

**Files:**
- Modify: `tests/test_adjacent16_analysis.py:942-1050`

**Interfaces:**
- Consumes: `analyze_adjacent16_run(tmp_path, bootstrap_repeats=20, seed=42) -> Adjacent16AnalysisOutputs`
- Produces: An integration-test contract for Korean headings, five figure links, camera wording, bounded module claims, counts, and hashes.

- [ ] **Step 1: Add Korean visual-report assertions to the atomic snapshot test**

After reading `report = outputs.report_path.read_text(...)`, replace the English-content assertions with the following contract while retaining the table, atomicity, row-count, seed, bootstrap, and sha256 checks:

```python
assert report.startswith(
    "# 인접 Upper 16-view DINO Feature Fidelity 및 Reliability 연구 보고서"
)
assert [
    line.removeprefix("## ")
    for line in report.splitlines()
    if line.startswith("## ")
] == [
    f"{index}. {title}"
    for index, title in enumerate(REPORT_SECTION_TITLES, start=1)
]
for filename in FIGURE_FILENAMES:
    assert f"](figures/{filename})" in report

assert "한눈에 보는 결론" in report
assert "관찰된 결과" in report
assert "해석" in report
assert "직접 확인된 사실" in report
assert "이번 실험에서 확인되지 않은 것" in report
assert "제안하는 모듈 설계" in report
assert "0 ≤ pixel_weight ≤ semantic_weight ≤ 1" in report
assert "fixed/precomputed DA3 camera (calibrated GT 아님)" in report
assert "절대적인 DA3 카메라 오차를 추정하지 않는다" in report
assert "해상도 hard gate가 직접 검증된 것은 아니다" in report
assert "결측 또는 abstention을 0점으로 대체하지 않았다" in report
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_analysis.py::test_analyze_adjacent16_run_publishes_exact_atomic_research_snapshot \
  -q
```

Expected: FAIL because the generated report still begins with the English title and has no embedded figure links.

- [ ] **Step 3: Commit the failing contract**

```bash
git add tests/test_adjacent16_analysis.py
git commit -m "test: require Korean visual research report"
```

---

### Task 2: Generate the Korean-first report with embedded evidence

**Files:**
- Modify: `met3r/adjacent16_analysis.py:101-112`
- Modify: `met3r/adjacent16_analysis.py:2324-2498`
- Test: `tests/test_adjacent16_analysis.py`

**Interfaces:**
- Consumes: the existing tables `support_coverage`, `backbone_common37`, `resolution_bilinear`, `naf_vs_bilinear`, `geometry_low_high`, `stress_reliability`, and `analyzability`.
- Produces: `_write_report(...) -> None`, writing a Korean-first `final_research_report.md` with all five relative image references and a complete artifact index.

- [ ] **Step 1: Replace the report section titles with the approved Korean structure**

Use these exact titles:

```python
REPORT_SECTION_TITLES = (
    "한눈에 보는 결론",
    "실험 범위와 해석 원칙",
    "인접 뷰 Support와 Overlap",
    "DINO Backbone 비교",
    "DINOv3-L 해상도 및 NAF 비교",
    "카메라와 Geometry 민감도",
    "실제 독립 Deformation의 Reliability",
    "View Reliability Module 시사점",
    "한계와 후속 검증",
    "시각화 및 재현성 부록",
)
CAMERA_LIMITATION = (
    "fixed/precomputed DA3 camera (calibrated GT 아님)"
)
```

- [ ] **Step 2: Derive all headline values from the published tables**

Inside `_write_report`, use `_representative_estimate` for these exact cells rather than hard-coding production results:

```python
support_by_step = {
    step: _representative_estimate(
        support,
        f"baseline_steps == '{step}' and "
        "metric == 'minimum_directional_overlap'",
        "estimate",
    )
    for step in ("1", "2", "3")
}
translation_by_backbone = {
    backbone_name: _representative_estimate(
        backbone,
        "baseline_steps == 'all' and "
        f"backbone == '{backbone_name}' and "
        "corruption == 'translation' and "
        "endpoint == 'severity_spearman'",
        "estimate",
    )
    for backbone_name in (
        "dino_vitb16",
        "dinov2_vitb14",
        "dinov3_vitb16",
        "dinov3_vitl16_pixal3d",
    )
}
resolution_endpoints = {
    (resolution_value, corruption): _representative_estimate(
        resolution,
        "baseline_steps == 'all' and "
        f"resolution == {resolution_value} and "
        f"corruption == '{corruption}' and "
        "endpoint == 'severity_spearman'",
        "estimate",
    )
    for resolution_value in (37, 592)
    for corruption in ("translation", "thin_plate_warp")
}
```

Also derive:

- all-baseline JPEG `nuisance_absolute_delta` at 37 and 592;
- all-baseline 592 NAF-minus-bilinear for JPEG absolute delta and TPS Spearman;
- all-baseline rotation responses at 0.25°, 1°, and 5° for L37-bilinear and L592-NAF;
- all-baseline strict/moderate/high-misalignment pass rates;
- baseline-step 1/2/3 finite low/high disagreement;
- the four continuous reliability correlations already collected by the function.

- [ ] **Step 3: Replace the English `lines` payload with the Korean narrative**

The generated Markdown must follow the approved ten sections and include:

```markdown
# 인접 Upper 16-view DINO Feature Fidelity 및 Reliability 연구 보고서

## 1. 한눈에 보는 결론

| 연구 질문 | 관찰된 결과 | 해석 |
|---|---|---|
| 인접 뷰가 geometry confound를 줄이는가? | ... | ... |
| 어떤 DINO backbone이 적합한가? | ... | ... |
| 높은 feature resolution이 항상 유리한가? | ... | ... |
| DA3 geometry 오차와 feature 차이를 분리할 수 있는가? | ... | ... |
| View Reliability Module을 무엇으로 학습해야 하는가? | ... | ... |
```

Place each figure in its matching section:

```markdown
![인접 뷰의 support와 overlap](figures/01_adjacent_support.png)
![37-grid에서 DINO backbone 비교](figures/02_common37_backbones.png)
![DINOv3-L 해상도와 NAF 비교](figures/03_dinov3l_resolution_naf.png)
![카메라와 geometry perturbation 민감도](figures/04_geometry_calibration.png)
![독립 deformation reliability 분석](figures/05_real_deformation_reliability.png)
```

For every figure, write three short paragraphs labelled `그림 읽는 법`,
`관찰된 결과`, and `해석`. The geometry section must say that injected local
perturbations estimate sensitivity around the precomputed camera but cannot
recover unknown absolute DA3 error. The module section must contain three
subheadings:

```markdown
### 직접 확인된 사실
### 이번 실험에서 확인되지 않은 것
### 제안하는 모듈 설계
```

State explicitly that reliability-stratified pass rates and finite L37/L592
disagreement support continuous reliability weighting, while the absence of
low-finite/high-abstained cases means a semantic-low/pixel-high hard gate was
not directly validated. Include:

```text
0 ≤ pixel_weight ≤ semantic_weight ≤ 1
```

Keep `input_lines` and `artifact_lines` under section 10, and add report-relative
links to `audit_visualizations/pca/`, `audit_visualizations/directional/`, and
`audit_visualizations/audit_manifest.json`.

- [ ] **Step 4: Run the focused analysis tests and confirm GREEN**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_analysis.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Run formatting and diff checks**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m ruff check \
  met3r/adjacent16_analysis.py tests/test_adjacent16_analysis.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit the generator implementation**

```bash
git add met3r/adjacent16_analysis.py tests/test_adjacent16_analysis.py
git commit -m "feat: publish Korean visual research report"
```

---

### Task 3: Republish the completed experiment without losing supplementary documents

**Files:**
- Regenerate: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/final_research_report.md`
- Regenerate: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/figures/*.png`
- Regenerate: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/audit_visualizations/**`
- Preserve if present: `reports/analysis_interpretation_ko.md`
- Preserve if present: `reports/execution_incidents.md`

**Interfaces:**
- Consumes: the already completed aggregate metrics and geometry preflight in the frozen run root.
- Produces: an atomically published Korean report, five figures, and restored PCA/directional audit PNGs.

- [ ] **Step 1: Preserve supplementary report files in an isolated temporary directory**

Run:

```bash
REPORT_BACKUP_DIR=$(mktemp -d)
RUN_ROOT=/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
for name in analysis_interpretation_ko.md execution_incidents.md; do
  if test -f "$RUN_ROOT/reports/$name"; then
    cp "$RUN_ROOT/reports/$name" "$REPORT_BACKUP_DIR/$name"
  fi
done
```

Expected: only the two named supplementary documents are copied when present.

- [ ] **Step 2: Republish analysis, figures, and audit visualizations**

Run from the worktree:

```bash
PHYSICAL_GPUS=0,1,2,3 bash scripts/run_adjacent16_core.sh finalize
```

Expected:

- aggregate and final gate exit 0;
- analysis publishes eight table groups, five core figures, and the Korean report;
- visualization uses physical GPU 3 as `cuda:0` under `CUDA_VISIBLE_DEVICES`;
- no feature extraction jobs are executed.

- [ ] **Step 3: Restore preserved supplementary documents**

Run:

```bash
for name in analysis_interpretation_ko.md execution_incidents.md; do
  if test -f "$REPORT_BACKUP_DIR/$name"; then
    cp "$REPORT_BACKUP_DIR/$name" "$RUN_ROOT/reports/$name"
  fi
done
```

Expected: the main generated report remains authoritative and the prior supplemental history remains available.

---

### Task 4: Verify the real report, images, gates, and repository state

**Files:**
- Inspect: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/final_research_report.md`
- Inspect: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/final_gate.json`
- Inspect: `/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1/reports/audit_visualizations/`

**Interfaces:**
- Consumes: Task 3's published report snapshot.
- Produces: evidence that every embedded link resolves, output counts remain exact, no raw feature artifacts exist, and the frozen gate still passes.

- [ ] **Step 1: Validate Markdown image targets and required Korean claims**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
from pathlib import Path
import re

report = Path(
    "/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/"
    "20260727_adjacent16_core_v1/reports/final_research_report.md"
)
text = report.read_text(encoding="utf-8")
targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
assert len(targets) == 5, targets
assert all((report.parent / target).is_file() for target in targets)
for phrase in (
    "한눈에 보는 결론",
    "fixed/precomputed DA3 camera (calibrated GT 아님)",
    "0 ≤ pixel_weight ≤ semantic_weight ≤ 1",
    "해상도 hard gate가 직접 검증된 것은 아니다",
):
    assert phrase in text, phrase
print({"report": str(report), "figure_links": targets})
PY
```

Expected: five valid image paths and no assertion failure.

- [ ] **Step 2: Verify row counts, visualization counts, and raw-artifact exclusion**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path(
    "/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/"
    "20260727_adjacent16_core_v1"
)
generation = root / "metrics/current"
counts = {
    name: len(pd.read_parquet(generation / name))
    for name in (
        "controlled_fidelity.parquet",
        "geometry_calibration.parquet",
        "real_deformation_stress.parquet",
    )
}
assert tuple(counts.values()) == (31_104, 6_528, 1_152), counts
audit = root / "reports/audit_visualizations"
assert len(tuple((audit / "pca").glob("*.png"))) == 160
assert len(tuple((audit / "directional").glob("*.png"))) == 80
raw = tuple(root.rglob("*.pt")) + tuple(root.rglob("*.npy")) + tuple(root.rglob("*.npz"))
assert not raw, raw[:10]
print({"metric_rows": counts, "pca_png": 160, "directional_png": 80})
PY
```

Expected: 38,784 total metric rows, 160 PCA PNGs, 80 directional PNGs, and zero raw feature artifacts.

- [ ] **Step 3: Re-run final gate and the full focused test set**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m met3r.adjacent16_cli \
  final-validate \
  /root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260727_adjacent16_core_v1
/opt/conda/envs/pixal3d/bin/python -m pytest \
  tests/test_adjacent16_analysis.py \
  tests/test_adjacent16_visualization.py \
  tests/test_adjacent16_gate.py \
  -q
```

Expected: final gate reports `passed: true` and every focused test passes.

- [ ] **Step 4: Inspect the final diff and protected file**

Run:

```bash
git status --short
git log -4 --oneline
```

Expected: only the pre-existing untracked `configs/environment_pixal3d.json` remains; it is not staged or modified.
