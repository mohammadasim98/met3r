# 한국어 중심 시각 연구 보고서 재구성 설계

## 목적

`20260727_adjacent16_core_v1` 실험의 `final_research_report.md`를 한국어 중심의
독립적인 연구 보고서로 재구성한다. 독자가 별도 표를 먼저 열지 않아도 핵심 결과,
그림의 의미, 연구적 시사점과 한계를 이해할 수 있어야 한다.

일회성 Markdown 수정을 하지 않고 `met3r/adjacent16_analysis.py`의 보고서 생성기를
수정한다. 따라서 동일한 분석을 다시 실행해도 한국어 보고서와 그림 참조가
재현되어야 한다.

## 독자와 문체

- 주요 독자는 Pixal3D의 View Reliability Module 설계를 검토하는 연구자다.
- 설명은 대체로 한국어로 쓰고 DINO, NAF, AUROC, Spearman correlation,
  finite/abstained 등 정확성이 필요한 기술 용어는 영문을 병기한다.
- 각 절은 먼저 관찰 결과를 설명하고, 그 다음 제한된 해석을 제시한다.
- DA3 추정 카메라는 항상 `fixed/precomputed DA3 camera (calibrated GT 아님)`로
  표기한다.

## 보고서 구조

1. **한눈에 보는 결론**
   - 전체 metric row 수와 데이터 범위를 짧게 설명한다.
   - 핵심 결론 다섯 개를 표로 제시한다.
   - 결론마다 대표 수치와 View Reliability Module에 주는 의미를 함께 쓴다.
2. **실험 범위와 해석 원칙**
   - 4개 신발, 인접 upper 16-view, metric row 구성, bootstrap 단위를 명시한다.
   - 결측/abstention은 0점이 아니라 관찰 불가능한 evidence임을 명시한다.
3. **인접 뷰 support와 overlap**
   - `figures/01_adjacent_support.png`를 본문에 삽입한다.
   - clean coverage와 baseline step별 최소 directional overlap을 설명한다.
4. **DINO backbone 비교**
   - `figures/02_common37_backbones.png`를 삽입한다.
   - identity AUROC 포화, translation/TPS 순위 보존, JPEG nuisance를 분리해 해석한다.
5. **DINOv3-L 해상도 및 NAF 비교**
   - `figures/03_dinov3l_resolution_naf.png`를 삽입한다.
   - bilinear resolution 변화와 NAF-minus-bilinear 결과를 함께 설명한다.
6. **카메라/geometry 민감도**
   - `figures/04_geometry_calibration.png`를 삽입한다.
   - rotation/depth perturbation의 local response를 설명한다.
   - calibrated GT 부재 때문에 절대 DA3 오차는 분리·복원할 수 없음을 명시한다.
7. **실제 독립 deformation의 reliability**
   - `figures/05_real_deformation_reliability.png`를 삽입한다.
   - strict/moderate/high-misalignment의 pass rate와 low/high disagreement를 설명한다.
   - 수동 reliability label은 제외 기준이 아니라 분석 stratum임을 명시한다.
8. **View Reliability Module 시사점**
   - 직접 관찰된 증거, 관찰되지 않은 경계, 제안 설계를 세 구역으로 나눈다.
   - 연속적인 view/geometry reliability weight를 지지하지만 해상도 hard gate가
     직접 검증된 것은 아니라고 쓴다.
   - 제안 관계 `0 ≤ pixel_weight ≤ semantic_weight ≤ 1`을 명시한다.
9. **한계와 후속 검증**
   - 4개 신발, upper-only, DA3 추정 geometry, descriptive bootstrap의 한계를 쓴다.
10. **시각화 및 재현성 부록**
    - PCA와 directional audit 폴더를 안내한다.
    - 원본 aggregate/table과 sha256 목록은 유지하되 본문에서 분리한다.

## 그림 배치와 설명 규칙

- 그림 링크는 보고서 기준 상대 경로를 사용한다.
- 각 그림은 관련 절의 첫 설명 다음에 바로 배치한다.
- 그림 아래에는 다음 세 요소를 반드시 포함한다.
  1. 패널 또는 축이 무엇을 측정하는지
  2. 대표 수치가 무엇인지
  3. 어떤 결론은 가능하고 어떤 결론은 불가능한지
- 기존 핵심 그림 다섯 개를 사용하며, 본 변경을 위해 새로운 raw DINO feature를
  저장하거나 실험 metric을 재계산하지 않는다.

## 데이터와 결과 보존

- 기존 parquet/CSV와 그림은 수정하지 않는다.
- 보고서 생성기가 현재 분석 table에서 대표 수치를 읽어 Markdown을 만든다.
- 원본 결과의 row count, bootstrap 설정, hash index를 보존한다.
- 분석 publisher가 `reports/`를 원자적으로 교체하는 현재 동작을 유지한다.
- 수동으로 추가한 문서에 의존하지 않고 주된 한국어 해석을 생성 보고서 자체에
  포함한다.

## 검증

- 먼저 실패하는 테스트로 다음 계약을 고정한다.
  - 한국어 제목과 핵심 절이 존재한다.
  - 다섯 개 핵심 그림의 상대 링크가 모두 존재한다.
  - 대표 관찰 수치와 DA3 한계 문구가 존재한다.
  - View Reliability Module의 관찰/비관찰/제안이 구분된다.
- 구현 후 관련 단위 테스트를 통과시킨다.
- 실제 run에 `analyze`, `visualize`, `final-validate`를 순서대로 실행한다.
- 최종 보고서의 그림 링크가 모두 유효한지, audit PNG 수와 raw feature artifact
  금지 조건, final gate를 다시 확인한다.

## 범위 밖

- metric 재실험 또는 새로운 backbone inference
- 새로운 학습형 View Reliability Module 구현
- calibrated camera GT가 없는 상태에서 DA3 절대 오차를 추정하는 주장
- 기존 결과를 넘어선 통계적 유의성 또는 population-level 우월성 주장
