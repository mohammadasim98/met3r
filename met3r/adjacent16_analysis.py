"""Focused descriptive analysis for the frozen adjacent 16-view experiment."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .job_registry import sha256_file
from .pilot_gate import resolve_aggregate_generation


COMMON_SUPPORT_COLUMNS = (
    "shoe_id",
    "quartet_id",
    "pair_id",
    "changed_view",
    "corruption",
    "severity",
)
GEOMETRY_SUPPORT_COLUMNS = (
    "shoe_id",
    "quartet_id",
    "pair_id",
    "changed_view",
    "perturbation_kind",
    "magnitude",
)
STRESS_SUPPORT_COLUMNS = (
    "shoe_id",
    "quartet_id",
    "pair_id",
)
COMMON37_BACKBONES = (
    "dino_vitb16",
    "dinov2_vitb14",
    "dinov3_vitb16",
    "dinov3_vitl16_pixal3d",
)
CONTROLLED_SEVERITIES = {
    "translation": (0.03, 0.06, 0.12, 0.2),
    "thin_plate_warp": (0.25, 0.5, 0.75, 1.0),
    "jpeg": (0.25, 0.5, 0.75, 1.0),
    "identity_swap": (1.0,),
}
FROZEN_SHOE_IDS = frozenset(
    {
        "air_huarache_run_ultra_111449",
        "air_max_90_nrg_dc6083_500",
        "air_max_90_surplus_cq7743_300",
        "air_tuned_max_cv6984_001",
    }
)
FROZEN_BASELINE_STEPS = (1, 2, 3)
FROZEN_BASELINE_DOMAIN: tuple[int | str, ...] = (
    "all",
    *FROZEN_BASELINE_STEPS,
)
BASELINE_KEY_COLUMNS = ("shoe_id", "quartet_id", "pair_id")
GEOMETRY_MAGNITUDES = {
    "rotation": (0.25, 0.5, 1.0, 2.0, 5.0),
    "relative_translation": (0.0025, 0.005, 0.01, 0.02),
    "depth_scale": (1.01, 1.02, 1.05, 1.1),
    "relative_depth_noise": (0.005, 0.01, 0.02, 0.05),
}
_ENDPOINTS = {
    "identity_swap": ("clean_corrupt_auroc",),
    "translation": ("severity_spearman", "pairwise_rank_accuracy"),
    "thin_plate_warp": ("severity_spearman", "pairwise_rank_accuracy"),
    "jpeg": ("nuisance_signed_delta", "nuisance_absolute_delta"),
}
TABLE_STEMS = (
    "support_coverage",
    "backbone_common37",
    "resolution_bilinear",
    "naf_vs_bilinear",
    "geometry_low_high",
    "stress_reliability",
    "common_support",
    "analyzability",
)
FIGURE_FILENAMES = (
    "01_adjacent_support.png",
    "02_common37_backbones.png",
    "03_dinov3l_resolution_naf.png",
    "04_geometry_calibration.png",
    "05_real_deformation_reliability.png",
)
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
CAMERA_LIMITATION = "fixed/precomputed DA3 camera (calibrated GT 아님)"
PLOT_CAMERA_LIMITATION = "fixed/precomputed DA3 camera (not calibrated GT)"
_CONTROLLED_SUPPORT_COMPARISONS = frozenset(
    {
        "four_backbone_common37",
        "five_resolution_bilinear",
        "naf_vs_bilinear_74",
        "naf_vs_bilinear_148",
        "naf_vs_bilinear_296",
        "naf_vs_bilinear_592",
    }
)


@dataclass(frozen=True)
class Adjacent16AnalysisOutputs:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    report_path: Path


@dataclass(frozen=True)
class _PreflightEvidence:
    frame: pd.DataFrame
    parquet_bytes: bytes
    decision_bytes: bytes
    parquet_sha256: str
    decision_sha256: str


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _require_frozen_shoe_cohort(frame: pd.DataFrame, label: str) -> None:
    _require(frame, ("shoe_id",), label)
    observed = frozenset(frame["shoe_id"].dropna().astype(str))
    if observed != FROZEN_SHOE_IDS:
        raise ValueError(
            f"{label} must contain the frozen four-shoe cohort; "
            f"observed={sorted(observed)!r}"
        )


def _validated_baseline_steps(
    frame: pd.DataFrame,
    label: str,
) -> None:
    _require(frame, (*BASELINE_KEY_COLUMNS, "baseline_steps"), label)
    numeric = pd.to_numeric(frame["baseline_steps"], errors="coerce")
    invalid = (
        ~np.isfinite(numeric)
        | (numeric != np.floor(numeric))
        | ~numeric.isin(FROZEN_BASELINE_STEPS)
    )
    if invalid.any():
        raise ValueError(
            f"{label} has baseline_steps outside the frozen 1/2/3 domain"
        )
    working = frame.loc[:, list(BASELINE_KEY_COLUMNS)].copy()
    working["_baseline_steps"] = numeric.astype(int)
    step_counts = working.groupby(
        list(BASELINE_KEY_COLUMNS),
        dropna=False,
        sort=False,
    )["_baseline_steps"].nunique()
    inconsistent = step_counts[step_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{label} has inconsistent baseline_steps for physical key "
            f"{inconsistent.index[0]!r}"
        )


def _safe_mean(values: Iterable[object]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if (
        valid.sum() < 3
        or np.unique(first[valid]).size < 2
        or np.unique(second[valid]).size < 2
    ):
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def _safe_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    label_values = np.asarray(labels, dtype=bool)
    score_values = np.asarray(scores, dtype=float)
    valid = np.isfinite(score_values)
    label_values = label_values[valid]
    score_values = score_values[valid]
    positives = int(label_values.sum())
    negatives = int((~label_values).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score_values)
    return float(
        (
            ranks[label_values].sum()
            - positives * (positives + 1) / 2
        )
        / (positives * negatives)
    )


def _safe_rank_accuracy(
    severity: Sequence[float],
    scores: Sequence[float],
) -> float:
    severity_values = np.asarray(severity, dtype=float)
    score_values = np.asarray(scores, dtype=float)
    valid = np.isfinite(severity_values) & np.isfinite(score_values)
    severity_values = severity_values[valid]
    score_values = score_values[valid]
    comparisons: list[float] = []
    for first in range(len(severity_values)):
        for second in range(first + 1, len(severity_values)):
            expected = np.sign(
                severity_values[first] - severity_values[second]
            )
            if expected == 0:
                continue
            observed = np.sign(score_values[first] - score_values[second])
            comparisons.append(
                1.0
                if observed == expected
                else 0.5
                if observed == 0
                else 0.0
            )
    return _safe_mean(comparisons)


def _primary_valid(frame: pd.DataFrame) -> pd.Series:
    abstained = (
        frame["abstained"].fillna(True).astype(bool)
        if "abstained" in frame
        else pd.Series(False, index=frame.index)
    )
    return (
        ~abstained
        & np.isfinite(pd.to_numeric(frame["s00"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["s10"], errors="coerce"))
    )


def _support_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in COMMON_SUPPORT_COLUMNS)


def _base_key(row: pd.Series) -> tuple[object, object, object]:
    return row["shoe_id"], row["quartet_id"], row["pair_id"]


def _variant_support(frame: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[object, ...]]]:
    valid = _primary_valid(frame)
    clean = frame[(frame["corruption"] == "clean") & valid]
    if clean.duplicated(["shoe_id", "quartet_id", "pair_id"]).any():
        raise ValueError("controlled clean baselines are duplicate")
    valid_bases = {_base_key(row) for _, row in clean.iterrows()}
    changed = frame[(frame["corruption"] != "clean") & valid].copy()
    changed = changed[
        changed.apply(lambda row: _base_key(row) in valid_bases, axis=1)
    ]
    if changed.duplicated(list(COMMON_SUPPORT_COLUMNS)).any():
        raise ValueError("controlled exact support keys are duplicate")
    return clean, {_support_key(row) for _, row in changed.iterrows()}


def _complete_series_coverage(
    common: set[tuple[object, ...]],
    union: set[tuple[object, ...]],
    corruption: str,
) -> float:
    expected = set(CONTROLLED_SEVERITIES[corruption])
    series_severities: dict[tuple[object, ...], set[float]] = {}
    for key in union:
        series = key[:-1]
        series_severities.setdefault(series, set())
    for key in common:
        series_severities.setdefault(key[:-1], set()).add(float(key[-1]))
    if not series_severities:
        return 0.0
    return float(
        np.mean(
            [
                expected <= observed
                for observed in series_severities.values()
            ]
        )
    )


def _series_endpoint(
    rows: pd.DataFrame,
    baseline_score: float,
    endpoint: str,
) -> float:
    scores = pd.to_numeric(rows["s10"], errors="coerce").to_numpy(float)
    severity = pd.to_numeric(rows["severity"], errors="coerce").to_numpy(float)
    if endpoint == "clean_corrupt_auroc":
        return _safe_auc(
            [False, *([True] * len(rows))],
            [baseline_score, *scores],
        )
    if endpoint == "severity_spearman":
        return _safe_spearman(
            [0.0, *severity],
            [baseline_score, *scores],
        )
    if endpoint == "pairwise_rank_accuracy":
        return _safe_rank_accuracy(
            [0.0, *severity],
            [baseline_score, *scores],
        )
    deltas = (
        pd.to_numeric(rows["s10"], errors="coerce")
        - pd.to_numeric(rows["s00"], errors="coerce")
    )
    if endpoint == "nuisance_signed_delta":
        return _safe_mean(deltas)
    if endpoint == "nuisance_absolute_delta":
        return _safe_mean(deltas.abs())
    raise ValueError(f"unknown adjacent endpoint: {endpoint}")


def _shoe_endpoint_values(
    variant: pd.DataFrame,
    common: set[tuple[object, ...]],
    endpoint: str,
) -> pd.DataFrame:
    clean, _ = _variant_support(variant)
    baseline = {
        _base_key(row): float(row["s00"])
        for _, row in clean.iterrows()
    }
    changed = variant[variant["corruption"] != "clean"].copy()
    changed["_support_key"] = changed.apply(_support_key, axis=1)
    changed = changed[changed["_support_key"].isin(common)]
    records: list[dict[str, Any]] = []
    series_columns = [
        "shoe_id",
        "quartet_id",
        "pair_id",
        "changed_view",
        "corruption",
    ]
    for key, rows in changed.groupby(series_columns, dropna=False, sort=True):
        shoe_id, quartet_id, pair_id, _changed, _corruption = key
        records.append(
            {
                "shoe_id": shoe_id,
                "value": _series_endpoint(
                    rows,
                    baseline[(shoe_id, quartet_id, pair_id)],
                    endpoint,
                ),
            }
        )
    if not records:
        return pd.DataFrame(columns=["shoe_id", "value"])
    return (
        pd.DataFrame(records)
        .groupby("shoe_id", sort=True, dropna=False)["value"]
        .mean()
        .reset_index()
    )


def _shoe_clustered_interval(
    shoes: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = pd.to_numeric(shoes["value"], errors="coerce").to_numpy(float)
    finite_count = int(np.isfinite(values).sum())
    if repeats <= 0 or finite_count < 2:
        return float("nan"), float("nan")
    if len(values) > 4:
        raise ValueError("adjacent16 bootstrap received more than four shoes")
    if len(values) < 4:
        values = np.pad(
            values,
            (0, 4 - len(values)),
            constant_values=np.nan,
        )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(repeats):
        sampled = values[rng.integers(0, 4, size=4)]
        estimate = _safe_mean(sampled)
        if np.isfinite(estimate):
            samples.append(estimate)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _summarize_shoes(
    shoes: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    finite = np.isfinite(pd.to_numeric(shoes["value"], errors="coerce"))
    low, high = _shoe_clustered_interval(
        shoes.loc[finite],
        repeats=repeats,
        seed=seed,
    )
    return {
        "estimate": _safe_mean(shoes["value"]),
        "n_shoes_valid": int(finite.sum()),
        "ci_low": low,
        "ci_high": high,
    }


def _paired_shoe_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    paired = first.merge(
        second,
        on="shoe_id",
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    paired["value"] = paired["value_first"] - paired["value_second"]
    return paired[["shoe_id", "value"]]


def _support_slice(
    supports: Sequence[set[tuple[object, ...]]],
    *,
    corruption: str,
    steps: int | str,
    step_by_key: dict[tuple[object, ...], int],
) -> tuple[set[tuple[object, ...]], set[tuple[object, ...]]]:
    def selected(key: tuple[object, ...]) -> bool:
        return key[-2] == corruption and (
            steps == "all" or step_by_key[key] == steps
        )

    selected_sets = [
        {key for key in support if selected(key)}
        for support in supports
    ]
    union = set().union(*selected_sets)
    common = set.intersection(*selected_sets)
    return common, union


def _controlled_support_groups(
    variants: dict[object, pd.DataFrame],
) -> Iterable[
    tuple[
        int | str,
        str,
        tuple[str, ...],
        set[tuple[object, ...]],
        dict[str, float | int],
    ]
]:
    selected = pd.concat(variants.values(), ignore_index=True, sort=False)
    _require_frozen_shoe_cohort(selected, "controlled comparison")
    _validated_baseline_steps(selected, "controlled comparison")
    supports = [_variant_support(variant)[1] for variant in variants.values()]
    step_by_key = {
        _support_key(row): int(row["baseline_steps"])
        for _, row in selected[selected["corruption"] != "clean"].iterrows()
    }
    for steps_value in FROZEN_BASELINE_DOMAIN:
        for corruption, endpoints in _ENDPOINTS.items():
            common, union = _support_slice(
                supports,
                corruption=corruption,
                steps=steps_value,
                step_by_key=step_by_key,
            )
            yield (
                steps_value,
                corruption,
                endpoints,
                common,
                {
                    "n_common_support": len(common),
                    "n_union_support": len(union),
                    "support_retention": (
                        len(common) / len(union)
                        if union
                        else 0.0
                    ),
                    "complete_series_coverage": _complete_series_coverage(
                        common,
                        union,
                        corruption,
                    ),
                },
            )


def summarize_backbone_common37(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare the four bilinear-37 backbones on one exact row intersection."""

    _require(
        frame,
        (
            *COMMON_SUPPORT_COLUMNS,
            "baseline_steps",
            "backbone",
            "feature_mode",
            "resolution",
            "s00",
            "s10",
            "abstained",
        ),
        "controlled metric frame",
    )
    selected = frame[
        (frame["backbone"].isin(COMMON37_BACKBONES))
        & (frame["feature_mode"] == "bilinear")
        & (pd.to_numeric(frame["resolution"], errors="coerce") == 37)
    ].copy()
    variants = {
        backbone: selected[selected["backbone"] == backbone].copy()
        for backbone in COMMON37_BACKBONES
    }
    if any(variant.empty for variant in variants.values()):
        raise ValueError("four common-37 backbone variants are required")
    records: list[dict[str, object]] = []
    for (
        steps_value,
        corruption,
        endpoints,
        common,
        support,
    ) in _controlled_support_groups(variants):
        for endpoint in endpoints:
            for backbone, variant in variants.items():
                shoes = _shoe_endpoint_values(variant, common, endpoint)
                records.append(
                    {
                        "baseline_steps": steps_value,
                        "backbone": backbone,
                        "corruption": corruption,
                        "endpoint": endpoint,
                        **_summarize_shoes(
                            shoes,
                            repeats=bootstrap_repeats,
                            seed=seed,
                        ),
                        **support,
                    }
                )
    return pd.DataFrame(records)


def _dinov3l_variant(
    frame: pd.DataFrame,
    *,
    feature_mode: str,
    resolution: int,
) -> pd.DataFrame:
    return frame[
        (frame["backbone"] == "dinov3_vitl16_pixal3d")
        & (frame["feature_mode"] == feature_mode)
        & (pd.to_numeric(frame["resolution"], errors="coerce") == resolution)
    ].copy()


def summarize_resolution_bilinear(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare five L-backbone bilinear resolutions on exact five-way support."""

    _require(
        frame,
        (
            *COMMON_SUPPORT_COLUMNS,
            "baseline_steps",
            "backbone",
            "feature_mode",
            "resolution",
            "s00",
            "s10",
            "abstained",
        ),
        "controlled metric frame",
    )
    resolutions = (37, 74, 148, 296, 592)
    variants = {
        resolution: _dinov3l_variant(
            frame,
            feature_mode="bilinear",
            resolution=resolution,
        )
        for resolution in resolutions
    }
    if any(variant.empty for variant in variants.values()):
        raise ValueError("five DINOv3-L bilinear resolutions are required")
    records: list[dict[str, object]] = []
    for (
        steps_value,
        corruption,
        endpoints,
        common,
        support,
    ) in _controlled_support_groups(variants):
        for endpoint in endpoints:
            shoe_values = {
                resolution: _shoe_endpoint_values(
                    variant,
                    common,
                    endpoint,
                )
                for resolution, variant in variants.items()
            }
            reference = shoe_values[37]
            for resolution in resolutions:
                estimate = _summarize_shoes(
                    shoe_values[resolution],
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                difference = _summarize_shoes(
                    _paired_shoe_difference(
                        shoe_values[resolution],
                        reference,
                    ),
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                records.append(
                    {
                        "baseline_steps": steps_value,
                        "backbone": "dinov3_vitl16_pixal3d",
                        "feature_mode": "bilinear",
                        "resolution": resolution,
                        "reference_resolution": 37,
                        "corruption": corruption,
                        "endpoint": endpoint,
                        **estimate,
                        "difference_from_37": difference["estimate"],
                        "difference_from_37_n_shoes_valid": difference[
                            "n_shoes_valid"
                        ],
                        "difference_from_37_ci_low": difference["ci_low"],
                        "difference_from_37_ci_high": difference["ci_high"],
                        **support,
                    }
                )
    return pd.DataFrame(records)


def summarize_naf_vs_bilinear(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute paired NAF-minus-bilinear effects on exact two-way support."""

    _require(
        frame,
        (
            *COMMON_SUPPORT_COLUMNS,
            "baseline_steps",
            "backbone",
            "feature_mode",
            "resolution",
            "s00",
            "s10",
            "abstained",
        ),
        "controlled metric frame",
    )
    records: list[dict[str, object]] = []
    for resolution in (74, 148, 296, 592):
        variants = {
            "bilinear": _dinov3l_variant(
                frame,
                feature_mode="bilinear",
                resolution=resolution,
            ),
            "naf_normal_guide": _dinov3l_variant(
                frame,
                feature_mode="naf_normal_guide",
                resolution=resolution,
            ),
        }
        if any(variant.empty for variant in variants.values()):
            raise ValueError(
                f"bilinear and NAF-normal resolution {resolution} are required"
            )
        for (
            steps_value,
            corruption,
            endpoints,
            common,
            support,
        ) in _controlled_support_groups(variants):
            for endpoint in endpoints:
                bilinear = _shoe_endpoint_values(
                    variants["bilinear"],
                    common,
                    endpoint,
                )
                naf = _shoe_endpoint_values(
                    variants["naf_normal_guide"],
                    common,
                    endpoint,
                )
                bilinear_summary = _summarize_shoes(
                    bilinear,
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                naf_summary = _summarize_shoes(
                    naf,
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                difference = _summarize_shoes(
                    _paired_shoe_difference(naf, bilinear),
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                records.append(
                    {
                        "baseline_steps": steps_value,
                        "backbone": "dinov3_vitl16_pixal3d",
                        "resolution": resolution,
                        "corruption": corruption,
                        "endpoint": endpoint,
                        "bilinear_estimate": bilinear_summary["estimate"],
                        "bilinear_n_shoes_valid": bilinear_summary[
                            "n_shoes_valid"
                        ],
                        "bilinear_ci_low": bilinear_summary["ci_low"],
                        "bilinear_ci_high": bilinear_summary["ci_high"],
                        "naf_estimate": naf_summary["estimate"],
                        "naf_n_shoes_valid": naf_summary["n_shoes_valid"],
                        "naf_ci_low": naf_summary["ci_low"],
                        "naf_ci_high": naf_summary["ci_high"],
                        "naf_minus_bilinear": difference["estimate"],
                        "naf_minus_bilinear_n_shoes_valid": difference[
                            "n_shoes_valid"
                        ],
                        "naf_minus_bilinear_ci_low": difference["ci_low"],
                        "naf_minus_bilinear_ci_high": difference["ci_high"],
                        **support,
                    }
                )
    return pd.DataFrame(records)


def _geometry_valid(frame: pd.DataFrame) -> pd.Series:
    abstained = (
        frame["abstained"].fillna(True).astype(bool)
        if "abstained" in frame
        else pd.Series(False, index=frame.index)
    )
    return (
        ~abstained
        & np.isfinite(
            pd.to_numeric(frame["geometry_effect"], errors="coerce")
        )
        & np.isfinite(pd.to_numeric(frame["overlap"], errors="coerce"))
    )


def _tuple_key(row: pd.Series, columns: Sequence[str]) -> tuple[object, ...]:
    return tuple(row[column] for column in columns)


def _geometry_variant(
    frame: pd.DataFrame,
    *,
    feature_mode: str,
    resolution: int,
) -> pd.DataFrame:
    return frame[
        (frame["backbone"] == "dinov3_vitl16_pixal3d")
        & (frame["feature_mode"] == feature_mode)
        & (pd.to_numeric(frame["resolution"], errors="coerce") == resolution)
    ].copy()


def summarize_geometry_low_high(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare the two frozen calibration envelopes on exact paired support."""

    _require(
        frame,
        (
            *GEOMETRY_SUPPORT_COLUMNS,
            "baseline_steps",
            "backbone",
            "feature_mode",
            "resolution",
            "geometry_effect",
            "overlap",
            "abstained",
        ),
        "geometry calibration frame",
    )
    _require_frozen_shoe_cohort(frame, "geometry calibration frame")
    _validated_baseline_steps(frame, "geometry calibration frame")
    variants = {
        "L37-bilinear": _geometry_variant(
            frame,
            feature_mode="bilinear",
            resolution=37,
        ),
        "L592-NAF": _geometry_variant(
            frame,
            feature_mode="naf_normal_guide",
            resolution=592,
        ),
    }
    if any(variant.empty for variant in variants.values()):
        raise ValueError("the two frozen geometry variants are required")
    support_sets: list[set[tuple[object, ...]]] = []
    for variant in variants.values():
        valid = variant[_geometry_valid(variant)]
        if valid.duplicated(list(GEOMETRY_SUPPORT_COLUMNS)).any():
            raise ValueError("geometry exact support keys are duplicate")
        support_sets.append(
            {
                _tuple_key(row, GEOMETRY_SUPPORT_COLUMNS)
                for _, row in valid.iterrows()
            }
        )
    union_all = set().union(*support_sets)
    common_all = set.intersection(*support_sets)
    raw_union = {
        _tuple_key(row, GEOMETRY_SUPPORT_COLUMNS)
        for variant in variants.values()
        for _, row in variant.iterrows()
    }
    step_by_key = {
        _tuple_key(row, GEOMETRY_SUPPORT_COLUMNS): int(
            row["baseline_steps"]
        )
        for _, row in frame.iterrows()
    }
    observed_kinds = {str(key[-2]) for key in raw_union}
    unknown_kinds = observed_kinds - set(GEOMETRY_MAGNITUDES)
    if unknown_kinds:
        raise ValueError(
            "geometry calibration frame has unknown perturbation kinds: "
            f"{sorted(unknown_kinds)!r}"
        )
    kinds_and_magnitudes = [
        (kind, magnitude)
        for kind in sorted(observed_kinds)
        for magnitude in GEOMETRY_MAGNITUDES[kind]
    ]
    records: list[dict[str, object]] = []
    for steps_value in FROZEN_BASELINE_DOMAIN:
        series_universe = {
            key[:-1]
            for key in raw_union
            if (
                steps_value == "all"
                or step_by_key[key] == steps_value
            )
        }
        for kind, magnitude in kinds_and_magnitudes:
            def selected(key: tuple[object, ...]) -> bool:
                return (
                    str(key[-2]) == kind
                    and float(key[-1]) == magnitude
                    and (
                        steps_value == "all"
                        or step_by_key[key] == steps_value
                    )
                )

            union = {key for key in union_all if selected(key)}
            common = {key for key in common_all if selected(key)}
            retention = len(common) / len(union) if union else 0.0
            kind_series = {
                series
                for series in series_universe
                if str(series[-1]) == kind
            }
            complete_series = 0
            expected_magnitudes = set(GEOMETRY_MAGNITUDES[kind])
            for series in kind_series:
                observed_common = {
                    float(key[-1])
                    for key in common_all
                    if key[:-1] == series
                }
                complete_series += int(
                    expected_magnitudes <= observed_common
                )
            completeness = (
                complete_series / len(kind_series)
                if kind_series
                else 0.0
            )
            shoes_by_variant: dict[str, pd.DataFrame] = {}
            for label, variant in variants.items():
                selected_rows = variant.copy()
                selected_rows["_support_key"] = selected_rows.apply(
                    lambda row: _tuple_key(row, GEOMETRY_SUPPORT_COLUMNS),
                    axis=1,
                )
                selected_rows = selected_rows[
                    selected_rows["_support_key"].isin(common)
                ].copy()
                selected_rows["value"] = pd.to_numeric(
                    selected_rows["geometry_effect"],
                    errors="coerce",
                ).abs()
                shoes_by_variant[label] = (
                    selected_rows.groupby(
                        "shoe_id",
                        dropna=False,
                        sort=True,
                    )["value"]
                    .mean()
                    .reset_index()
                )
            difference = _summarize_shoes(
                _paired_shoe_difference(
                    shoes_by_variant["L592-NAF"],
                    shoes_by_variant["L37-bilinear"],
                ),
                repeats=bootstrap_repeats,
                seed=seed,
            )
            for label, shoes in shoes_by_variant.items():
                estimate = _summarize_shoes(
                    shoes,
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                records.append(
                    {
                        "baseline_steps": steps_value,
                        "perturbation_kind": kind,
                        "magnitude": magnitude,
                        "variant": label,
                        "geometry_response": estimate["estimate"],
                        "geometry_response_n_shoes_valid": estimate[
                            "n_shoes_valid"
                        ],
                        "geometry_response_ci_low": estimate["ci_low"],
                        "geometry_response_ci_high": estimate["ci_high"],
                        "high_minus_low": difference["estimate"],
                        "high_minus_low_n_shoes_valid": difference[
                            "n_shoes_valid"
                        ],
                        "high_minus_low_ci_low": difference["ci_low"],
                        "high_minus_low_ci_high": difference["ci_high"],
                        "n_common_support": len(common),
                        "n_union_support": len(union),
                        "support_retention": retention,
                        "complete_series_coverage": completeness,
                    }
                )
    return pd.DataFrame(records)


def _stress_valid(frame: pd.DataFrame) -> pd.Series:
    abstained = (
        frame["abstained"].fillna(True).astype(bool)
        if "abstained" in frame
        else pd.Series(False, index=frame.index)
    )
    required = (
        "s00",
        "s10",
        "s11",
        "score",
        "s11_first_to_second_overlap",
        "s11_second_to_first_overlap",
    )
    finite = pd.Series(True, index=frame.index)
    for column in required:
        finite &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    return ~abstained & finite


def _fixed_feature_effect(frame: pd.DataFrame) -> pd.Series:
    derived = (
        pd.to_numeric(frame["s10"], errors="coerce")
        - pd.to_numeric(frame["s00"], errors="coerce")
    )
    if "feature_effect" in frame:
        emitted = pd.to_numeric(frame["feature_effect"], errors="coerce")
        comparable = np.isfinite(derived) & np.isfinite(emitted)
        if comparable.any() and not np.allclose(
            derived[comparable],
            emitted[comparable],
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "worker feature_effect differs from the fixed s10-s00 effect"
            )
    return derived


def _paired_stress_rows(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    set[tuple[object, ...]],
    set[tuple[object, ...]],
]:
    _require(
        frame,
        (
            *STRESS_SUPPORT_COLUMNS,
            "baseline_steps",
            "backbone",
            "feature_mode",
            "resolution",
            "s00",
            "s10",
            "s11",
            "score",
            "s11_first_to_second_overlap",
            "s11_second_to_first_overlap",
            "abstained",
        ),
        "stress metric frame",
    )
    _require_frozen_shoe_cohort(frame, "stress metric frame")
    _validated_baseline_steps(frame, "stress metric frame")
    low = _dinov3l_variant(
        frame,
        feature_mode="bilinear",
        resolution=37,
    )
    high = _dinov3l_variant(
        frame,
        feature_mode="naf_normal_guide",
        resolution=592,
    )
    if low.empty or high.empty:
        raise ValueError("stress L37-bilinear and L592-NAF rows are required")
    for label, variant in (("low", low), ("high", high)):
        if variant.duplicated(list(STRESS_SUPPORT_COLUMNS)).any():
            raise ValueError(f"stress {label} support keys are duplicate")
    low = low.copy()
    high = high.copy()
    low["_key"] = low.apply(
        lambda row: _tuple_key(row, STRESS_SUPPORT_COLUMNS),
        axis=1,
    )
    high["_key"] = high.apply(
        lambda row: _tuple_key(row, STRESS_SUPPORT_COLUMNS),
        axis=1,
    )
    low["_valid"] = _stress_valid(low)
    high["_valid"] = _stress_valid(high)
    low["_effect"] = _fixed_feature_effect(low)
    high["_effect"] = _fixed_feature_effect(high)
    paired = low.merge(
        high,
        on=list(STRESS_SUPPORT_COLUMNS),
        how="inner",
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    return (
        low,
        high,
        paired,
        set(low["_key"]),
        set(high["_key"]),
    )


def summarize_stress_reliability(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize paired low/high real-deformation validity and disagreement."""

    low, high, paired, low_keys, high_keys = _paired_stress_rows(frame)
    reliability_columns = (
        "first_alignment_class",
        "second_alignment_class",
        "first_center_shift_normalized",
        "second_center_shift_normalized",
        "first_bottom_shift_normalized",
        "second_bottom_shift_normalized",
        "first_width_ratio",
        "second_width_ratio",
        "first_same_shoe_identity",
        "second_same_shoe_identity",
    )
    _require(frame, reliability_columns, "stress reliability covariates")
    class_rank = {"strict": 0, "moderate": 1, "high_misalignment": 2}
    for column in ("first_alignment_class", "second_alignment_class"):
        low_column = f"{column}_low"
        high_column = f"{column}_high"
        if not (
            paired[low_column].isin(class_rank).all()
            and paired[high_column].isin(class_rank).all()
            and (paired[low_column] == paired[high_column]).all()
        ):
            raise ValueError("stress reliability class differs across feature variants")
    paired["_alignment_class"] = paired.apply(
        lambda row: max(
            (
                str(row["first_alignment_class_low"]),
                str(row["second_alignment_class_low"]),
            ),
            key=class_rank.__getitem__,
        ),
        axis=1,
    )
    paired["_max_center_shift"] = paired[
        [
            "first_center_shift_normalized_low",
            "second_center_shift_normalized_low",
        ]
    ].max(axis=1)
    paired["_max_bottom_shift"] = paired[
        [
            "first_bottom_shift_normalized_low",
            "second_bottom_shift_normalized_low",
        ]
    ].max(axis=1)
    paired["_max_width_deviation"] = (
        paired[
            ["first_width_ratio_low", "second_width_ratio_low"]
        ].astype(float)
        - 1.0
    ).abs().max(axis=1)
    paired["_min_identity"] = paired[
        ["first_same_shoe_identity_low", "second_same_shoe_identity_low"]
    ].min(axis=1)
    union = low_keys | high_keys
    common = low_keys & high_keys
    paired["low_symmetric_pass_rate"] = (
        paired["_valid_low"]
        & (
            paired[
                [
                    "s11_first_to_second_overlap_low",
                    "s11_second_to_first_overlap_low",
                ]
            ].min(axis=1)
            >= 0.08
        )
    ).astype(float)
    paired["high_symmetric_pass_rate"] = (
        paired["_valid_high"]
        & (
            paired[
                [
                    "s11_first_to_second_overlap_high",
                    "s11_second_to_first_overlap_high",
                ]
            ].min(axis=1)
            >= 0.08
        )
    ).astype(float)
    paired["both_finite_rate"] = (
        paired["_valid_low"] & paired["_valid_high"]
    ).astype(float)
    paired["low_finite_high_abstained_rate"] = (
        paired["_valid_low"] & ~paired["_valid_high"]
    ).astype(float)
    paired["low_abstained_high_finite_rate"] = (
        ~paired["_valid_low"] & paired["_valid_high"]
    ).astype(float)
    paired["both_abstained_rate"] = (
        ~paired["_valid_low"] & ~paired["_valid_high"]
    ).astype(float)
    paired["finite_low_high_disagreement"] = (
        paired["_effect_high"] - paired["_effect_low"]
    ).abs()
    paired.loc[
        ~(paired["_valid_low"] & paired["_valid_high"]),
        "finite_low_high_disagreement",
    ] = np.nan
    metrics = (
        "low_symmetric_pass_rate",
        "high_symmetric_pass_rate",
        "both_finite_rate",
        "low_finite_high_abstained_rate",
        "low_abstained_high_finite_rate",
        "both_abstained_rate",
        "finite_low_high_disagreement",
    )
    records: list[dict[str, object]] = []
    for steps_value in FROZEN_BASELINE_DOMAIN:
        selected = (
            paired
            if steps_value == "all"
            else paired[paired["baseline_steps_low"] == steps_value]
        )
        shoes = (
            selected.groupby("shoe_id", dropna=False, sort=True)[list(metrics)]
            .mean()
            .reset_index()
        )
        step_union = {
            key
            for key in union
            if (
                steps_value == "all"
                or int(
                    low.loc[low["_key"] == key, "baseline_steps"].iloc[0]
                    if key in low_keys
                    else high.loc[
                        high["_key"] == key,
                        "baseline_steps",
                    ].iloc[0]
                )
                == steps_value
            )
        }
        step_common = {
            key
            for key in common
            if (
                steps_value == "all"
                or int(
                    low.loc[low["_key"] == key, "baseline_steps"].iloc[0]
                )
                == steps_value
            )
        }
        retention = (
            len(step_common) / len(step_union)
            if step_union
            else 0.0
        )
        complete_series = {
            _tuple_key(row, STRESS_SUPPORT_COLUMNS)
            for _, row in selected[
                selected["_valid_low"] & selected["_valid_high"]
            ].iterrows()
        }
        completeness = (
            len(complete_series) / len(step_union)
            if step_union
            else 0.0
        )
        for metric in metrics:
            values = shoes[["shoe_id", metric]].rename(
                columns={metric: "value"}
            )
            summary = _summarize_shoes(
                values,
                repeats=bootstrap_repeats,
                seed=seed,
            )
            record = {
                "baseline_steps": steps_value,
                "metric": metric,
                **summary,
                "n_common_support": len(step_common),
                "n_union_support": len(step_union),
                "support_retention": retention,
                "complete_series_coverage": completeness,
            }
            for stratum in class_rank:
                stratum_rows = selected[
                    selected["_alignment_class"] == stratum
                ]
                stratum_shoes = (
                    stratum_rows.groupby(
                        "shoe_id", dropna=False, sort=True
                    )[metric]
                    .mean()
                    .reset_index(name="value")
                )
                stratum_summary = _summarize_shoes(
                    stratum_shoes,
                    repeats=bootstrap_repeats,
                    seed=seed,
                )
                record.update(
                    {
                        f"{stratum}_{key}": value
                        for key, value in stratum_summary.items()
                    }
                )
                record[f"{stratum}_n_pairs"] = len(stratum_rows)
            records.append(record)
        correlations = {
            "spearman_disagreement_vs_center_shift": "_max_center_shift",
            "spearman_disagreement_vs_bottom_shift": "_max_bottom_shift",
            "spearman_disagreement_vs_width_deviation": "_max_width_deviation",
            "spearman_disagreement_vs_identity": "_min_identity",
        }
        for metric, covariate in correlations.items():
            shoe_values = []
            for shoe_id, shoe_rows in selected.groupby(
                "shoe_id", dropna=False, sort=True
            ):
                usable = shoe_rows[
                    np.isfinite(shoe_rows["finite_low_high_disagreement"])
                    & np.isfinite(pd.to_numeric(shoe_rows[covariate], errors="coerce"))
                ]
                value = (
                    usable["finite_low_high_disagreement"].corr(
                        pd.to_numeric(usable[covariate], errors="coerce"),
                        method="spearman",
                    )
                    if (
                        len(usable) >= 2
                        and usable["finite_low_high_disagreement"].nunique() >= 2
                        and pd.to_numeric(
                            usable[covariate], errors="coerce"
                        ).nunique()
                        >= 2
                    )
                    else np.nan
                )
                shoe_values.append({"shoe_id": shoe_id, "value": value})
            records.append(
                {
                    "baseline_steps": steps_value,
                    "metric": metric,
                    **_summarize_shoes(
                        pd.DataFrame(
                            shoe_values,
                            columns=("shoe_id", "value"),
                        ),
                        repeats=bootstrap_repeats,
                        seed=seed,
                    ),
                    "n_common_support": len(step_common),
                    "n_union_support": len(step_union),
                    "support_retention": retention,
                    "complete_series_coverage": completeness,
                    **{
                        f"{stratum}_{key}": np.nan
                        for stratum in class_rank
                        for key in (
                            "estimate",
                            "n_shoes_valid",
                            "ci_low",
                            "ci_high",
                        )
                    },
                    **{
                        f"{stratum}_n_pairs": int(
                            (selected["_alignment_class"] == stratum).sum()
                        )
                        for stratum in class_rank
                    },
                }
            )
    return pd.DataFrame(records)


def _controlled_series_status(frame: pd.DataFrame) -> pd.DataFrame:
    variant_columns = (
        "backbone",
        "feature_track",
        "feature_mode",
        "resolution",
    )
    _require(
        frame,
        (
            *variant_columns,
            *COMMON_SUPPORT_COLUMNS,
            "baseline_steps",
            "s00",
            "s10",
            "abstained",
        ),
        "controlled metric frame",
    )
    _require_frozen_shoe_cohort(frame, "controlled metric frame")
    _validated_baseline_steps(frame, "controlled metric frame")
    working = frame.copy()
    working["_valid"] = _primary_valid(working)
    base_columns = [
        *variant_columns,
        "shoe_id",
        "quartet_id",
        "pair_id",
    ]
    valid_bases = {
        tuple(row[column] for column in base_columns)
        for _, row in working[
            (working["corruption"] == "clean") & working["_valid"]
        ].iterrows()
    }
    series_columns = [
        *variant_columns,
        "shoe_id",
        "quartet_id",
        "pair_id",
        "changed_view",
        "corruption",
        "baseline_steps",
    ]
    records: list[dict[str, object]] = []
    changed = working[working["corruption"] != "clean"]
    for key, rows in changed.groupby(
        series_columns,
        dropna=False,
        sort=True,
    ):
        values = dict(zip(series_columns, key))
        corruption = str(values["corruption"])
        expected = set(CONTROLLED_SEVERITIES[corruption])
        observed = set(
            pd.to_numeric(
                rows.loc[rows["_valid"], "severity"],
                errors="coerce",
            ).dropna()
        )
        base_key = tuple(values[column] for column in base_columns)
        records.append(
            {
                **values,
                "complete": base_key in valid_bases and expected <= observed,
            }
        )
    return pd.DataFrame(records)


def _controlled_coverage_values(
    frame: pd.DataFrame,
    *,
    baseline_steps: int | str = "all",
) -> tuple[float, float, int]:
    status = _controlled_series_status(frame)
    if baseline_steps != "all":
        status = status[status["baseline_steps"] == baseline_steps]
    if status.empty:
        return 0.0, 0.0, 0
    complete_series = float(status["complete"].mean())
    status = status.copy()
    pair_parts = status["pair_id"].astype(str).str.split("__")
    status["_first_view"] = pair_parts.str[-2]
    status["_second_view"] = pair_parts.str[-1]
    physical_columns = [
        "backbone",
        "feature_track",
        "feature_mode",
        "resolution",
        "quartet_id",
        "_first_view",
        "_second_view",
        "changed_view",
        "corruption",
        "baseline_steps",
    ]
    complete_shoe_sets = [
        frozenset(
            rows.loc[rows["complete"], "shoe_id"].dropna().astype(str)
        )
        for _, rows in status.groupby(
            physical_columns,
            dropna=False,
            sort=True,
        )
    ]
    all_four = float(
        np.mean(
            [
                shoe_ids == FROZEN_SHOE_IDS
                for shoe_ids in complete_shoe_sets
            ]
        )
    )
    valid_shoes = int(
        len(
            frozenset(
                status.loc[
                    status["complete"],
                    "shoe_id",
                ].dropna().astype(str)
            )
            & FROZEN_SHOE_IDS
        )
    )
    return complete_series, all_four, valid_shoes


def summarize_support_coverage(
    preflight: pd.DataFrame,
    controlled: pd.DataFrame,
    stress: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize clean adjacent support by shoe and angular-baseline step."""

    _require(
        preflight,
        (
            "shoe_id",
            "quartet_id",
            "pair_id",
            "baseline_steps",
            "first_to_second_overlap",
            "second_to_first_overlap",
            "minimum_directional_overlap",
            "symmetric_pass",
        ),
        "geometry preflight frame",
    )
    _require_frozen_shoe_cohort(preflight, "geometry preflight frame")
    _validated_baseline_steps(preflight, "geometry preflight frame")
    _require_frozen_shoe_cohort(stress, "stress metric frame")
    _validated_baseline_steps(stress, "stress metric frame")
    working = preflight.copy()
    working["symmetric_coverage"] = (
        working["symmetric_pass"].fillna(False).astype(bool).astype(float)
    )
    metrics = (
        "symmetric_coverage",
        "first_to_second_overlap",
        "second_to_first_overlap",
        "minimum_directional_overlap",
    )
    records: list[dict[str, object]] = []
    for steps_value in FROZEN_BASELINE_DOMAIN:
        selected = (
            working
            if steps_value == "all"
            else working[working["baseline_steps"] == steps_value]
        )
        complete, _all_four, _valid_shoes = _controlled_coverage_values(
            controlled,
            baseline_steps=steps_value,
        )
        shoes = (
            selected.groupby("shoe_id", dropna=False, sort=True)[list(metrics)]
            .mean()
            .reset_index()
        )
        for metric in metrics:
            values = shoes[["shoe_id", metric]].rename(
                columns={metric: "value"}
            )
            summary = _summarize_shoes(
                values,
                repeats=bootstrap_repeats,
                seed=seed,
            )
            records.append(
                {
                    "baseline_steps": steps_value,
                    "metric": metric,
                    **summary,
                    "n_common_support": len(selected),
                    "n_union_support": len(selected),
                    "support_retention": (
                        1.0 if len(selected) else 0.0
                    ),
                    "complete_series_coverage": complete,
                }
            )
    return pd.DataFrame(records)


def summarize_common_support(
    backbone: pd.DataFrame,
    resolution: pd.DataFrame,
    naf: pd.DataFrame,
    geometry: pd.DataFrame,
    stress: pd.DataFrame,
) -> pd.DataFrame:
    """Collect each frozen comparison's exact support and coverage diagnostics."""

    records: list[dict[str, object]] = []

    def append_groups(
        source: pd.DataFrame,
        comparison: str,
        group_columns: Sequence[str],
        shoe_columns: Sequence[str],
    ) -> None:
        for key, rows in source.groupby(
            list(group_columns),
            dropna=False,
            sort=True,
        ):
            values = key if isinstance(key, tuple) else (key,)
            first = rows.iloc[0]
            valid_counts = [
                pd.to_numeric(rows[column], errors="coerce").min()
                for column in shoe_columns
                if column in rows
            ]
            records.append(
                {
                    "comparison": comparison,
                    **dict(zip(group_columns, values)),
                    "n_shoes_valid": (
                        int(min(valid_counts)) if valid_counts else 0
                    ),
                    "n_common_support": int(first["n_common_support"]),
                    "n_union_support": int(first["n_union_support"]),
                    "support_retention": float(first["support_retention"]),
                    "complete_series_coverage": float(
                        first["complete_series_coverage"]
                    ),
                }
            )

    append_groups(
        backbone,
        "four_backbone_common37",
        ("baseline_steps", "corruption"),
        ("n_shoes_valid",),
    )
    append_groups(
        resolution,
        "five_resolution_bilinear",
        ("baseline_steps", "corruption"),
        ("n_shoes_valid", "difference_from_37_n_shoes_valid"),
    )
    for resolution_value in (74, 148, 296, 592):
        append_groups(
            naf[naf["resolution"] == resolution_value],
            f"naf_vs_bilinear_{resolution_value}",
            ("baseline_steps", "corruption"),
            (
                "bilinear_n_shoes_valid",
                "naf_n_shoes_valid",
                "naf_minus_bilinear_n_shoes_valid",
            ),
        )
    append_groups(
        geometry,
        "geometry_low_high",
        ("baseline_steps", "perturbation_kind", "magnitude"),
        (
            "geometry_response_n_shoes_valid",
            "high_minus_low_n_shoes_valid",
        ),
    )
    append_groups(
        stress,
        "stress_low_high",
        ("baseline_steps",),
        ("n_shoes_valid",),
    )
    return pd.DataFrame(records)


def _overall_stress_pass(frame: pd.DataFrame) -> tuple[float, int]:
    _low, _high, paired, _low_keys, _high_keys = _paired_stress_rows(
        frame
    )
    low_pass = (
        paired["_valid_low"]
        & (
            paired[
                [
                    "s11_first_to_second_overlap_low",
                    "s11_second_to_first_overlap_low",
                ]
            ].min(axis=1)
            >= 0.08
        )
    )
    high_pass = (
        paired["_valid_high"]
        & (
            paired[
                [
                    "s11_first_to_second_overlap_high",
                    "s11_second_to_first_overlap_high",
                ]
            ].min(axis=1)
            >= 0.08
        )
    )
    working = pd.concat(
        [
            paired[["shoe_id"]].assign(passed=low_pass.astype(float)),
            paired[["shoe_id"]].assign(passed=high_pass.astype(float)),
        ],
        ignore_index=True,
    )
    shoes = (
        working.groupby("shoe_id", dropna=False, sort=True)["passed"]
        .mean()
        .reset_index()
    )
    return _safe_mean(shoes["passed"]), int(len(shoes))


def summarize_analyzability(
    controlled: pd.DataFrame,
    geometry: pd.DataFrame,
    stress: pd.DataFrame,
    common_support: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate frozen descriptive analyzability criteria from aggregate rows."""

    _require_frozen_shoe_cohort(geometry, "geometry calibration frame")
    _validated_baseline_steps(geometry, "geometry calibration frame")
    complete, all_four, valid_shoes = _controlled_coverage_values(controlled)
    controlled_support = common_support[
        common_support["comparison"].isin(_CONTROLLED_SUPPORT_COMPARISONS)
    ]
    finite_retentions = pd.to_numeric(
        controlled_support["support_retention"],
        errors="coerce",
    )
    minimum_retention = (
        float(finite_retentions[np.isfinite(finite_retentions)].min())
        if np.isfinite(finite_retentions).any()
        else float("nan")
    )
    stress_rate, stress_shoes = _overall_stress_pass(stress)
    common_valid_shoes = (
        int(
            pd.to_numeric(
                controlled_support["n_shoes_valid"],
                errors="coerce",
            ).min()
        )
        if not controlled_support.empty
        else 0
    )
    shared = {
        "support_retention": minimum_retention,
        "complete_series_coverage": complete,
    }
    records = [
        {
            "criterion": "controlled_complete_series_coverage",
            "observed": complete,
            "threshold": 0.8,
            "comparison": ">=",
            "passed": bool(np.isfinite(complete) and complete >= 0.8),
            "n_shoes_valid": valid_shoes,
            **shared,
        },
        {
            "criterion": "all_four_shoe_primary_cell_rate",
            "observed": all_four,
            "threshold": 0.8,
            "comparison": ">=",
            "passed": bool(np.isfinite(all_four) and all_four >= 0.8),
            "n_shoes_valid": valid_shoes,
            **shared,
        },
        {
            "criterion": "minimum_common_support_retention",
            "observed": minimum_retention,
            "threshold": 0.7,
            "comparison": ">=",
            "passed": bool(
                np.isfinite(minimum_retention)
                and minimum_retention >= 0.7
            ),
            "n_shoes_valid": common_valid_shoes,
            **shared,
        },
        {
            "criterion": "stress_symmetric_pass_rate",
            "observed": stress_rate,
            "threshold": 0.7,
            "comparison": ">=",
            "passed": bool(
                np.isfinite(stress_rate) and stress_rate >= 0.7
            ),
            "n_shoes_valid": stress_shoes,
            **shared,
        },
    ]
    expected_counts = {
        "controlled_completed_rows": (len(controlled), 31_104),
        "geometry_calibration_completed_rows": (len(geometry), 6_528),
        "stress_completed_rows": (len(stress), 1_152),
    }
    for criterion, (actual, expected) in expected_counts.items():
        source = (
            controlled
            if criterion.startswith("controlled")
            else geometry
            if criterion.startswith("geometry")
            else stress
        )
        records.append(
            {
                "criterion": criterion,
                "observed": actual,
                "threshold": expected,
                "comparison": "==",
                "passed": actual == expected,
                "n_shoes_valid": int(source["shoe_id"].nunique()),
                **shared,
            }
        )
    result = pd.DataFrame(records)
    result["n_common_support"] = (
        int(common_support["n_common_support"].min())
        if not common_support.empty
        else 0
    )
    result["n_union_support"] = (
        int(common_support["n_union_support"].min())
        if not common_support.empty
        else 0
    )
    return result


def _finite_rows(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[
        np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    ].copy()


def _support_note(frame: pd.DataFrame) -> str:
    shoe_columns = [
        column
        for column in frame.columns
        if "n_shoes_valid" in column
    ]
    counts = [
        int(value)
        for column in shoe_columns
        for value in pd.to_numeric(frame[column], errors="coerce")
        if np.isfinite(value)
    ]
    valid = (
        f"{min(counts)}-{max(counts)}/4"
        if counts and min(counts) != max(counts)
        else f"{counts[0]}/4"
        if counts
        else "0/4"
    )
    retention = pd.to_numeric(
        frame.get("support_retention", pd.Series(dtype=float)),
        errors="coerce",
    )
    complete = pd.to_numeric(
        frame.get("complete_series_coverage", pd.Series(dtype=float)),
        errors="coerce",
    )
    finite_retention = retention[np.isfinite(retention)]
    finite_complete = complete[np.isfinite(complete)]
    return (
        f"valid shoes: {valid}\n"
        f"support retention: "
        f"{float(finite_retention.min()):.1%}"
        if len(finite_retention)
        else f"valid shoes: {valid}\nsupport retention: n/a"
    ) + (
        f"\ncomplete-series coverage: {float(finite_complete.min()):.1%}"
        if len(finite_complete)
        else "\ncomplete-series coverage: n/a"
    )


def _annotate_support(axis: Any, frame: pd.DataFrame) -> None:
    axis.text(
        0.99,
        0.01,
        _support_note(frame),
        ha="right",
        va="bottom",
        fontsize=7,
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )
    axis.grid(alpha=0.25)


def _numeric_step_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_step"] = pd.to_numeric(
        result["baseline_steps"],
        errors="coerce",
    )
    return result[np.isfinite(result["_step"])]


def _plot_lines(
    axis: Any,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    groups: Sequence[str],
) -> None:
    if frame.empty or y not in frame:
        axis.text(0.5, 0.5, "No estimable observations", ha="center")
        return
    group_columns = [column for column in groups if column in frame]
    iterator = (
        frame.groupby(group_columns, dropna=False, sort=True)
        if group_columns
        else (("estimate", frame),)
    )
    for key, rows in iterator:
        values = key if isinstance(key, tuple) else (key,)
        label = (
            " / ".join(
                f"{column}={value}"
                for column, value in zip(group_columns, values)
            )
            if group_columns
            else "estimate"
        )
        rows = _finite_rows(rows, y).sort_values(x)
        if rows.empty:
            continue
        axis.plot(rows[x], rows[y], marker="o", label=label)
    if len(axis.lines) <= 12 and axis.lines:
        axis.legend(fontsize=6)


def _build_support_figure(
    tables: dict[str, pd.DataFrame],
    plt: Any,
) -> Any:
    support = tables["support_coverage"]
    support_steps = _numeric_step_rows(support)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    axis = axes[0, 0]
    symmetric = support_steps[
        support_steps["metric"] == "symmetric_coverage"
    ]
    _plot_lines(axis, symmetric, x="_step", y="estimate", groups=())
    axis.set_title("Symmetric clean-pair coverage")
    axis.set_xlabel("baseline steps")
    axis.set_ylabel("coverage")
    _annotate_support(axis, symmetric)
    axis = axes[0, 1]
    directional = support_steps[
        support_steps["metric"].isin(
            ("first_to_second_overlap", "second_to_first_overlap")
        )
    ]
    _plot_lines(
        axis,
        directional,
        x="_step",
        y="estimate",
        groups=("metric",),
    )
    axis.set_title("Directional overlap")
    axis.set_xlabel("baseline steps")
    axis.set_ylabel("overlap")
    _annotate_support(axis, directional)
    axis = axes[0, 2]
    minimum = support_steps[
        support_steps["metric"] == "minimum_directional_overlap"
    ]
    _plot_lines(axis, minimum, x="_step", y="estimate", groups=())
    axis.set_title("Baseline-step support")
    axis.set_xlabel("baseline steps")
    axis.set_ylabel("minimum directional overlap")
    _annotate_support(axis, minimum)
    figure.suptitle(f"Adjacent support — {PLOT_CAMERA_LIMITATION}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _build_backbone_figure(
    tables: dict[str, pd.DataFrame],
    plt: Any,
) -> Any:
    backbone = tables["backbone_common37"]
    backbone_all = backbone[backbone["baseline_steps"] == "all"].copy()
    panels = (
        (
            "Identity-swap clean/corrupt AUROC",
            backbone_all[
                backbone_all["endpoint"] == "clean_corrupt_auroc"
            ],
        ),
        (
            "Translation/TPS severity Spearman",
            backbone_all[
                backbone_all["endpoint"] == "severity_spearman"
            ],
        ),
        (
            "Translation/TPS pairwise rank accuracy",
            backbone_all[
                backbone_all["endpoint"] == "pairwise_rank_accuracy"
            ],
        ),
        (
            "JPEG signed and absolute nuisance delta",
            backbone_all[
                backbone_all["endpoint"].isin(
                    ("nuisance_signed_delta", "nuisance_absolute_delta")
                )
            ],
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    backbone_positions = {
        value: index for index, value in enumerate(COMMON37_BACKBONES)
    }
    for axis, (title, panel) in zip(axes.flat, panels):
        panel = panel.copy()
        panel["_backbone_position"] = panel["backbone"].map(
            backbone_positions
        )
        _plot_lines(
            axis,
            panel,
            x="_backbone_position",
            y="estimate",
            groups=("corruption", "endpoint"),
        )
        axis.set_xticks(
            range(len(COMMON37_BACKBONES)),
            COMMON37_BACKBONES,
            rotation=20,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylabel("endpoint estimate")
        _annotate_support(axis, panel)
    figure.suptitle("Four common-37 backbones on exact common support")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _build_resolution_figure(
    tables: dict[str, pd.DataFrame],
    plt: Any,
) -> Any:
    resolution = tables["resolution_bilinear"]
    resolution_all = resolution[
        resolution["baseline_steps"] == "all"
    ]
    naf = tables["naf_vs_bilinear"]
    naf_all = naf[naf["baseline_steps"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    _plot_lines(
        axes[0, 0],
        resolution_all,
        x="resolution",
        y="estimate",
        groups=("corruption", "endpoint"),
    )
    axes[0, 0].set_title("DINOv3-L bilinear resolution curves")
    axes[0, 0].set_xlabel("target resolution")
    axes[0, 0].set_ylabel("paired-support endpoint")
    _annotate_support(axes[0, 0], resolution_all)
    _plot_lines(
        axes[0, 1],
        naf_all,
        x="resolution",
        y="naf_minus_bilinear",
        groups=("corruption", "endpoint"),
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Paired NAF-minus-bilinear differences")
    axes[0, 1].set_xlabel("target resolution")
    axes[0, 1].set_ylabel("paired shoe-level difference")
    _annotate_support(axes[0, 1], naf_all)
    figure.suptitle("DINOv3-L resolution and NAF")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def _build_geometry_figure(
    tables: dict[str, pd.DataFrame],
    plt: Any,
) -> Any:
    geometry = tables["geometry_low_high"]
    geometry_all = geometry[geometry["baseline_steps"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), squeeze=False)
    for axis, variant in zip(axes.flat, ("L37-bilinear", "L592-NAF")):
        panel = geometry_all[geometry_all["variant"] == variant]
        _plot_lines(
            axis,
            panel,
            x="magnitude",
            y="geometry_response",
            groups=("perturbation_kind",),
        )
        axis.set_title(f"{variant} geometry-response envelope")
        axis.set_xlabel("raw perturbation magnitude")
        axis.set_ylabel("absolute geometry effect")
        _annotate_support(axis, panel)
    figure.suptitle("Local sensitivity around fixed/precomputed DA3 geometry")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _build_stress_figure(
    tables: dict[str, pd.DataFrame],
    plt: Any,
) -> Any:
    stress = tables["stress_reliability"]
    stress_steps = _numeric_step_rows(stress)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    pass_rows = stress_steps[
        stress_steps["metric"].isin(
            ("low_symmetric_pass_rate", "high_symmetric_pass_rate")
        )
    ]
    _plot_lines(
        axes[0, 0],
        pass_rows,
        x="_step",
        y="estimate",
        groups=("metric",),
    )
    axes[0, 0].set_title("Stress symmetric pass rates")
    axes[0, 0].set_xlabel("baseline steps")
    axes[0, 0].set_ylabel("pass rate")
    _annotate_support(axes[0, 0], pass_rows)
    validity = stress_steps[
        stress_steps["metric"].isin(
            (
                "both_finite_rate",
                "low_finite_high_abstained_rate",
                "low_abstained_high_finite_rate",
                "both_abstained_rate",
            )
        )
    ]
    _plot_lines(
        axes[0, 1],
        validity,
        x="_step",
        y="estimate",
        groups=("metric",),
    )
    axes[0, 1].set_title("Low/high validity states")
    axes[0, 1].set_xlabel("baseline steps")
    axes[0, 1].set_ylabel("state rate")
    _annotate_support(axes[0, 1], validity)
    disagreement = stress_steps[
        stress_steps["metric"] == "finite_low_high_disagreement"
    ]
    _plot_lines(
        axes[1, 0],
        disagreement,
        x="_step",
        y="estimate",
        groups=(),
    )
    axes[1, 0].set_title("Finite low/high disagreement")
    axes[1, 0].set_xlabel("baseline steps")
    axes[1, 0].set_ylabel("absolute feature-effect difference")
    _annotate_support(axes[1, 0], disagreement)
    high_all = stress[
        (stress["baseline_steps"] == "all")
        & (stress["metric"] == "high_symmetric_pass_rate")
    ].iloc[0]
    strata = ("strict", "moderate", "high_misalignment")
    stratum_values = np.asarray(
        [high_all[f"{stratum}_estimate"] for stratum in strata],
        dtype=float,
    )
    finite = np.isfinite(stratum_values)
    axes[1, 1].bar(
        np.arange(len(strata))[finite],
        stratum_values[finite],
    )
    axes[1, 1].set_xticks(
        np.arange(len(strata)),
        ("strict", "moderate", "high misalignment"),
        rotation=25,
        ha="right",
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("L592 pass rate by generation reliability")
    axes[1, 1].set_ylabel("symmetric pass rate")
    pair_counts = ", ".join(
        f"{label}: n={int(high_all[f'{stratum}_n_pairs'])}"
        for label, stratum in zip(
            ("strict", "moderate", "high"),
            strata,
            strict=True,
        )
    )
    axes[1, 1].text(
        0.01,
        0.02,
        pair_counts,
        transform=axes[1, 1].transAxes,
        fontsize=8,
        va="bottom",
    )
    figure.suptitle("Independent-deformation reliability stress")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def _build_core_figures(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    builders = (
        _build_support_figure,
        _build_backbone_figure,
        _build_resolution_figure,
        _build_geometry_figure,
        _build_stress_figure,
    )
    return {
        filename: builder(tables, plt)
        for filename, builder in zip(FIGURE_FILENAMES, builders)
    }


def _write_tables(
    tables_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    for stem in TABLE_STEMS:
        publishable = tables[stem].copy()
        if "baseline_steps" in publishable:
            publishable["baseline_steps"] = (
                publishable["baseline_steps"].astype(str)
            )
        publishable.to_csv(tables_dir / f"{stem}.csv", index=False)
        publishable.to_parquet(
            tables_dir / f"{stem}.parquet",
            index=False,
        )


def _representative_estimate(
    frame: pd.DataFrame,
    query: str,
    column: str,
) -> str:
    value = _representative_value(frame, query, column)
    return f"{value:.4f}" if np.isfinite(value) else "not estimable"


def _representative_value(
    frame: pd.DataFrame,
    query: str,
    column: str,
) -> float:
    selected = frame.query(query)
    if selected.empty:
        return float("nan")
    value = pd.to_numeric(selected[column], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def _module_design_interpretation(stress: pd.DataFrame) -> str:
    selected = stress.query(
        "baseline_steps == 'all' and "
        "metric == 'low_finite_high_abstained_rate'"
    )
    value = (
        pd.to_numeric(selected["estimate"], errors="coerce").iloc[0]
        if not selected.empty
        else float("nan")
    )
    if not np.isfinite(value):
        return (
            "The low-finite/high-abstained rate was not estimable, so this "
            "experiment does not provide direct evidence for preserving a "
            "valid semantic low-resolution path while gating an invalid "
            "high-resolution path."
        )
    if float(value) <= 0.0:
        return (
            "No low-finite/high-abstained cases were observed (all-baseline "
            f"rate `{float(value):.4f}`). Therefore this experiment does not "
            "provide direct evidence for preserving a valid semantic "
            "low-resolution path while gating an invalid high-resolution "
            "path. The observed module-design evidence is instead the "
            "reliability-stratified pass rate and the finite low/high "
            "disagreement."
        )
    return (
        "Low-finite/high-abstained cases were observed; the all-baseline "
        f"observed rate was `{float(value):.4f}`. This is a candidate gating "
        "signal for preserving low-resolution semantic evidence while "
        "suppressing invalid high-resolution evidence, but this experiment "
        "does not train or validate that learnable module."
    )


def _module_design_interpretation_ko(stress: pd.DataFrame) -> str:
    value = _representative_value(
        stress,
        "baseline_steps == 'all' and "
        "metric == 'low_finite_high_abstained_rate'",
        "estimate",
    )
    if not np.isfinite(value):
        return (
            "low-finite/high-abstained rate를 추정할 수 없으므로 "
            "semantic-low/pixel-high hard gate는 직접 검증되지 않았다. 즉, "
            "해상도 hard gate가 직접 검증된 것은 아니다."
        )
    if value <= 0.0:
        return (
            "low-finite/high-abstained case가 관찰되지 않았으므로 "
            "semantic-low/pixel-high hard gate는 직접 검증되지 않았다. 즉, "
            "해상도 hard gate가 직접 검증된 것은 아니다."
        )
    return (
        f"low-finite/high-abstained rate는 `{value:.4f}`이지만, 이 관찰만으로 "
        "semantic-low/pixel-high hard gate를 학습 또는 검증하지 않았다. 즉, "
        "해상도 hard gate가 직접 검증된 것은 아니다."
    )


def _write_report(
    destination: Path,
    *,
    tables: dict[str, pd.DataFrame],
    controlled_rows: int,
    geometry_rows: int,
    stress_rows: int,
    bootstrap_repeats: int,
    seed: int,
    generation: Path,
    artifact_paths: Sequence[Path],
) -> None:
    support = tables["support_coverage"]
    backbone = tables["backbone_common37"]
    resolution = tables["resolution_bilinear"]
    naf = tables["naf_vs_bilinear"]
    geometry = tables["geometry_low_high"]
    stress = tables["stress_reliability"]
    analyzability = tables["analyzability"]
    symmetric = _representative_estimate(
        support,
        "baseline_steps == 'all' and metric == 'symmetric_coverage'",
        "estimate",
    )
    support_by_step = {
        step: _representative_estimate(
            support,
            f"baseline_steps == {step} and "
            "metric == 'minimum_directional_overlap'",
            "estimate",
        )
        for step in ("1", "2", "3")
    }
    backbone_endpoints = {
        (backbone_name, corruption, endpoint): _representative_estimate(
            backbone,
            "baseline_steps == 'all' and "
            f"backbone == '{backbone_name}' and "
            f"corruption == '{corruption}' and endpoint == '{endpoint}'",
            "estimate",
        )
        for backbone_name in (
            "dino_vitb16",
            "dinov2_vitb14",
            "dinov3_vitb16",
            "dinov3_vitl16_pixal3d",
        )
        for corruption, endpoint in (
            ("identity_swap", "clean_corrupt_auroc"),
            ("translation", "severity_spearman"),
            ("translation", "pairwise_rank_accuracy"),
            ("thin_plate_warp", "severity_spearman"),
            ("thin_plate_warp", "pairwise_rank_accuracy"),
            ("jpeg", "nuisance_absolute_delta"),
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
    jpeg_by_resolution = {
        resolution_value: _representative_estimate(
            resolution,
            "baseline_steps == 'all' and "
            f"resolution == {resolution_value} and corruption == 'jpeg' and "
            "endpoint == 'nuisance_absolute_delta'",
            "estimate",
        )
        for resolution_value in (37, 592)
    }
    naf_minus_bilinear = {
        label: _representative_estimate(
            naf,
            "baseline_steps == 'all' and resolution == 592 and "
            f"corruption == '{corruption}' and endpoint == '{endpoint}'",
            "naf_minus_bilinear",
        )
        for label, corruption, endpoint in (
            ("JPEG absolute delta", "jpeg", "nuisance_absolute_delta"),
            ("TPS Spearman", "thin_plate_warp", "severity_spearman"),
        )
    }
    geometry_responses = {
        (perturbation_kind, magnitude, variant): _representative_estimate(
            geometry,
            "baseline_steps == 'all' and "
            f"perturbation_kind == '{perturbation_kind}' and "
            f"magnitude == {magnitude} and variant == '{variant}'",
            "geometry_response",
        )
        for perturbation_kind, magnitudes in (
            ("depth_scale", (1.01, 1.05)),
            ("relative_depth_noise", (0.005, 0.05)),
            ("relative_translation", (0.0025, 0.02)),
            ("rotation", (0.25, 1.0, 5.0)),
        )
        for magnitude in magnitudes
        for variant in ("L37-bilinear", "L592-NAF")
    }
    depth_boundary_query = (
        "baseline_steps == 'all' and perturbation_kind == 'depth_scale' and "
        "magnitude == 1.1 and variant == 'L37-bilinear'"
    )
    depth_boundary_common = _representative_value(
        geometry,
        depth_boundary_query,
        "n_common_support",
    )
    depth_boundary_union = _representative_value(
        geometry,
        depth_boundary_query,
        "n_union_support",
    )
    depth_boundary_common_text = (
        str(int(depth_boundary_common))
        if np.isfinite(depth_boundary_common)
        else "not estimable"
    )
    depth_boundary_union_text = (
        str(int(depth_boundary_union))
        if np.isfinite(depth_boundary_union)
        else "not estimable"
    )
    disagreement_by_step = {
        step: _representative_estimate(
            stress,
            f"baseline_steps == {step} and "
            "metric == 'finite_low_high_disagreement'",
            "estimate",
        )
        for step in ("1", "2", "3")
    }
    module_design_interpretation_ko = _module_design_interpretation_ko(stress)
    high_pass_strata = {
        stratum: _representative_estimate(
            stress,
            "baseline_steps == 'all' and "
            "metric == 'high_symmetric_pass_rate'",
            f"{stratum}_estimate",
        )
        for stratum in ("strict", "moderate", "high_misalignment")
    }
    reliability_correlations = {
        label: _representative_estimate(
            stress,
            f"baseline_steps == 'all' and metric == '{metric}'",
            "estimate",
        )
        for label, metric in (
            ("center shift", "spearman_disagreement_vs_center_shift"),
            ("bottom shift", "spearman_disagreement_vs_bottom_shift"),
            ("width deviation", "spearman_disagreement_vs_width_deviation"),
            ("identity score", "spearman_disagreement_vs_identity"),
        )
    }
    incomplete = analyzability.loc[
        ~analyzability["passed"].astype(bool),
        "criterion",
    ].astype(str).tolist()
    artifact_lines = [
        f"- `{path.relative_to(destination.parent)}` — sha256 `{sha256_file(path)}`"
        for path in sorted(artifact_paths)
    ]
    input_lines = [
        f"- aggregate `{path.name}` — sha256 `{sha256_file(path)}`"
        for path in sorted(
            generation / name
            for name in (
                "controlled_fidelity.parquet",
                "geometry_calibration.parquet",
                "real_deformation_stress.parquet",
                "summary_preview.csv",
            )
        )
    ]
    backbone_lines = [
        f"- `{name}`: identity AUROC "
        f"`{backbone_endpoints[(name, 'identity_swap', 'clean_corrupt_auroc')]}`; "
        "translation Spearman "
        f"`{backbone_endpoints[(name, 'translation', 'severity_spearman')]}`, "
        "pairwise rank accuracy "
        f"`{backbone_endpoints[(name, 'translation', 'pairwise_rank_accuracy')]}`; "
        "TPS Spearman "
        f"`{backbone_endpoints[(name, 'thin_plate_warp', 'severity_spearman')]}`, "
        "pairwise rank accuracy "
        f"`{backbone_endpoints[(name, 'thin_plate_warp', 'pairwise_rank_accuracy')]}`; "
        "JPEG absolute delta "
        f"`{backbone_endpoints[(name, 'jpeg', 'nuisance_absolute_delta')]}`."
        for name in COMMON37_BACKBONES
    ]
    depth_boundary_line = (
        "depth_scale 10%에서는 common finite support "
        f"`{depth_boundary_common_text}` / union finite support "
        f"`{depth_boundary_union_text}`로, abstention boundary다."
        if np.isfinite(depth_boundary_common)
        and depth_boundary_common <= 0.0
        else "depth_scale 10%에서는 common finite support "
        f"`{depth_boundary_common_text}` / union finite support "
        f"`{depth_boundary_union_text}`가 관찰되어 완전 abstention boundary는 아니다."
    )
    lines = [
        "# 인접 Upper 16-view DINO Feature Fidelity 및 Reliability 연구 보고서",
        "",
        f"## 1. {REPORT_SECTION_TITLES[0]}",
        "",
        "| 연구 질문 | 관찰된 결과 | 해석 |",
        "|---|---|---|",
        "| 인접 뷰가 geometry confound를 줄이는가? | "
        f"대칭 coverage `{symmetric}`, step 1/2/3 최소 overlap "
        f"`{support_by_step['1']}` / `{support_by_step['2']}` / "
        f"`{support_by_step['3']}` | 인접 upper-view support는 관찰했지만 실제 "
        "confound 감소는 직접 검증하지 않았다. |",
        "| 어떤 DINO backbone이 적합한가? | translation Spearman: "
        + ", ".join(
            f"{name} "
            f"`{backbone_endpoints[(name, 'translation', 'severity_spearman')]}`"
            for name in COMMON37_BACKBONES
        )
        + " | 공통 support 안의 조건부 비교이며 일반적 우월성 주장은 아니다. |",
        "| 높은 feature resolution이 항상 유리한가? | L37/L592 translation "
        f"`{resolution_endpoints[(37, 'translation')]}` / "
        f"`{resolution_endpoints[(592, 'translation')]}` | endpoint별로 "
        "확인해야 하며 항상이라는 결론은 낼 수 없다. |",
        "| DA3 geometry 오차와 feature 차이를 분리할 수 있는가? | 국소 perturbation "
        "반응을 별도 측정했다. | precomputed camera 주변 민감도이지 절대 오차 추정은 아니다. |",
        "| View Reliability Module을 무엇으로 학습해야 하는가? | reliability strata의 "
        "pass rate와 유한 L37/L592 disagreement를 관찰했다. | 연속 reliability "
        "weighting은 지지되지만 hard gate는 직접 검증되지 않았다. |",
        "",
        f"## 2. {REPORT_SECTION_TITLES[1]}",
        "",
        f"카메라 계약은 {CAMERA_LIMITATION}이다. controlled rows: "
        f"{controlled_rows:,}; geometry calibration rows: {geometry_rows:,}; "
        f"stress rows: {stress_rows:,}; total metric rows: "
        f"{controlled_rows + geometry_rows + stress_rows:,}; seed: `{seed}`; "
        f"bootstrap repeats: `{bootstrap_repeats}`.",
        "",
        "모든 구간은 네 shoe cluster만 재표본화한다. quartet, pair, changed-view "
        "배치와 severity는 반복 측정으로 남으므로, n=4에서의 구간과 순위는 "
        "기술적 요약이다. 결측 또는 abstention을 0점으로 대체하지 않았다.",
        "",
        f"## 3. {REPORT_SECTION_TITLES[2]}",
        "",
        "**그림 읽는 법.** 대칭 coverage와 direction별 overlap을 all 및 baseline "
        "step별로 읽는다; 낮은 support는 낮은 similarity가 아니라 관찰 불가를 뜻한다.",
        "",
        "![인접 뷰의 support와 overlap](figures/01_adjacent_support.png)",
        "",
        "**관찰된 결과.** step 1/2/3의 최소 direction overlap은 각각 "
        f"`{support_by_step['1']}`, `{support_by_step['2']}`, "
        f"`{support_by_step['3']}`이고 all-baseline 대칭 coverage는 "
        f"`{symmetric}`이다.",
        "",
        "**해석.** 인접 upper-view support는 관찰했지만 실제 confound 감소는 직접 "
        "검증하지 않았다. support 자체도 완전한 geometry 정답은 아니다.",
        "",
        f"## 4. {REPORT_SECTION_TITLES[3]}",
        "",
        "**그림 읽는 법.** 네 backbone은 exact shared "
        "`shoe_id × quartet_id × pair_id × changed_view × corruption × severity` "
        "support의 bilinear 37 결과만 비교한다. 네 panel은 identity AUROC, "
        "translation/TPS severity Spearman, 두 corruption의 pairwise rank "
        "accuracy, JPEG signed/absolute nuisance delta를 각각 보여 준다.",
        "",
        "![37-grid에서 DINO backbone 비교](figures/02_common37_backbones.png)",
        "",
        "**관찰된 결과.** all-baseline endpoint를 backbone별로 나누면 다음과 같다. "
        "Identity AUROC saturation, translation과 TPS의 두 rank-preservation "
        "endpoint, JPEG absolute nuisance delta를 서로 바꾸어 읽지 않는다.",
        "",
        *backbone_lines,
        "",
        "**해석.** identity AUROC의 포화는 이 corruption에서의 ceiling을 뜻한다. "
        "Translation/TPS Spearman과 pairwise accuracy는 severity 순서 보존을, "
        "JPEG absolute delta는 작을수록 nuisance 불변성이 높음을 뜻한다. "
        "DINOv3-L은 합리적인 선택이지만 모든 endpoint에서 지배적이지 않다. "
        "이 exact-support 실험에서 original DINO가 DINOv2보다 낫다는 순서는 "
        "재현하지 못했다. 이 비교는 네 shoe에 조건부이며 모집단 우월성 검증이 아니다.",
        "",
        f"## 5. {REPORT_SECTION_TITLES[4]}",
        "",
        "**그림 읽는 법.** bilinear 37과 592의 endpoint를 exact five-way support에서, "
        "NAF 차이는 동일 resolution의 paired support에서 읽는다.",
        "",
        "![DINOv3-L 해상도와 NAF 비교](figures/03_dinov3l_resolution_naf.png)",
        "",
        "**관찰된 결과.** L37/L592의 translation Spearman은 "
        f"`{resolution_endpoints[(37, 'translation')]}` / "
        f"`{resolution_endpoints[(592, 'translation')]}`, TPS Spearman은 "
        f"`{resolution_endpoints[(37, 'thin_plate_warp')]}` / "
        f"`{resolution_endpoints[(592, 'thin_plate_warp')]}`, JPEG absolute "
        f"delta는 `{jpeg_by_resolution[37]}` / `{jpeg_by_resolution[592]}`이다. "
        "592 NAF-minus-bilinear는 JPEG absolute delta "
        f"`{naf_minus_bilinear['JPEG absolute delta']}`, TPS Spearman "
        f"`{naf_minus_bilinear['TPS Spearman']}`이다.",
        "",
        "**해석.** 해상도와 NAF의 차이는 endpoint 및 paired support에 의존한다. "
        "따라서 높은 resolution이 항상 더 낫다는 주장은 이 표에서 나오지 않는다.",
        "",
        f"## 6. {REPORT_SECTION_TITLES[5]}",
        "",
        "**그림 읽는 법.** L37-bilinear와 L592-NAF의 absolute response를 동일 "
        "paired support에서 비교한다. depth_scale은 precomputed depth의 전역 scale, "
        "relative_depth_noise는 국소 상대 depth noise, relative_translation은 camera "
        "위치 perturbation, rotation은 camera 각도 perturbation에 대한 국소 민감도다.",
        "",
        "![카메라와 geometry perturbation 민감도](figures/04_geometry_calibration.png)",
        "",
        "**관찰된 결과.** rotation 0.25°/1°/5°에서 L37-bilinear는 "
        f"`{geometry_responses[('rotation', 0.25, 'L37-bilinear')]}` / "
        f"`{geometry_responses[('rotation', 1.0, 'L37-bilinear')]}` / "
        f"`{geometry_responses[('rotation', 5.0, 'L37-bilinear')]}`, "
        "L592-NAF는 "
        f"`{geometry_responses[('rotation', 0.25, 'L592-NAF')]}` / "
        f"`{geometry_responses[('rotation', 1.0, 'L592-NAF')]}` / "
        f"`{geometry_responses[('rotation', 5.0, 'L592-NAF')]}`이다.",
        "",
        "depth_scale 1%/5%에서 L37-bilinear는 "
        f"`{geometry_responses[('depth_scale', 1.01, 'L37-bilinear')]}` / "
        f"`{geometry_responses[('depth_scale', 1.05, 'L37-bilinear')]}`, "
        "L592-NAF는 "
        f"`{geometry_responses[('depth_scale', 1.01, 'L592-NAF')]}` / "
        f"`{geometry_responses[('depth_scale', 1.05, 'L592-NAF')]}`이다.",
        "",
        "relative_depth_noise 0.5%/5%에서 L37-bilinear는 "
        f"`{geometry_responses[('relative_depth_noise', 0.005, 'L37-bilinear')]}` / "
        f"`{geometry_responses[('relative_depth_noise', 0.05, 'L37-bilinear')]}`, "
        "L592-NAF는 "
        f"`{geometry_responses[('relative_depth_noise', 0.005, 'L592-NAF')]}` / "
        f"`{geometry_responses[('relative_depth_noise', 0.05, 'L592-NAF')]}`이다.",
        "",
        "relative_translation 0.25%/2%에서 L37-bilinear는 "
        f"`{geometry_responses[('relative_translation', 0.0025, 'L37-bilinear')]}` / "
        f"`{geometry_responses[('relative_translation', 0.02, 'L37-bilinear')]}`, "
        "L592-NAF는 "
        f"`{geometry_responses[('relative_translation', 0.0025, 'L592-NAF')]}` / "
        f"`{geometry_responses[('relative_translation', 0.02, 'L592-NAF')]}`이다.",
        "",
        depth_boundary_line,
        "",
        "**해석.** injected local perturbation은 precomputed camera 주변의 민감도를 "
        "추정한다. 각 series는 서로 다른 물리 단위의 국소 response envelope이며, "
        "10% depth_scale의 support 소실은 score 0이 아니라 abstention 경계다. "
        "이 실험은 알려지지 않은 절대적인 DA3 카메라 오차를 추정하지 않는다.",
        "",
        f"## 7. {REPORT_SECTION_TITLES[6]}",
        "",
        "**그림 읽는 법.** pair는 두 view 중 더 나쁜 alignment class를 따르며, "
        "pass rate와 finite L37/L592 disagreement를 feature score와 분리해 읽는다.",
        "",
        "![독립 deformation reliability 분석](figures/05_real_deformation_reliability.png)",
        "",
        "**관찰된 결과.** all-baseline L592 symmetric pass rate는 strict "
        f"`{high_pass_strata['strict']}`, moderate "
        f"`{high_pass_strata['moderate']}`, high-misalignment "
        f"`{high_pass_strata['high_misalignment']}`이다. baseline step 1/2/3의 "
        "finite low/high disagreement는 "
        f"`{disagreement_by_step['1']}` / `{disagreement_by_step['2']}` / "
        f"`{disagreement_by_step['3']}`이다. 연속 covariate와의 per-shoe "
        "Spearman은 "
        + ", ".join(
            f"{label} `{value}`"
            for label, value in reliability_correlations.items()
        )
        + "이다.",
        "",
        "**해석.** abstention은 0 similarity가 아니라 reliability boundary다. "
        "strata는 제외 기준이 아니라 관찰된 generation reliability의 기술이다.",
        "",
        f"## 8. {REPORT_SECTION_TITLES[7]}",
        "",
        "### 직접 확인된 사실",
        "",
        "Reliability-stratified pass rate와 finite L37/L592 disagreement는 연속적인 "
        "reliability weighting을 지지한다. 이 결과는 feature fidelity와 geometry/visibility "
        "reliability를 분리해 보고해야 함을 뒷받침한다.",
        "",
        "### 이번 실험에서 확인되지 않은 것",
        "",
        module_design_interpretation_ko,
        "",
        "### 제안하는 모듈 설계",
        "",
        "연속적인 reliability head가 semantic 및 pixel evidence를 가중하도록 설계한다. "
        "학습과 별도의 held-out 검증을 전제로 하며 제약은 다음과 같다:",
        "",
        "0 ≤ pixel_weight ≤ semantic_weight ≤ 1",
        "",
        f"## 9. {REPORT_SECTION_TITLES[8]}",
        "",
        "결과는 adjacent upper view, 고정된 네 shoe 표본, DA3-estimated geometry에 "
        "조건부다. calibrated-camera ground truth, bottom-view behavior, 모집단 수준 "
        "backbone superiority, 확증적 유의성, 이 shoe 밖의 일반화를 확립하지 않는다. "
        "Identity swap은 semantic corruption의 upper-bound이며, local perturbation "
        "envelope은 absolute camera error를 복원하지 않는다.",
        (
            "충족하지 못한 frozen analyzability criterion: "
            + ", ".join(f"`{value}`" for value in incomplete)
            + ". 이 실패는 reliability boundary로 남긴다."
            if incomplete
            else "이 snapshot의 artifact-derived frozen analyzability criteria는 모두 충족했다."
        ),
        "",
        f"## 10. {REPORT_SECTION_TITLES[9]}",
        "",
        "- [PCA audit visualizations](audit_visualizations/pca/)",
        "- [Directional audit visualizations](audit_visualizations/directional/)",
        "- [Audit manifest](audit_visualizations/audit_manifest.json)",
        *input_lines,
        *artifact_lines,
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _publish_reports(root: Path, staging: Path) -> Path:
    destination = root / "reports"
    stale_backups = sorted(root.glob(".reports.backup.*"))
    if destination.exists():
        for stale in stale_backups:
            try:
                shutil.rmtree(stale)
            except OSError:
                pass
    elif len(stale_backups) == 1:
        os.replace(stale_backups[0], destination)
    elif len(stale_backups) > 1:
        raise RuntimeError(
            "ambiguous report recovery: reports is absent and multiple "
            "backups are present"
        )
    backup = root / f".reports.backup.{uuid.uuid4().hex}"
    had_destination = destination.exists()
    try:
        if had_destination:
            os.replace(destination, backup)
        os.replace(staging, destination)
    except Exception:
        if destination.exists() and not had_destination:
            shutil.rmtree(destination)
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            # The second rename is the commit point. Cleanup is recoverable:
            # a later publication retries any retained backup before staging.
            pass
    return destination


def _load_analysis_inputs(
    root: Path,
) -> tuple[
    Path,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    _PreflightEvidence,
]:
    generation = resolve_aggregate_generation(root)
    controlled = pd.read_parquet(
        generation / "controlled_fidelity.parquet"
    )
    geometry = pd.read_parquet(
        generation / "geometry_calibration.parquet"
    )
    stress = pd.read_parquet(
        generation / "real_deformation_stress.parquet"
    )
    preflight_path = root / "reports" / "geometry_preflight.parquet"
    decision_path = root / "reports" / "geometry_preflight.json"
    if not preflight_path.is_file() or not decision_path.is_file():
        raise ValueError("validated adjacent geometry preflight is required")
    parquet_bytes = preflight_path.read_bytes()
    decision_bytes = decision_path.read_bytes()
    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    try:
        decision = json.loads(decision_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("adjacent geometry preflight decision is invalid") from error
    if (
        not isinstance(decision, dict)
        or decision.get("passed") is not True
        or decision.get("output_sha256") != parquet_sha256
    ):
        raise ValueError("adjacent geometry preflight evidence is not validated")
    return (
        generation,
        controlled,
        geometry,
        stress,
        _PreflightEvidence(
            frame=pd.read_parquet(io.BytesIO(parquet_bytes)),
            parquet_bytes=parquet_bytes,
            decision_bytes=decision_bytes,
            parquet_sha256=parquet_sha256,
            decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        ),
    )


def _compute_analysis_tables(
    controlled: pd.DataFrame,
    geometry_raw: pd.DataFrame,
    stress_raw: pd.DataFrame,
    preflight: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    backbone = summarize_backbone_common37(
        controlled,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    resolution = summarize_resolution_bilinear(
        controlled,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    naf = summarize_naf_vs_bilinear(
        controlled,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    geometry = summarize_geometry_low_high(
        geometry_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    stress = summarize_stress_reliability(
        stress_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    support = summarize_support_coverage(
        preflight,
        controlled,
        stress_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    common = summarize_common_support(
        backbone,
        resolution,
        naf,
        geometry,
        stress,
    )
    return {
        "support_coverage": support,
        "backbone_common37": backbone,
        "resolution_bilinear": resolution,
        "naf_vs_bilinear": naf,
        "geometry_low_high": geometry,
        "stress_reliability": stress,
        "common_support": common,
        "analyzability": summarize_analyzability(
            controlled,
            geometry_raw,
            stress_raw,
            common,
        ),
    }


def _render_analysis_snapshot(
    staging: Path,
    *,
    preflight: _PreflightEvidence,
    tables: dict[str, pd.DataFrame],
    controlled_rows: int,
    geometry_rows: int,
    stress_rows: int,
    bootstrap_repeats: int,
    seed: int,
    generation: Path,
) -> None:
    (staging / "geometry_preflight.parquet").write_bytes(
        preflight.parquet_bytes
    )
    (staging / "geometry_preflight.json").write_bytes(
        preflight.decision_bytes
    )
    tables_dir = staging / "tables"
    figures_dir = staging / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(tables_dir, tables)

    import matplotlib.pyplot as plt

    figures = _build_core_figures(tables)
    for filename in FIGURE_FILENAMES:
        figure = figures[filename]
        figure.savefig(
            figures_dir / filename,
            dpi=180,
            metadata={"Software": "met3r.adjacent16_analysis"},
        )
        plt.close(figure)
    artifact_paths = [
        *[
            tables_dir / f"{stem}.{suffix}"
            for stem in TABLE_STEMS
            for suffix in ("csv", "parquet")
        ],
        *[figures_dir / filename for filename in FIGURE_FILENAMES],
        staging / "geometry_preflight.parquet",
        staging / "geometry_preflight.json",
    ]
    _write_report(
        staging / "final_research_report.md",
        tables=tables,
        controlled_rows=controlled_rows,
        geometry_rows=geometry_rows,
        stress_rows=stress_rows,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
        generation=generation,
        artifact_paths=artifact_paths,
    )


def analyze_adjacent16_run(
    run_root: str | Path,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 42,
) -> Adjacent16AnalysisOutputs:
    """Publish the frozen adjacent-view analysis atomically."""

    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    root = Path(run_root).expanduser().resolve()
    (
        generation,
        controlled,
        geometry_raw,
        stress_raw,
        preflight_evidence,
    ) = _load_analysis_inputs(root)
    tables = _compute_analysis_tables(
        controlled,
        geometry_raw,
        stress_raw,
        preflight_evidence.frame,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )

    staging = Path(tempfile.mkdtemp(prefix=".reports.staging.", dir=root))
    try:
        _render_analysis_snapshot(
            staging,
            preflight=preflight_evidence,
            tables=tables,
            controlled_rows=len(controlled),
            geometry_rows=len(geometry_raw),
            stress_rows=len(stress_raw),
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
            generation=generation,
        )
        published = _publish_reports(root, staging)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return Adjacent16AnalysisOutputs(
        tables=tuple(
            published / "tables" / f"{stem}.csv"
            for stem in TABLE_STEMS
        ),
        figures=tuple(
            published / "figures" / filename
            for filename in FIGURE_FILENAMES
        ),
        report_path=published / "final_research_report.md",
    )
