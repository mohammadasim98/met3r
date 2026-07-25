"""Scale-safe descriptive analysis for the upper-view fidelity experiment."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .pilot_gate import resolve_aggregate_generation


CAMERA_LIMITATION = "fixed/precomputed DA3 cameras (not calibrated GT)"
VARIANT_COLUMNS = (
    "backbone",
    "feature_track",
    "feature_mode",
    "resolution",
)
SERIES_COLUMNS = (
    "shoe_id",
    "pair_id",
    "changed_view",
    *VARIANT_COLUMNS,
    "corruption",
)
CONTINUOUS_CORRUPTIONS = {
    "translation",
    "thin_plate_warp",
    "exposure",
    "blur",
    "jpeg",
}
NUISANCE_CORRUPTIONS = {"exposure", "blur", "jpeg"}
CALIBRATION_KINDS = (
    "rotation",
    "relative_translation",
    "depth_scale",
    "relative_depth_noise",
)
TABLE_FILENAMES = (
    "controlled_fidelity.csv",
    "severity_curves.csv",
    "backbone_comparison.csv",
    "resolution_utility.csv",
    "naf_guide_ablation.csv",
    "attribution.csv",
    "geometry_calibration.csv",
    "real_deformation_stress.csv",
    "leave_one_shoe_out.csv",
)
FIGURE_FILENAMES = (
    "resolution_fidelity.png",
    "severity_curves.png",
    "backbone_discrimination.png",
    "naf_guide_ablation.png",
    "attribution.png",
    "geometry_calibration.png",
    "real_deformation_stress.png",
    "coverage.png",
)
_METRIC_COLUMNS = (
    "coverage",
    "abstention_rate",
    "severity_spearman",
    "clean_corrupt_auc",
    "pairwise_rank_accuracy",
    "nuisance_signed_delta",
    "nuisance_absolute_delta",
    "first_to_second_overlap",
    "second_to_first_overlap",
)


@dataclass(frozen=True)
class AnalysisOutputs:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    report_path: Path


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[
        np.isfinite(np.asarray(values, dtype=float))
    ]


def _safe_mean(values: Iterable[object]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_spearman(x: Iterable[object], y: Iterable[object]) -> float:
    first = np.asarray(list(x), dtype=float)
    second = np.asarray(list(y), dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if (
        valid.sum() < 3
        or np.unique(first[valid]).size < 2
        or np.unique(second[valid]).size < 2
    ):
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def _safe_auc(labels: Iterable[object], scores: Iterable[object]) -> float:
    labels_array = np.asarray(list(labels), dtype=bool)
    scores_array = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(scores_array)
    labels_array = labels_array[valid]
    scores_array = scores_array[valid]
    positives = int(labels_array.sum())
    negatives = int((~labels_array).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores_array)
    return float(
        (
            ranks[labels_array].sum()
            - positives * (positives + 1) / 2
        )
        / (positives * negatives)
    )


def _safe_rank_accuracy(severity: Iterable[object], scores: Iterable[object]) -> float:
    severity_array = np.asarray(list(severity), dtype=float)
    score_array = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(severity_array) & np.isfinite(score_array)
    severity_array = severity_array[valid]
    score_array = score_array[valid]
    comparisons: list[float] = []
    for first in range(len(severity_array)):
        for second in range(first + 1, len(severity_array)):
            expected = np.sign(severity_array[first] - severity_array[second])
            if expected == 0:
                continue
            observed = np.sign(score_array[first] - score_array[second])
            comparisons.append(
                1.0 if observed == expected else 0.5 if observed == 0 else 0.0
            )
    return _safe_mean(comparisons)


def robust_clean_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize scores within each exact backbone/track/mode/resolution variant."""

    _require(frame, (*VARIANT_COLUMNS, "corruption", "s00", "s10"), "metric frame")
    normalized = frame.copy()
    normalized["clean_center"] = np.nan
    normalized["clean_scale"] = np.nan
    normalized["normalized_score"] = np.nan
    normalized["normalized_delta"] = np.nan
    normalized["metric_delta"] = (
        pd.to_numeric(normalized["s10"], errors="coerce")
        - pd.to_numeric(normalized["s00"], errors="coerce")
    )
    for _, indexes in normalized.groupby(
        list(VARIANT_COLUMNS), dropna=False, sort=False
    ).groups.items():
        group = normalized.loc[indexes]
        clean = _finite(
            group.loc[group["corruption"] == "clean", "s00"]
        )
        # Controlled rows repeat the same fixed clean baseline in s00. This
        # fallback keeps synthetic/stress-only subsets analyzable.
        if len(clean) == 0:
            clean = _finite(group["s00"])
        if len(clean) == 0:
            center = float("nan")
            scale = 1e-6
        else:
            center = float(np.median(clean))
            scale = float(1.4826 * np.median(np.abs(clean - center)))
            if not np.isfinite(scale) or scale < 1e-6:
                q25, q75 = np.quantile(clean, [0.25, 0.75])
                fallback = float((q75 - q25) / 1.349)
                scale = max(fallback if np.isfinite(fallback) else 0.0, 1e-6)
        normalized.loc[indexes, "clean_center"] = center
        normalized.loc[indexes, "clean_scale"] = scale
        normalized.loc[indexes, "normalized_score"] = (
            pd.to_numeric(group["s10"], errors="coerce") - center
        ) / scale
        normalized.loc[indexes, "normalized_delta"] = (
            pd.to_numeric(group["s10"], errors="coerce")
            - pd.to_numeric(group["s00"], errors="coerce")
        ) / scale
    return normalized


def prepare_primary_metric(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the fixed-geometry primary and adaptive-geometry secondary scores."""

    _require(frame, ("s00", "s10", "s11"), "factorial metric frame")
    prepared = robust_clean_normalize(frame)
    prepared["score"] = pd.to_numeric(prepared["s10"], errors="coerce")
    prepared["adaptive_score"] = pd.to_numeric(prepared["s11"], errors="coerce")
    prepared["adaptive_delta"] = (
        pd.to_numeric(prepared["s11"], errors="coerce")
        - pd.to_numeric(prepared["s00"], errors="coerce")
    )
    if "s01" in prepared:
        prepared["geometry_delta"] = (
            pd.to_numeric(prepared["s01"], errors="coerce")
            - pd.to_numeric(prepared["s00"], errors="coerce")
        )
        prepared["interaction_delta"] = (
            pd.to_numeric(prepared["s11"], errors="coerce")
            - pd.to_numeric(prepared["s10"], errors="coerce")
            - pd.to_numeric(prepared["s01"], errors="coerce")
            + pd.to_numeric(prepared["s00"], errors="coerce")
        )
    return prepared


def _baseline_for_series(
    baselines: Mapping[tuple[object, ...], pd.Series],
    series: pd.DataFrame,
) -> pd.Series:
    first = series.iloc[0].copy()
    key = tuple(first[column] for column in ("shoe_id", "pair_id", *VARIANT_COLUMNS))
    baseline = baselines.get(key, first).copy()
    baseline["changed_view"] = first["changed_view"]
    baseline["corruption"] = first["corruption"]
    baseline["severity"] = 0.0
    baseline["s10"] = baseline["s00"]
    baseline["s11"] = baseline["s00"]
    baseline["score"] = baseline["s00"]
    baseline["adaptive_score"] = baseline["s00"]
    baseline["metric_delta"] = 0.0
    baseline["normalized_delta"] = 0.0
    baseline["adaptive_delta"] = 0.0
    baseline["abstained"] = False
    baseline["_synthetic_baseline"] = True
    for direction in ("first_to_second", "second_to_first"):
        source = f"s00_{direction}_overlap"
        target = f"s10_{direction}_overlap"
        if source in baseline.index:
            baseline[target] = baseline[source]
    return baseline


def _controlled_with_zero(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_primary_metric(frame)
    prepared["_synthetic_baseline"] = False
    nonclean = prepared[prepared["corruption"] != "clean"]
    baseline_columns = ("shoe_id", "pair_id", *VARIANT_COLUMNS)
    baselines = {
        tuple(row[column] for column in baseline_columns): row
        for _, row in prepared[prepared["corruption"] == "clean"]
        .drop_duplicates(list(baseline_columns))
        .iterrows()
    }
    additions = [
        _baseline_for_series(baselines, series)
        for _, series in nonclean.groupby(
            list(SERIES_COLUMNS), dropna=False, sort=False
        )
        if not bool((pd.to_numeric(series["severity"], errors="coerce") == 0).any())
    ]
    if additions:
        nonclean = pd.concat(
            [nonclean, pd.DataFrame(additions)],
            ignore_index=True,
            sort=False,
        )
    return nonclean


def _series_endpoints(series: pd.DataFrame, corruption: str) -> dict[str, float]:
    abstained = (
        series["abstained"].fillna(True).astype(bool)
        if "abstained" in series
        else pd.Series(False, index=series.index)
    )
    valid = (
        ~abstained
        & np.isfinite(pd.to_numeric(series["score"], errors="coerce"))
        & np.isfinite(pd.to_numeric(series["severity"], errors="coerce"))
    )
    rows = series.loc[valid]
    observed = ~series.get(
        "_synthetic_baseline", pd.Series(False, index=series.index)
    ).fillna(False).astype(bool)
    observed_rows = series.loc[valid & observed]
    identity = corruption == "identity_swap"
    continuous = corruption in CONTINUOUS_CORRUPTIONS
    result = {
        "coverage": (
            float(valid[observed].mean()) if bool(observed.any()) else float("nan")
        ),
        "abstention_rate": (
            float(1.0 - valid[observed].mean())
            if bool(observed.any())
            else float("nan")
        ),
        "severity_spearman": (
            _safe_spearman(rows["severity"], rows["score"])
            if continuous
            else float("nan")
        ),
        "clean_corrupt_auc": _safe_auc(
            pd.to_numeric(rows["severity"], errors="coerce") > 0,
            rows["score"],
        ),
        "pairwise_rank_accuracy": (
            _safe_rank_accuracy(rows["severity"], rows["score"])
            if continuous or identity
            else float("nan")
        ),
        "nuisance_signed_delta": (
            _safe_mean(
                rows.loc[
                    pd.to_numeric(rows["severity"], errors="coerce") > 0,
                    "normalized_delta",
                ]
            )
            if corruption in NUISANCE_CORRUPTIONS
            else float("nan")
        ),
        "nuisance_absolute_delta": (
            _safe_mean(
                rows.loc[
                    pd.to_numeric(rows["severity"], errors="coerce") > 0,
                    "normalized_delta",
                ].abs()
            )
            if corruption in NUISANCE_CORRUPTIONS
            else float("nan")
        ),
    }
    for direction in ("first_to_second", "second_to_first"):
        column = f"s10_{direction}_overlap"
        result[f"{direction}_overlap"] = (
            _safe_mean(observed_rows[column])
            if column in observed_rows
            else float("nan")
        )
    return result


def _directional_units(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, series in frame.groupby(list(SERIES_COLUMNS), dropna=False, sort=True):
        values = dict(zip(SERIES_COLUMNS, key))
        records.append(
            {
                **values,
                **_series_endpoints(series, str(values["corruption"])),
            }
        )
    return pd.DataFrame(records)


def _hierarchical_metric_values(
    units: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame(columns=[*group_columns, "shoe_id", *_METRIC_COLUMNS])
    placement = (
        units.groupby(
            [*group_columns, "shoe_id", "pair_id"],
            dropna=False,
            sort=True,
        )[list(_METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    return (
        placement.groupby(
            [*group_columns, "shoe_id"],
            dropna=False,
            sort=True,
        )[list(_METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )


def _bootstrap_intervals(
    shoe_values: pd.DataFrame,
    metrics: Sequence[str],
    *,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if repeats <= 0 or len(shoe_values) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    matrix = shoe_values.loc[:, list(metrics)].to_numpy(float)
    eligible = {
        metric: int(np.isfinite(matrix[:, column]).sum()) >= 2
        for column, metric in enumerate(metrics)
    }
    for _ in range(repeats):
        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))]
        for column, metric in enumerate(metrics):
            if not eligible[metric]:
                continue
            value = _safe_mean(sampled[:, column])
            if np.isfinite(value):
                samples[metric].append(value)
    return {
        metric: tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))
        for metric, values in samples.items()
        if values
    }


def _valid_shoe_count(shoe_values: pd.DataFrame, metric: str) -> int:
    return int(
        np.isfinite(pd.to_numeric(shoe_values[metric], errors="coerce")).sum()
    )


def summarize_controlled_fidelity(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize controlled endpoints with directional, pair, and shoe hierarchy."""

    _require(
        frame,
        (*SERIES_COLUMNS, "severity", "s00", "s10", "s11"),
        "controlled metric frame",
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *VARIANT_COLUMNS,
                "corruption",
                "n_shoes",
                "n_pairs",
                "n_directional_series",
                *_METRIC_COLUMNS,
                *(f"{metric}_n_shoes_valid" for metric in _METRIC_COLUMNS),
                *(
                    f"{metric}_{bound}"
                    for metric in _METRIC_COLUMNS
                    for bound in ("ci_low", "ci_high")
                ),
            ]
        )
    units = _directional_units(_controlled_with_zero(frame))
    groups = [*VARIANT_COLUMNS, "corruption"]
    shoe_values = _hierarchical_metric_values(units, groups)
    records: list[dict[str, object]] = []
    for key, shoes in shoe_values.groupby(groups, dropna=False, sort=True):
        matching = units
        for column, value in zip(groups, key):
            matching = matching[matching[column] == value]
        intervals = _bootstrap_intervals(
            shoes, _METRIC_COLUMNS, repeats=bootstrap_repeats, seed=seed
        )
        record: dict[str, object] = {
            **dict(zip(groups, key)),
            "n_shoes": int(shoes["shoe_id"].nunique()),
            "n_pairs": int(matching[["shoe_id", "pair_id"]].drop_duplicates().shape[0]),
            "n_directional_series": int(len(matching)),
        }
        for metric in _METRIC_COLUMNS:
            record[metric] = _safe_mean(shoes[metric])
            record[f"{metric}_n_shoes_valid"] = _valid_shoe_count(shoes, metric)
            if metric in intervals:
                record[f"{metric}_ci_low"], record[f"{metric}_ci_high"] = intervals[
                    metric
                ]
            else:
                record[f"{metric}_ci_low"] = float("nan")
                record[f"{metric}_ci_high"] = float("nan")
        records.append(record)
    return pd.DataFrame(records)


def _mean_hierarchy(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    placement: bool,
) -> pd.DataFrame:
    keys = [*group_columns, "shoe_id", "pair_id"]
    if placement and "changed_view" in frame:
        keys.append("changed_view")
    directional = (
        frame.groupby(keys, dropna=False, sort=True)[list(value_columns)]
        .mean()
        .reset_index()
    )
    if placement:
        directional = (
            directional.groupby(
                [*group_columns, "shoe_id", "pair_id"],
                dropna=False,
                sort=True,
            )[list(value_columns)]
            .mean()
            .reset_index()
        )
    return (
        directional.groupby(
            [*group_columns, "shoe_id"],
            dropna=False,
            sort=True,
        )[list(value_columns)]
        .mean()
        .reset_index()
    )


def _summarize_means(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    placement: bool,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "n_shoes",
                *value_columns,
                *(f"{value}_n_shoes_valid" for value in value_columns),
                *(
                    f"{value}_{bound}"
                    for value in value_columns
                    for bound in ("ci_low", "ci_high")
                ),
            ]
        )
    shoes = _mean_hierarchy(frame, group_columns, value_columns, placement=placement)
    records: list[dict[str, object]] = []
    iterator = shoes.groupby(list(group_columns), dropna=False, sort=True)
    for key, group in iterator:
        key_tuple = key if isinstance(key, tuple) else (key,)
        intervals = _bootstrap_intervals(
            group, value_columns, repeats=bootstrap_repeats, seed=seed
        )
        record: dict[str, object] = {
            **dict(zip(group_columns, key_tuple)),
            "n_shoes": int(group["shoe_id"].nunique()),
        }
        for value in value_columns:
            record[value] = _safe_mean(group[value])
            record[f"{value}_n_shoes_valid"] = _valid_shoe_count(group, value)
            if value in intervals:
                record[f"{value}_ci_low"], record[f"{value}_ci_high"] = intervals[
                    value
                ]
            else:
                record[f"{value}_ci_low"] = float("nan")
                record[f"{value}_ci_high"] = float("nan")
        records.append(record)
    return pd.DataFrame(records)


def summarize_attribution(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    value_columns = (
        "feature_effect",
        "geometry_effect",
        "interaction",
        "adaptive_total_effect",
        "feature_effect_abs_mean",
        "geometry_effect_abs_mean",
        "absolute_effect_denominator",
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *VARIANT_COLUMNS,
                "corruption",
                "n_shoes",
                *value_columns,
                "absolute_feature_share",
                *(
                    f"{value}_{suffix}"
                    for value in (*value_columns, "absolute_feature_share")
                    for suffix in ("n_shoes_valid", "ci_low", "ci_high")
                ),
            ]
        )
    prepared = prepare_primary_metric(frame)
    valid = prepared[
        ~prepared.get("abstained", pd.Series(False, index=prepared.index))
        .fillna(True)
        .astype(bool)
    ].copy()
    valid["feature_effect"] = valid["metric_delta"]
    valid["geometry_effect"] = valid["geometry_delta"]
    valid["interaction"] = valid["interaction_delta"]
    valid["adaptive_total_effect"] = valid["adaptive_delta"]
    valid["feature_effect_abs_mean"] = valid["feature_effect"].abs()
    valid["geometry_effect_abs_mean"] = valid["geometry_effect"].abs()
    groups = [*VARIANT_COLUMNS, "corruption"]
    shoes = _mean_hierarchy(
        valid,
        groups,
        value_columns[:-1],
        placement=True,
    )
    shoes["absolute_effect_denominator"] = (
        shoes["feature_effect_abs_mean"] + shoes["geometry_effect_abs_mean"]
    )
    records: list[dict[str, object]] = []
    for key, group in shoes.groupby(groups, dropna=False, sort=True):
        intervals = _bootstrap_intervals(
            group,
            value_columns,
            repeats=bootstrap_repeats,
            seed=seed,
        )
        numerator = _safe_mean(group["feature_effect_abs_mean"])
        geometry = _safe_mean(group["geometry_effect_abs_mean"])
        denominator = numerator + geometry
        share = (
            float(numerator / denominator)
            if np.isfinite(denominator) and denominator > 0
            else float("nan")
        )
        share_interval = _bootstrap_ratio_interval(
            group,
            "feature_effect_abs_mean",
            "geometry_effect_abs_mean",
            repeats=bootstrap_repeats,
            seed=seed,
        )
        record: dict[str, object] = {
            **dict(zip(groups, key)),
            "n_shoes": int(group["shoe_id"].nunique()),
        }
        for value in value_columns:
            record[value] = (
                denominator
                if value == "absolute_effect_denominator"
                else _safe_mean(group[value])
            )
            record[f"{value}_n_shoes_valid"] = _valid_shoe_count(group, value)
            if value in intervals:
                record[f"{value}_ci_low"], record[f"{value}_ci_high"] = intervals[
                    value
                ]
            else:
                record[f"{value}_ci_low"] = float("nan")
                record[f"{value}_ci_high"] = float("nan")
        share_valid = (
            np.isfinite(group["feature_effect_abs_mean"])
            & np.isfinite(group["geometry_effect_abs_mean"])
            & (group["absolute_effect_denominator"] > 0)
        )
        record["absolute_feature_share"] = share
        record["absolute_feature_share_n_shoes_valid"] = int(share_valid.sum())
        record["absolute_feature_share_ci_low"] = share_interval[0]
        record["absolute_feature_share_ci_high"] = share_interval[1]
        records.append(record)
    return pd.DataFrame(records)


def _bootstrap_ratio_interval(
    shoe_values: pd.DataFrame,
    numerator: str,
    other: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = shoe_values[[numerator, other]].to_numpy(float)
    valid = (
        np.isfinite(values[:, 0])
        & np.isfinite(values[:, 1])
        & ((values[:, 0] + values[:, 1]) > 0)
    )
    if repeats <= 0 or int(valid.sum()) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(repeats):
        sample = values[rng.integers(0, len(values), size=len(values))]
        finite = (
            np.isfinite(sample[:, 0])
            & np.isfinite(sample[:, 1])
            & ((sample[:, 0] + sample[:, 1]) > 0)
        )
        if not finite.any():
            continue
        feature = float(sample[finite, 0].mean())
        geometry = float(sample[finite, 1].mean())
        samples.append(feature / (feature + geometry))
    if not samples:
        return float("nan"), float("nan")
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def summarize_stress(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize real-deformation effects separately under quality strata."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "stratum",
                *VARIANT_COLUMNS,
                "n_shoes",
                "metric_delta",
                "adaptive_delta",
                "geometry_delta",
                "interaction_delta",
                "coverage",
                "first_to_second_overlap",
                "second_to_first_overlap",
            ]
        )
    working = frame.copy()
    if "corruption" not in working:
        working["corruption"] = "deformed"
    prepared = prepare_primary_metric(working)
    masks: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=prepared.index))
    ]
    for name, field in (
        ("both_camera_preserved", "camera_preserved"),
        ("both_semantic_gate", "semantic_gate"),
        ("both_preservation_gate", "preservation_gate"),
    ):
        first = f"first_{field}"
        second = f"second_{field}"
        if first in prepared and second in prepared:
            masks.append(
                (
                    name,
                    prepared[first].fillna(False).astype(bool)
                    & prepared[second].fillna(False).astype(bool),
                )
            )
    if {"first_fit_relation", "second_fit_relation"} <= set(prepared):
        fit_labels = pd.Series(index=prepared.index, dtype="string")
        for index, row in prepared[
            ["first_fit_relation", "second_fit_relation"]
        ].iterrows():
            first = (
                "unknown"
                if pd.isna(row["first_fit_relation"])
                else str(row["first_fit_relation"])
            )
            second = (
                "unknown"
                if pd.isna(row["second_fit_relation"])
                else str(row["second_fit_relation"])
            )
            fit_labels.loc[index] = (
                f"both_fit_relation:{first}"
                if first == second
                else f"mixed_fit_relation:{first}|{second}"
            )
        for label in sorted(str(value) for value in fit_labels.unique()):
            masks.append((label, fit_labels == label))
    records: list[pd.DataFrame] = []
    for stratum, mask in masks:
        selected = prepared[mask].copy()
        if selected.empty:
            continue
        selected["coverage"] = (
            ~selected.get("abstained", pd.Series(False, index=selected.index))
            .fillna(True)
            .astype(bool)
        ).astype(float)
        invalid = selected["coverage"] == 0
        selected.loc[
            invalid,
            (
                "metric_delta",
                "adaptive_delta",
                "geometry_delta",
                "interaction_delta",
            ),
        ] = np.nan
        for direction in ("first_to_second", "second_to_first"):
            column = f"s10_{direction}_overlap"
            selected[f"{direction}_overlap"] = (
                pd.to_numeric(selected[column], errors="coerce")
                if column in selected
                else np.nan
            )
        summary = _summarize_means(
            selected,
            list(VARIANT_COLUMNS),
            (
                "metric_delta",
                "adaptive_delta",
                "geometry_delta",
                "interaction_delta",
                "coverage",
                "first_to_second_overlap",
                "second_to_first_overlap",
            ),
            placement=False,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
        )
        summary.insert(0, "stratum", stratum)
        records.append(summary)
    return (
        pd.concat(records, ignore_index=True, sort=False)
        if records
        else pd.DataFrame(columns=["stratum", *VARIANT_COLUMNS])
    )


def _summarize_shoe_differences(
    shoe_values: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    if shoe_values.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "n_shoes",
                *value_columns,
                *(f"{value}_n_shoes_valid" for value in value_columns),
                *(
                    f"{value}_{bound}"
                    for value in value_columns
                    for bound in ("ci_low", "ci_high")
                ),
            ]
        )
    records: list[dict[str, object]] = []
    for key, group in shoe_values.groupby(
        list(group_columns), dropna=False, sort=True
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        intervals = _bootstrap_intervals(
            group,
            value_columns,
            repeats=bootstrap_repeats,
            seed=seed,
        )
        record: dict[str, object] = {
            **dict(zip(group_columns, key_tuple)),
            "n_shoes": int(group["shoe_id"].nunique()),
        }
        for value in value_columns:
            record[value] = _safe_mean(group[value])
            record[f"{value}_n_shoes_valid"] = _valid_shoe_count(group, value)
            if value in intervals:
                record[f"{value}_ci_low"], record[f"{value}_ci_high"] = intervals[
                    value
                ]
            else:
                record[f"{value}_ci_low"] = float("nan")
                record[f"{value}_ci_high"] = float("nan")
        records.append(record)
    return pd.DataFrame(records)


def _controlled_shoe_values(frame: pd.DataFrame) -> pd.DataFrame:
    units = _directional_units(_controlled_with_zero(frame))
    return _hierarchical_metric_values(
        units,
        [*VARIANT_COLUMNS, "corruption"],
    )


def _resolution_utility_from_raw(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    shoes = _controlled_shoe_values(frame)
    endpoints = list(_METRIC_COLUMNS)
    keys = ["backbone", "feature_track", "corruption", "shoe_id"]
    native = shoes[shoes["feature_mode"] == "native"][
        [*keys, "resolution", *endpoints]
    ].rename(
        columns={
            "resolution": "native_resolution",
            **{column: f"native_{column}" for column in endpoints},
        }
    )
    compared = shoes.merge(native, on=keys, how="left", validate="many_to_one")
    differences: list[str] = []
    for endpoint in endpoints:
        column = f"{endpoint}_difference_from_native"
        compared[column] = compared[endpoint] - compared[f"native_{endpoint}"]
        differences.append(column)
    groups = [
        *VARIANT_COLUMNS,
        "corruption",
        "native_resolution",
    ]
    return _summarize_shoe_differences(
        compared,
        groups,
        differences,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )


def _naf_ablation_from_raw(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    shoes = _controlled_shoe_values(frame)
    keys = [
        "backbone",
        "feature_track",
        "resolution",
        "corruption",
        "shoe_id",
    ]
    endpoints = list(_METRIC_COLUMNS)
    normal = shoes[shoes["feature_mode"] == "naf_normal_guide"][
        [*keys, *endpoints]
    ]
    records: list[pd.DataFrame] = []
    difference_columns = [
        f"{endpoint}_normal_minus_comparator" for endpoint in endpoints
    ]
    for comparator in (
        "bilinear",
        "naf_blurred_guide",
        "naf_shuffled_guide",
    ):
        other = shoes[shoes["feature_mode"] == comparator][[*keys, *endpoints]]
        merged = normal.merge(
            other,
            on=keys,
            suffixes=("_normal", "_comparator"),
            how="inner",
            validate="one_to_one",
        )
        merged["comparator"] = comparator
        for endpoint, difference in zip(endpoints, difference_columns):
            merged[difference] = (
                merged[f"{endpoint}_normal"] - merged[f"{endpoint}_comparator"]
            )
        records.append(merged)
    paired = (
        pd.concat(records, ignore_index=True, sort=False)
        if records
        else pd.DataFrame(columns=[*keys, "comparator", *difference_columns])
    )
    return _summarize_shoe_differences(
        paired,
        ["backbone", "feature_track", "resolution", "corruption", "comparator"],
        difference_columns,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )


def _geometry_envelope(
    controlled_raw: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    required = (
        *VARIANT_COLUMNS,
        "shoe_id",
        "pair_id",
        "changed_view",
        "perturbation_kind",
        "magnitude",
        "geometry_effect",
    )
    observed = prepare_primary_metric(controlled_raw)
    observed = observed[observed["corruption"] != "clean"].copy()
    if "abstained" in observed:
        observed = observed[~observed["abstained"].fillna(True).astype(bool)]
    observed["absolute_feature_effect"] = observed["metric_delta"].abs()
    if calibration.empty:
        calibration_valid = pd.DataFrame(columns=required)
    else:
        _require(calibration, required, "geometry calibration frame")
        calibration_valid = calibration.copy()
        if "abstained" in calibration_valid:
            calibration_valid = calibration_valid[
                ~calibration_valid["abstained"].fillna(True).astype(bool)
            ]
    records: list[dict[str, object]] = []
    match_columns = [
        *VARIANT_COLUMNS,
        "shoe_id",
        "pair_id",
        "changed_view",
    ]
    curves: dict[tuple[object, ...], pd.DataFrame] = {}
    for key, curve in calibration_valid.groupby(
        [*match_columns, "perturbation_kind"],
        dropna=False,
        sort=False,
    ):
        curves[key] = curve[
            np.isfinite(pd.to_numeric(curve["magnitude"], errors="coerce"))
            & np.isfinite(pd.to_numeric(curve["geometry_effect"], errors="coerce"))
        ].sort_values("magnitude")
    for _, row in observed.iterrows():
        match = tuple(row[column] for column in match_columns)
        for kind in CALIBRATION_KINDS:
            curve = curves.get((*match, kind), pd.DataFrame())
            raw_magnitude, effective_magnitude = _interpolated_crossing(
                curve,
                kind,
                float(row["absolute_feature_effect"]),
            )
            covered = np.isfinite(raw_magnitude)
            raw_unit, effective_unit = _calibration_units(kind)
            records.append(
                {
                    **{column: row[column] for column in VARIANT_COLUMNS},
                    "shoe_id": row["shoe_id"],
                    "pair_id": row["pair_id"],
                    "changed_view": row["changed_view"],
                    "corruption": row["corruption"],
                    "perturbation_kind": kind,
                    "matched_magnitude": raw_magnitude,
                    "matched_magnitude_raw": raw_magnitude,
                    "matched_magnitude_effective": effective_magnitude,
                    "raw_magnitude_unit": raw_unit,
                    "effective_magnitude_unit": effective_unit,
                    "envelope_coverage": float(covered),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                *VARIANT_COLUMNS,
                "corruption",
                "perturbation_kind",
                "matched_magnitude",
                "matched_magnitude_raw",
                "matched_magnitude_effective",
                "envelope_coverage",
            ]
        )
    record_frame = pd.DataFrame(records)
    group_columns = [*VARIANT_COLUMNS, "corruption", "perturbation_kind"]
    summary = _summarize_means(
        record_frame,
        group_columns,
        (
            "matched_magnitude",
            "matched_magnitude_raw",
            "matched_magnitude_effective",
            "envelope_coverage",
        ),
        placement=True,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    counts = (
        record_frame.groupby(group_columns, dropna=False, sort=True)
        .agg(
            n_feature_cells_total=("envelope_coverage", "size"),
            n_feature_cells_covered=("envelope_coverage", "sum"),
        )
        .reset_index()
    )
    counts["n_feature_cells_covered"] = counts[
        "n_feature_cells_covered"
    ].astype(int)
    counts["n_feature_cells_censored"] = (
        counts["n_feature_cells_total"] - counts["n_feature_cells_covered"]
    )
    units = record_frame[
        [
            "perturbation_kind",
            "raw_magnitude_unit",
            "effective_magnitude_unit",
        ]
    ].drop_duplicates()
    return summary.merge(counts, on=group_columns, validate="one_to_one").merge(
        units,
        on="perturbation_kind",
        validate="many_to_one",
    )


def _calibration_units(kind: str) -> tuple[str, str]:
    if kind == "rotation":
        return "degrees", "degrees_from_neutral"
    if kind == "relative_translation":
        return "camera_radius_fraction", "absolute_fraction_from_neutral"
    if kind == "depth_scale":
        return "scale_factor", "absolute_fraction_from_1"
    if kind == "relative_depth_noise":
        return "relative_depth_std", "relative_depth_std_from_neutral"
    raise ValueError(f"unknown calibration kind: {kind}")


def _interpolated_crossing(
    curve: pd.DataFrame,
    kind: str,
    target_effect: float,
) -> tuple[float, float]:
    if not np.isfinite(target_effect) or target_effect < 0:
        return float("nan"), float("nan")
    neutral_raw = 1.0 if kind == "depth_scale" else 0.0
    raw = (
        pd.to_numeric(curve["magnitude"], errors="coerce").to_numpy(float)
        if not curve.empty
        else np.asarray([], dtype=float)
    )
    effects = (
        pd.to_numeric(curve["geometry_effect"], errors="coerce")
        .abs()
        .to_numpy(float)
        if not curve.empty
        else np.asarray([], dtype=float)
    )
    valid = np.isfinite(raw) & np.isfinite(effects)
    raw = raw[valid]
    effects = effects[valid]
    effective = np.abs(raw - neutral_raw)
    raw = np.concatenate(([neutral_raw], raw))
    effective = np.concatenate(([0.0], effective))
    effects = np.concatenate(([0.0], effects))
    order = np.argsort(effective, kind="mergesort")
    raw = raw[order]
    effective = effective[order]
    effects = effects[order]
    envelope = np.maximum.accumulate(effects)
    crossing = np.flatnonzero(envelope >= target_effect)
    if not len(crossing):
        return float("nan"), float("nan")
    upper = int(crossing[0])
    if upper == 0:
        return float(raw[0]), float(effective[0])
    lower = upper - 1
    effect_span = envelope[upper] - envelope[lower]
    fraction = (
        1.0
        if effect_span <= 0
        else float((target_effect - envelope[lower]) / effect_span)
    )
    raw_match = raw[lower] + fraction * (raw[upper] - raw[lower])
    effective_match = effective[lower] + fraction * (
        effective[upper] - effective[lower]
    )
    return float(raw_match), float(effective_match)


def _leave_one_shoe_out(
    controlled_raw: pd.DataFrame,
    calibration_raw: pd.DataFrame,
    stress_raw: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    shoes = sorted(
        set(controlled_raw.get("shoe_id", pd.Series(dtype="object")).dropna())
        | set(stress_raw.get("shoe_id", pd.Series(dtype="object")).dropna())
    )
    for omitted in shoes:
        controlled = controlled_raw[controlled_raw["shoe_id"] != omitted]
        if not controlled.empty:
            families = {
                "controlled_fidelity": summarize_controlled_fidelity(
                    controlled, bootstrap_repeats=0
                ),
                "resolution_utility": _resolution_utility_from_raw(
                    controlled, bootstrap_repeats=0, seed=42
                ),
                "naf_guide_ablation": _naf_ablation_from_raw(
                    controlled, bootstrap_repeats=0, seed=42
                ),
                "attribution": summarize_attribution(
                    controlled, bootstrap_repeats=0
                ),
                "geometry_calibration": _geometry_envelope(
                    controlled,
                    (
                        calibration_raw[calibration_raw["shoe_id"] != omitted]
                        if "shoe_id" in calibration_raw
                        else calibration_raw
                    ),
                    bootstrap_repeats=0,
                    seed=42,
                ),
            }
            for family, summary in families.items():
                for row in summary.to_dict("records"):
                    records.append(
                        {
                            "analysis_track": family,
                            "omitted_shoe": omitted,
                            **row,
                        }
                    )
        stress = stress_raw[stress_raw["shoe_id"] != omitted]
        if not stress.empty:
            summary = summarize_stress(stress, bootstrap_repeats=0)
            for row in summary.to_dict("records"):
                records.append(
                    {
                        "analysis_track": "real_deformation_stress",
                        "omitted_shoe": omitted,
                        **row,
                    }
                )
    return pd.DataFrame(records)


def _backbone_comparison(controlled: pd.DataFrame) -> pd.DataFrame:
    # These are rank/normalized endpoint comparisons, never raw cross-backbone
    # score magnitudes. Keep each endpoint's validity count and interval beside
    # the point estimate.
    return controlled.copy()


def _plot(
    frame: pd.DataFrame,
    destination: Path,
    *,
    x: str | None,
    y: str | None,
    group: str | Sequence[str] | None,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    if (
        not frame.empty
        and y is not None
        and y in frame
        and np.isfinite(pd.to_numeric(frame[y], errors="coerce")).any()
    ):
        plotted = frame.copy()
        if x is None or x not in plotted:
            plotted["_x"] = np.arange(len(plotted))
            x = "_x"
        raw_x = plotted[x]
        numeric_x = pd.to_numeric(raw_x, errors="coerce")
        categorical = bool(raw_x.notna().any() and numeric_x.isna().any())
        category_labels: list[str] = []
        if categorical:
            category_labels = sorted(str(value) for value in raw_x.dropna().unique())
            category_positions = {
                label: index for index, label in enumerate(category_labels)
            }
            plotted["_plot_x"] = raw_x.astype(str).map(category_positions)
        else:
            plotted["_plot_x"] = numeric_x
        group_columns = (
            (group,)
            if isinstance(group, str)
            else tuple(group)
            if group
            else ()
        )
        iterator = _semantic_plot_groups(
            plotted,
            x=x,
            group_columns=group_columns,
        )
        line_count = 0
        for label, rows in iterator:
            rows = rows.sort_values("_plot_x")
            axis.plot(
                rows["_plot_x"],
                pd.to_numeric(rows[y], errors="coerce"),
                marker="o",
                label=str(label),
                alpha=0.8,
            )
            line_count += 1
        if categorical:
            axis.set_xticks(range(len(category_labels)), category_labels, rotation=20)
        if group_columns and line_count <= 20:
            axis.legend(fontsize=7)
        elif line_count > 20:
            axis.text(
                0.99,
                0.01,
                f"{line_count} separate semantic groups",
                ha="right",
                va="bottom",
                fontsize=7,
                transform=axis.transAxes,
            )
    else:
        axis.text(
            0.5,
            0.5,
            "No estimable observations",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title(title)
    if x:
        axis.set_xlabel(x)
    if y:
        axis.set_ylabel(y)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _semantic_plot_groups(
    frame: pd.DataFrame,
    *,
    x: str,
    group_columns: Sequence[str],
) -> tuple[tuple[str, pd.DataFrame], ...]:
    columns = [
        column
        for column in group_columns
        if column != x and column in frame.columns
    ]
    if not columns:
        return (("estimate", frame),)
    groups: list[tuple[str, pd.DataFrame]] = []
    for key, rows in frame.groupby(columns, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        label = " / ".join(
            f"{column}={value}" for column, value in zip(columns, values)
        )
        groups.append((label, rows))
    return tuple(groups)


def _severity_curve_rows(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _controlled_with_zero(frame)
    valid = prepared[
        ~prepared.get("abstained", pd.Series(False, index=prepared.index))
        .fillna(True)
        .astype(bool)
    ]
    groups = [*VARIANT_COLUMNS, "corruption", "severity"]
    placement = (
        valid.groupby(
            [*groups, "shoe_id", "pair_id", "changed_view"],
            dropna=False,
            sort=True,
        )["normalized_delta"]
        .mean()
        .reset_index()
    )
    pairs = (
        placement.groupby(
            [*groups, "shoe_id", "pair_id"],
            dropna=False,
            sort=True,
        )["normalized_delta"]
        .mean()
        .reset_index()
    )
    shoes = (
        pairs.groupby(
            [*groups, "shoe_id"],
            dropna=False,
            sort=True,
        )["normalized_delta"]
        .mean()
        .reset_index()
    )
    summary = (
        shoes.groupby(groups, dropna=False, sort=True)
        .agg(
            normalized_delta=("normalized_delta", "mean"),
            normalized_delta_n_shoes_valid=("normalized_delta", "count"),
        )
        .reset_index()
    )
    summary["n_shoes"] = summary["normalized_delta_n_shoes_valid"]
    return summary


def _write_tables(tables_dir: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for filename in TABLE_FILENAMES:
        frame = frames[filename]
        frame.to_csv(tables_dir / filename, index=False)
        frame.to_parquet(tables_dir / f"{Path(filename).stem}.parquet", index=False)


def _write_report(
    destination: Path,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    controlled = tables["controlled_fidelity.csv"]
    stress = tables["real_deformation_stress.csv"]
    destination.write_text(
        "\n".join(
            [
                f"# {CAMERA_LIMITATION}",
                "",
                "This experiment used upper views only and four shoes. The "
                "shoe-cluster bootstrap intervals and leave-one-shoe-out "
                "estimates are descriptive intervals; they are not "
                "confirmatory inference.",
                "",
                "## Controlled fidelity track",
                "",
                "The primary fixed-geometry endpoint is `s10 - s00`, scored "
                "with `s10`. Identity swap uses clean/corrupt AUROC and rank "
                "accuracy without a severity Spearman endpoint. Continuous "
                "corruptions use within-pair severity ranks. Exposure, blur, "
                "and JPEG retain signed and absolute clean-normalized effects.",
                "Severity-curve points first average changed-view placements "
                "within pair, then pairs within shoe, then shoes equally. "
                "Every plotted line keeps the complete compatible semantic "
                "variant identity.",
                f"This track produced {len(controlled)} endpoint rows.",
                "",
                "## Geometry calibration track",
                "",
                "Adaptive-depth `s11` appears only as a secondary endpoint. "
                "Feature, adaptive-depth geometry, interaction, and the "
                "known pose/depth perturbation envelope are reported "
                "separately. Geometry-envelope crossings are interpolated on "
                "a monotone absolute-effect envelope; missing, abstained, or "
                "out-of-range curves remain explicitly censored.",
                "Absolute feature share is the ratio of separately "
                "hierarchically weighted absolute feature and geometry "
                "effects, not an average of rowwise ratios.",
                "",
                "## Real-deformation stress track",
                "",
                "The real deformations are a separate stress test with all, "
                "camera-preservation, semantic-gate, preservation-gate, and "
                "mutually exclusive joint two-view fit-relation strata.",
                f"This track produced {len(stress)} stratum rows.",
                "",
                "## Scope",
                "",
                "Results do not establish calibrated-camera ground truth, "
                "bottom-view behavior, cross-object generalization beyond "
                "four shoes, or inferential significance.",
                "Every endpoint reports its own finite-shoe count. A "
                "descriptive interval is emitted only with at least two "
                "distinct finite shoes.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _publish_reports(root: Path, staging: Path) -> Path:
    destination = root / "reports"
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
        shutil.rmtree(backup)
    return destination


def analyze_upper_run(
    run_root: str | Path,
    *,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
) -> AnalysisOutputs:
    """Analyze one immutable aggregate snapshot and atomically publish reports."""

    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    root = Path(run_root).expanduser().resolve()
    generation = resolve_aggregate_generation(root)
    controlled_raw = pd.read_parquet(generation / "controlled_fidelity.parquet")
    calibration_raw = pd.read_parquet(generation / "geometry_calibration.parquet")
    stress_raw = pd.read_parquet(generation / "real_deformation_stress.parquet")

    controlled = summarize_controlled_fidelity(
        controlled_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    attribution = summarize_attribution(
        controlled_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    stress = summarize_stress(
        stress_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    geometry = _geometry_envelope(
        controlled_raw,
        calibration_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    resolution = _resolution_utility_from_raw(
        controlled_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    naf = _naf_ablation_from_raw(
        controlled_raw,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    severity = _severity_curve_rows(controlled_raw)
    loso = _leave_one_shoe_out(controlled_raw, calibration_raw, stress_raw)
    frames = {
        "controlled_fidelity.csv": controlled,
        "severity_curves.csv": severity,
        "backbone_comparison.csv": _backbone_comparison(controlled),
        "resolution_utility.csv": resolution,
        "naf_guide_ablation.csv": naf,
        "attribution.csv": attribution,
        "geometry_calibration.csv": geometry,
        "real_deformation_stress.csv": stress,
        "leave_one_shoe_out.csv": loso,
    }

    staging = Path(tempfile.mkdtemp(prefix=".reports.staging.", dir=root))
    try:
        existing = root / "reports"
        if existing.exists():
            shutil.copytree(existing, staging, dirs_exist_ok=True)
        tables_dir = staging / "tables"
        figures_dir = staging / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        _write_tables(tables_dir, frames)
        plot_specs: tuple[
            tuple[
                str,
                pd.DataFrame,
                str | None,
                str | None,
                tuple[str, ...] | None,
                str,
            ],
            ...,
        ] = (
            (
                "resolution_fidelity.png",
                resolution,
                "resolution",
                "severity_spearman_difference_from_native",
                (
                    "backbone",
                    "feature_track",
                    "feature_mode",
                    "corruption",
                    "native_resolution",
                ),
                "Resolution utility",
            ),
            (
                "severity_curves.png",
                severity,
                "severity",
                "normalized_delta",
                (*VARIANT_COLUMNS, "corruption"),
                "Controlled severity curves",
            ),
            (
                "backbone_discrimination.png",
                controlled,
                "backbone",
                "clean_corrupt_auc",
                ("feature_track", "feature_mode", "resolution", "corruption"),
                "Backbone discrimination",
            ),
            (
                "naf_guide_ablation.png",
                naf,
                "comparator",
                "pairwise_rank_accuracy_normal_minus_comparator",
                ("backbone", "feature_track", "resolution", "corruption"),
                "NAF guide ablation",
            ),
            (
                "attribution.png",
                attribution,
                "resolution",
                "absolute_feature_share",
                ("backbone", "feature_track", "feature_mode", "corruption"),
                "Feature and geometry attribution",
            ),
            (
                "geometry_calibration.png",
                geometry,
                "resolution",
                "matched_magnitude_effective",
                (
                    "backbone",
                    "feature_track",
                    "feature_mode",
                    "corruption",
                    "perturbation_kind",
                ),
                "Geometry perturbation envelope",
            ),
            (
                "real_deformation_stress.png",
                stress,
                "resolution",
                "metric_delta",
                ("backbone", "feature_track", "feature_mode", "stratum"),
                "Real-deformation stress",
            ),
            (
                "coverage.png",
                controlled,
                "resolution",
                "coverage",
                ("backbone", "feature_track", "feature_mode", "corruption"),
                "Directional coverage and abstention",
            ),
        )
        for filename, frame, x, y, group, title in plot_specs:
            _plot(
                frame,
                figures_dir / filename,
                x=x,
                y=y,
                group=group,
                title=title,
            )
        _write_report(staging / "final_report.md", frames)
        expected = [
            *(tables_dir / name for name in TABLE_FILENAMES),
            *(figures_dir / name for name in FIGURE_FILENAMES),
            staging / "final_report.md",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
            raise OSError("analysis staging tree is incomplete")
        reports = _publish_reports(root, staging)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return AnalysisOutputs(
        tables=tuple(reports / "tables" / name for name in TABLE_FILENAMES),
        figures=tuple(reports / "figures" / name for name in FIGURE_FILENAMES),
        report_path=reports / "final_report.md",
    )
