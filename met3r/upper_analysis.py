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
TABLE_FILENAMES = (
    "controlled_fidelity.csv",
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
    for _ in range(repeats):
        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))]
        for column, metric in enumerate(metrics):
            value = _safe_mean(sampled[:, column])
            if np.isfinite(value):
                samples[metric].append(value)
    return {
        metric: tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))
        for metric, values in samples.items()
        if values
    }


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
    if frame.empty:
        return _summarize_means(
            pd.DataFrame(),
            [*VARIANT_COLUMNS, "corruption"],
            (
                "feature_effect",
                "geometry_effect",
                "interaction",
                "adaptive_total_effect",
                "absolute_feature_share",
            ),
            placement=True,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
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
    denominator = valid["feature_effect"].abs() + valid["geometry_effect"].abs()
    valid["absolute_feature_share"] = np.divide(
        valid["feature_effect"].abs(),
        denominator,
        out=np.full(len(valid), np.nan),
        where=np.asarray(denominator > 0),
    )
    return _summarize_means(
        valid,
        [*VARIANT_COLUMNS, "corruption"],
        (
            "feature_effect",
            "geometry_effect",
            "interaction",
            "adaptive_total_effect",
            "absolute_feature_share",
        ),
        placement=True,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )


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
    relations: set[str] = set()
    for column in ("first_fit_relation", "second_fit_relation"):
        if column in prepared:
            relations.update(
                str(value)
                for value in prepared[column].dropna().unique()
                if str(value)
            )
    for relation in sorted(relations):
        relation_mask = pd.Series(False, index=prepared.index)
        for column in ("first_fit_relation", "second_fit_relation"):
            if column in prepared:
                relation_mask |= prepared[column].astype("string") == relation
        masks.append((f"fit_relation:{relation}", relation_mask))
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
    endpoints = [column for column in _METRIC_COLUMNS if column != "coverage"]
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
    endpoints = [
        "severity_spearman",
        "clean_corrupt_auc",
        "pairwise_rank_accuracy",
        "nuisance_absolute_delta",
    ]
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
    if calibration.empty:
        return pd.DataFrame(
            columns=[
                *VARIANT_COLUMNS,
                "corruption",
                "perturbation_kind",
                "matched_magnitude",
                "envelope_coverage",
            ]
        )
    _require(calibration, required, "geometry calibration frame")
    observed = prepare_primary_metric(controlled_raw)
    observed = observed[observed["corruption"] != "clean"].copy()
    if "abstained" in observed:
        observed = observed[~observed["abstained"].fillna(True).astype(bool)]
    observed["absolute_feature_effect"] = observed["metric_delta"].abs()
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
                & np.isfinite(
                    pd.to_numeric(curve["geometry_effect"], errors="coerce")
                )
            ].sort_values("magnitude")
    kinds_by_match: dict[tuple[object, ...], list[object]] = {}
    for key in curves:
        kinds_by_match.setdefault(key[:-1], []).append(key[-1])
    for _, row in observed.iterrows():
        match = tuple(row[column] for column in match_columns)
        for kind in kinds_by_match.get(match, ()):
            curve = curves[(*match, kind)]
            reached = curve[
                curve["geometry_effect"].abs() >= row["absolute_feature_effect"]
            ]
            records.append(
                {
                    **{column: row[column] for column in VARIANT_COLUMNS},
                    "shoe_id": row["shoe_id"],
                    "pair_id": row["pair_id"],
                    "changed_view": row["changed_view"],
                    "corruption": row["corruption"],
                    "perturbation_kind": kind,
                    "matched_magnitude": (
                        float(reached.iloc[0]["magnitude"])
                        if len(reached)
                        else float("nan")
                    ),
                    "envelope_coverage": float(bool(len(reached))),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                *VARIANT_COLUMNS,
                "corruption",
                "perturbation_kind",
                "matched_magnitude",
                "envelope_coverage",
            ]
        )
    return _summarize_means(
        pd.DataFrame(records),
        [*VARIANT_COLUMNS, "corruption", "perturbation_kind"],
        ("matched_magnitude", "envelope_coverage"),
        placement=True,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )


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
                    calibration_raw[calibration_raw["shoe_id"] != omitted],
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
    columns = [
        *VARIANT_COLUMNS,
        "corruption",
        "n_shoes",
        "severity_spearman",
        "clean_corrupt_auc",
        "pairwise_rank_accuracy",
        "nuisance_signed_delta",
        "nuisance_absolute_delta",
    ]
    return controlled[[column for column in columns if column in controlled]].copy()


def _plot(
    frame: pd.DataFrame,
    destination: Path,
    *,
    x: str | None,
    y: str | None,
    group: str | None,
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
        iterator = (
            plotted.groupby(group, dropna=False, sort=True)
            if group and group in plotted
            else [("estimate", plotted)]
        )
        for label, rows in iterator:
            rows = rows.sort_values(x)
            axis.plot(
                pd.to_numeric(rows[x], errors="coerce"),
                pd.to_numeric(rows[y], errors="coerce"),
                marker="o",
                label=str(label),
            )
        if group and group in plotted:
            axis.legend(fontsize=7)
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


def _severity_curve_rows(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _controlled_with_zero(frame)
    valid = prepared[
        ~prepared.get("abstained", pd.Series(False, index=prepared.index))
        .fillna(True)
        .astype(bool)
    ]
    return (
        valid.groupby(
            [*VARIANT_COLUMNS, "corruption", "severity"],
            dropna=False,
            sort=True,
        )["normalized_delta"]
        .mean()
        .reset_index()
    )


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
                f"This track produced {len(controlled)} endpoint rows.",
                "",
                "## Geometry calibration track",
                "",
                "Adaptive-depth `s11` appears only as a secondary endpoint. "
                "Feature, adaptive-depth geometry, interaction, and the "
                "known pose/depth perturbation envelope are reported "
                "separately.",
                "",
                "## Real-deformation stress track",
                "",
                "The real deformations are a separate stress test with all, "
                "camera-preservation, semantic-gate, preservation-gate, and "
                "fit-relation strata.",
                f"This track produced {len(stress)} stratum rows.",
                "",
                "## Scope",
                "",
                "Results do not establish calibrated-camera ground truth, "
                "bottom-view behavior, cross-object generalization beyond "
                "four shoes, or inferential significance.",
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
    loso = _leave_one_shoe_out(controlled_raw, calibration_raw, stress_raw)
    frames = {
        "controlled_fidelity.csv": controlled,
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
        severity = _severity_curve_rows(controlled_raw)
        plot_specs: tuple[
            tuple[str, pd.DataFrame, str | None, str | None, str | None, str],
            ...,
        ] = (
            (
                "resolution_fidelity.png",
                resolution,
                "resolution",
                "severity_spearman_difference_from_native",
                "backbone",
                "Resolution utility",
            ),
            (
                "severity_curves.png",
                severity,
                "severity",
                "normalized_delta",
                "corruption",
                "Controlled severity curves",
            ),
            (
                "backbone_discrimination.png",
                controlled,
                "resolution",
                "clean_corrupt_auc",
                "backbone",
                "Backbone discrimination",
            ),
            (
                "naf_guide_ablation.png",
                naf,
                "resolution",
                "pairwise_rank_accuracy_normal_minus_comparator",
                "comparator",
                "NAF guide ablation",
            ),
            (
                "attribution.png",
                attribution,
                "resolution",
                "absolute_feature_share",
                "backbone",
                "Feature and geometry attribution",
            ),
            (
                "geometry_calibration.png",
                geometry,
                "matched_magnitude",
                "envelope_coverage",
                "perturbation_kind",
                "Geometry perturbation envelope",
            ),
            (
                "real_deformation_stress.png",
                stress,
                "resolution",
                "metric_delta",
                "stratum",
                "Real-deformation stress",
            ),
            (
                "coverage.png",
                controlled,
                "resolution",
                "coverage",
                "backbone",
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
