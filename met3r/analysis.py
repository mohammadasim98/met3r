"""Statistical summaries and figures for feature-fidelity experiments."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


DEFAULT_GROUP_COLUMNS = (
    "backbone",
    "resolution",
    "corruption",
    "pair_scope",
    "camera_group",
    "analysis_tier",
)

HUMAN_RATING_COLUMNS = (
    "geometric_alignment",
    "semantic_identity",
    "local_detail_alignment",
    "single_view_plausibility",
)


def _available_groups(frame: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    return [column for column in requested if column in frame.columns]


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.unique(first[valid]).size < 2 or np.unique(second[valid]).size < 2:
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _ranking_accuracy(rows: pd.DataFrame) -> float:
    severity = rows["severity"].to_numpy(float)
    scores = rows["score"].to_numpy(float)
    comparisons = []
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            severity_direction = np.sign(severity[first] - severity[second])
            if severity_direction == 0:
                continue
            score_direction = np.sign(scores[first] - scores[second])
            comparisons.append(
                1.0
                if score_direction == severity_direction
                else 0.5
                if score_direction == 0
                else 0.0
            )
    return float(np.mean(comparisons)) if comparisons else float("nan")


def _equal_shoe_pair_mean(
    rows: pd.DataFrame,
    function: Callable[[pd.DataFrame], float],
) -> float:
    unit_values = []
    for (shoe_id, _), unit in rows.groupby(["shoe_id", "pair_id"], dropna=False):
        value = float(function(unit))
        if np.isfinite(value):
            unit_values.append((shoe_id, value))
    if not unit_values:
        return float("nan")
    unit_frame = pd.DataFrame(unit_values, columns=["shoe_id", "value"])
    return float(unit_frame.groupby("shoe_id")["value"].mean().mean())


def _compute_endpoints(rows: pd.DataFrame) -> dict[str, float]:
    valid = rows[rows["_valid"]]
    endpoints = {
        "coverage": _equal_shoe_pair_mean(rows, lambda unit: unit["_valid"].mean()),
        "severity_spearman": _equal_shoe_pair_mean(
            valid,
            lambda unit: _safe_spearman(
                unit["score"].to_numpy(float),
                unit["severity"].to_numpy(float),
            ),
        ),
        "clean_corrupt_auc": _equal_shoe_pair_mean(
            valid,
            lambda unit: _binary_auc(
                unit["severity"].to_numpy(float) > 0,
                unit["score"].to_numpy(float),
            ),
        ),
        "pairwise_rank_accuracy": _equal_shoe_pair_mean(valid, _ranking_accuracy),
        "score_dynamic_range": _equal_shoe_pair_mean(
            valid,
            lambda unit: unit["score"].max() - unit["score"].min(),
        ),
        "overlap_mean": _equal_shoe_pair_mean(
            valid,
            lambda unit: unit["overlap"].mean(),
        )
        if "overlap" in valid
        else float("nan"),
    }
    for dimension in HUMAN_RATING_COLUMNS:
        if dimension in valid:
            endpoints[f"human_{dimension}_disagreement_spearman"] = (
                _equal_shoe_pair_mean(
                    valid,
                    lambda unit, column=dimension: _safe_spearman(
                        unit["score"].to_numpy(float),
                        -unit[column].to_numpy(float),
                    ),
                )
            )
    return endpoints


def _cluster_bootstrap_endpoints(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    shoe_ids = rows["shoe_id"].drop_duplicates().to_numpy()
    if repeats <= 0 or len(shoe_ids) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        sampled_ids = rng.choice(shoe_ids, size=len(shoe_ids), replace=True)
        parts = []
        for draw, shoe_id in enumerate(sampled_ids):
            part = rows[rows["shoe_id"] == shoe_id].copy()
            part["shoe_id"] = f"{shoe_id}#{draw}"
            parts.append(part)
        endpoints = _compute_endpoints(pd.concat(parts, ignore_index=True))
        for name, value in endpoints.items():
            if np.isfinite(value):
                samples.setdefault(name, []).append(value)
    return {
        name: tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
        for name, values in samples.items()
        if values
    }


def summarize_fidelity(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    bootstrap_repeats: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize metric fidelity; larger score is assumed to mean more inconsistency."""

    required = {
        "shoe_id",
        "pair_id",
        "analysis_tier",
        "severity",
        "score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"result table is missing columns: {sorted(missing)}")
    working = frame.copy()
    if "abstained" not in working:
        working["abstained"] = False
    working["_valid"] = (~working["abstained"].astype(bool)) & np.isfinite(working["score"])
    groups = _available_groups(working, group_columns)
    iterator = working.groupby(groups, dropna=False, sort=True) if groups else [((), working)]
    records = []
    for key, all_rows in iterator:
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(groups, key_values))
        endpoints = _compute_endpoints(all_rows)
        intervals = _cluster_bootstrap_endpoints(
            all_rows,
            repeats=bootstrap_repeats,
            seed=seed,
        )
        record.update(
            {
                "n_total": len(all_rows),
                "n_valid": int(all_rows["_valid"].sum()),
                **endpoints,
            }
        )
        for name, (low, high) in intervals.items():
            record[f"{name}_ci_low"] = low
            record[f"{name}_ci_high"] = high
        records.append(record)
    return pd.DataFrame(records)


def summarize_attribution(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    bootstrap_repeats: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate 2x2 feature/geometry/interaction effects without conflating them."""

    required = {
        "shoe_id",
        "pair_id",
        "feature_effect",
        "geometry_effect",
        "interaction",
        "total_effect",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"attribution table is missing columns: {sorted(missing)}")
    valid = frame.copy()
    if "abstained" in valid:
        valid = valid[~valid["abstained"].astype(bool)]
    groups = _available_groups(valid, group_columns)
    iterator = valid.groupby(groups, dropna=False, sort=True) if groups else [((), valid)]
    records = []
    for key, rows in iterator:
        key_values = key if isinstance(key, tuple) else (key,)
        endpoints = _compute_attribution_endpoints(rows)
        intervals = _cluster_bootstrap_attribution(
            rows,
            repeats=bootstrap_repeats,
            seed=seed,
        )
        feature_abs = endpoints["feature_effect_abs_mean"]
        geometry_abs = endpoints["geometry_effect_abs_mean"]
        denominator = feature_abs + geometry_abs
        record = {
            **dict(zip(groups, key_values)),
            "n_valid": len(rows),
            "n_shoes": rows["shoe_id"].nunique(),
            **{
                name: value
                for name, value in endpoints.items()
                if not name.endswith("_abs_mean")
            },
            "feature_share_abs": feature_abs / denominator
            if denominator > 0
            else float("nan"),
        }
        for name, (low, high) in intervals.items():
            if name.endswith("_abs_mean"):
                continue
            record[f"{name}_ci_low"] = low
            record[f"{name}_ci_high"] = high
        records.append(record)
    return pd.DataFrame(records)


def _compute_attribution_endpoints(rows: pd.DataFrame) -> dict[str, float]:
    return {
        f"{column}_mean": _equal_shoe_pair_mean(
            rows,
            lambda unit, name=column: unit[name].mean(),
        )
        for column in (
            "feature_effect",
            "geometry_effect",
            "interaction",
            "total_effect",
        )
    } | {
        "feature_effect_abs_mean": _equal_shoe_pair_mean(
            rows,
            lambda unit: unit["feature_effect"].abs().mean(),
        ),
        "geometry_effect_abs_mean": _equal_shoe_pair_mean(
            rows,
            lambda unit: unit["geometry_effect"].abs().mean(),
        ),
    }


def _cluster_bootstrap_attribution(
    rows: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    shoe_ids = rows["shoe_id"].drop_duplicates().to_numpy()
    if repeats <= 0 or len(shoe_ids) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        sampled_ids = rng.choice(shoe_ids, size=len(shoe_ids), replace=True)
        parts = []
        for draw, shoe_id in enumerate(sampled_ids):
            part = rows[rows["shoe_id"] == shoe_id].copy()
            part["shoe_id"] = f"{shoe_id}#{draw}"
            parts.append(part)
        endpoints = _compute_attribution_endpoints(
            pd.concat(parts, ignore_index=True)
        )
        for name, value in endpoints.items():
            if np.isfinite(value):
                samples.setdefault(name, []).append(value)
    return {
        name: tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
        for name, values in samples.items()
        if values
    }


def plot_fidelity_summary(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    bootstrap_repeats: int = 500,
    seed: int = 42,
) -> Path:
    """Plot pair-baseline-normalized curves with shoe-cluster bootstrap bands."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = {"shoe_id", "pair_id", "severity", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"plot table is missing columns: {sorted(missing)}")
    normalized_parts = []
    for _, unit in frame.groupby(["shoe_id", "pair_id"], dropna=False):
        unit = unit.copy()
        baseline_severity = unit["severity"].min()
        baseline = unit.loc[unit["severity"] == baseline_severity, "score"].mean()
        unit["score_delta"] = unit["score"] - baseline
        normalized_parts.append(unit)
    normalized = pd.concat(normalized_parts, ignore_index=True)
    label_columns = _available_groups(frame, ("backbone", "resolution", "corruption"))
    figure, axis = plt.subplots(figsize=(8, 5))
    iterator = (
        normalized.groupby(label_columns, dropna=False, sort=True)
        if label_columns
        else [((), normalized)]
    )
    for key, rows in iterator:
        by_shoe = (
            rows.groupby(["shoe_id", "severity"], dropna=False)["score_delta"]
            .mean()
            .reset_index()
        )
        summary = by_shoe.groupby("severity")["score_delta"].mean().reset_index()
        shoe_ids = by_shoe["shoe_id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(seed)
        bootstrap_by_severity: dict[float, list[float]] = {
            float(severity): [] for severity in summary["severity"]
        }
        if bootstrap_repeats > 0 and len(shoe_ids) >= 2:
            for _ in range(bootstrap_repeats):
                sampled = rng.choice(shoe_ids, size=len(shoe_ids), replace=True)
                for severity in bootstrap_by_severity:
                    values = [
                        by_shoe[
                            (by_shoe["shoe_id"] == shoe_id)
                            & (by_shoe["severity"] == severity)
                        ]["score_delta"].mean()
                        for shoe_id in sampled
                    ]
                    finite = [value for value in values if np.isfinite(value)]
                    if finite:
                        bootstrap_by_severity[severity].append(float(np.mean(finite)))
        lows = []
        highs = []
        for severity, mean in zip(summary["severity"], summary["score_delta"]):
            values = bootstrap_by_severity[float(severity)]
            if values:
                low, high = np.quantile(values, [0.025, 0.975])
            else:
                low = high = mean
            lows.append(low)
            highs.append(high)
        key_values = key if isinstance(key, tuple) else (key,)
        label = " / ".join(str(value) for value in key_values) if key_values else "metric"
        axis.plot(summary["severity"], summary["score_delta"], marker="o", label=label)
        axis.fill_between(
            summary["severity"],
            lows,
            highs,
            alpha=0.15,
        )
    axis.set_xlabel("Controlled inconsistency severity")
    axis.set_ylabel("Change from pair clean baseline (higher = worse)")
    axis.grid(alpha=0.25)
    if len(frame):
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
