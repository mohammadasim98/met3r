"""Convert tensor metric outputs into analysis-ready scalar result rows."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .known_camera import FactorialMetricResult, PairMetricResult


def _scalar(value: Tensor, batch_index: int) -> float:
    return float(value.detach().cpu().reshape(-1)[batch_index].item())


def _masked_median(value: Tensor, mask: Tensor, batch_index: int) -> float:
    selected = value[batch_index][mask[batch_index]]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.detach().float().median().cpu().item())


def _direction_diagnostics(
    direction: PairMetricResult,
    batch_index: int,
) -> tuple[float, float]:
    return (
        _masked_median(direction.cycle_error_px, direction.valid_mask, batch_index),
        _masked_median(direction.relative_depth_error, direction.valid_mask, batch_index),
    )


def _finite_mean(values: tuple[float, ...]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def flatten_factorial_result(
    result: FactorialMetricResult,
    *,
    batch_index: int = 0,
) -> dict[str, float | bool]:
    """Flatten one batch element and retain geometry-quality diagnostics."""

    directions = (result.s11.first_to_second, result.s11.second_to_first)
    cycle_values, depth_values = zip(
        *[_direction_diagnostics(direction, batch_index) for direction in directions]
    )
    overlap = min(
        _scalar(result.s11.first_to_second.overlap_fraction, batch_index),
        _scalar(result.s11.second_to_first.overlap_fraction, batch_index),
    )
    return {
        "s00": _scalar(result.s00.score, batch_index),
        "s10": _scalar(result.s10.score, batch_index),
        "s01": _scalar(result.s01.score, batch_index),
        "s11": _scalar(result.s11.score, batch_index),
        "score": _scalar(result.s11.score, batch_index),
        "feature_effect": _scalar(result.feature_effect, batch_index),
        "geometry_effect": _scalar(result.geometry_effect, batch_index),
        "interaction": _scalar(result.interaction, batch_index),
        "total_effect": _scalar(result.total, batch_index),
        "overlap": overlap,
        "abstained": bool(result.s11.abstained.reshape(-1)[batch_index].item()),
        "cycle_error_px_median": _finite_mean(cycle_values),
        "relative_depth_error_median": _finite_mean(depth_values),
    }
