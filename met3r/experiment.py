"""High-level, feature-scale-aware experiment scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry import confidence_percentile_mask, resize_camera_geometry
from .known_camera import (
    FactorialMetricResult,
    KnownCameraConfig,
    score_factorial_known_camera_pair,
)


@dataclass(frozen=True)
class ViewGeometry:
    depth: Tensor
    mask: Tensor
    intrinsics: Tensor
    extrinsics_w2c: Tensor
    confidence: Tensor | None = None


def _filtered_mask(view: ViewGeometry, confidence_percentile: float) -> Tensor:
    if view.confidence is None:
        return view.mask
    filtered, _ = confidence_percentile_mask(
        view.mask,
        view.confidence,
        percentile=confidence_percentile,
    )
    return filtered


def _validate_feature_keys(*feature_sets: Mapping[str, Tensor]) -> tuple[str, ...]:
    expected = tuple(feature_sets[0].keys())
    expected_set = set(expected)
    for feature_set in feature_sets[1:]:
        if set(feature_set) != expected_set:
            raise ValueError("all feature sets must have identical keys")
    return expected


def score_multiscale_factorial(
    first_clean_features: Mapping[str, Tensor],
    second_clean_features: Mapping[str, Tensor],
    first_changed_features: Mapping[str, Tensor],
    second_changed_features: Mapping[str, Tensor],
    first_base: ViewGeometry,
    second_base: ViewGeometry,
    first_adaptive: ViewGeometry,
    second_adaptive: ViewGeometry,
    *,
    config: KnownCameraConfig | None = None,
    confidence_percentile: float = 10.0,
) -> dict[str, FactorialMetricResult]:
    """Compute 2x2 attribution for every shared multi-resolution feature key."""

    keys = _validate_feature_keys(
        first_clean_features,
        second_clean_features,
        first_changed_features,
        second_changed_features,
    )
    first_shared_mask = _filtered_mask(first_base, confidence_percentile) & _filtered_mask(
        first_adaptive,
        confidence_percentile,
    )
    second_shared_mask = _filtered_mask(second_base, confidence_percentile) & _filtered_mask(
        second_adaptive,
        confidence_percentile,
    )
    if not torch.allclose(first_base.intrinsics, first_adaptive.intrinsics):
        raise ValueError("first-view base and adaptive intrinsics must be fixed")
    if not torch.allclose(second_base.intrinsics, second_adaptive.intrinsics):
        raise ValueError("second-view base and adaptive intrinsics must be fixed")
    if not torch.allclose(first_base.extrinsics_w2c, first_adaptive.extrinsics_w2c):
        raise ValueError("first-view base and adaptive extrinsics must be fixed")
    if not torch.allclose(second_base.extrinsics_w2c, second_adaptive.extrinsics_w2c):
        raise ValueError("second-view base and adaptive extrinsics must be fixed")

    results: dict[str, FactorialMetricResult] = {}
    for key in keys:
        feature_tensors = (
            first_clean_features[key],
            second_clean_features[key],
            first_changed_features[key],
            second_changed_features[key],
        )
        spatial_shapes = {tuple(value.shape[-2:]) for value in feature_tensors}
        if len(spatial_shapes) != 1:
            raise ValueError(f"feature shapes for {key!r} do not match")
        output_size = next(iter(spatial_shapes))
        first_base_scaled = resize_camera_geometry(
            first_base.depth,
            first_shared_mask,
            first_base.confidence,
            first_base.intrinsics,
            output_size=output_size,
        )
        second_base_scaled = resize_camera_geometry(
            second_base.depth,
            second_shared_mask,
            second_base.confidence,
            second_base.intrinsics,
            output_size=output_size,
        )
        first_adaptive_scaled = resize_camera_geometry(
            first_adaptive.depth,
            first_shared_mask,
            first_adaptive.confidence,
            first_adaptive.intrinsics,
            output_size=output_size,
        )
        second_adaptive_scaled = resize_camera_geometry(
            second_adaptive.depth,
            second_shared_mask,
            second_adaptive.confidence,
            second_adaptive.intrinsics,
            output_size=output_size,
        )
        results[key] = score_factorial_known_camera_pair(
            *feature_tensors,
            first_base_scaled.depth,
            second_base_scaled.depth,
            first_adaptive_scaled.depth,
            second_adaptive_scaled.depth,
            first_base_scaled.intrinsics,
            second_base_scaled.intrinsics,
            first_base.extrinsics_w2c,
            second_base.extrinsics_w2c,
            first_mask=first_base_scaled.mask,
            second_mask=second_base_scaled.mask,
            config=config,
        )
    return results
