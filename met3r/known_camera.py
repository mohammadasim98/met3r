"""Known-camera feature reprojection and consistency scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class KnownCameraConfig:
    """Thresholds used to establish valid cross-view correspondences."""

    relative_depth_tolerance: float = 0.03
    min_overlap_fraction: float = 0.10
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.relative_depth_tolerance < 0:
            raise ValueError("relative_depth_tolerance must be non-negative")
        if not 0 <= self.min_overlap_fraction <= 1:
            raise ValueError("min_overlap_fraction must be in [0, 1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class PairMetricResult:
    """Directional feature comparison from a target into a source view."""

    score: Tensor
    score_map: Tensor
    valid_mask: Tensor
    overlap_fraction: Tensor
    abstained: Tensor
    projected_xy: Tensor
    relative_depth_error: Tensor
    cycle_error_px: Tensor


@dataclass(frozen=True)
class SymmetricPairMetricResult:
    """Both directional comparisons and their abstention-aware mean."""

    score: Tensor
    first_to_second: PairMetricResult
    second_to_first: PairMetricResult
    abstained: Tensor


@dataclass(frozen=True)
class FactorialMetricResult:
    """Four common-support scores and their 2x2 factorial attribution."""

    s00: SymmetricPairMetricResult
    s10: SymmetricPairMetricResult
    s01: SymmetricPairMetricResult
    s11: SymmetricPairMetricResult
    feature_effect: Tensor
    geometry_effect: Tensor
    interaction: Tensor
    total: Tensor


@dataclass(frozen=True)
class GeometrySensitivityResult:
    """Common-support baseline and explicitly perturbed geometry scores."""

    baseline: SymmetricPairMetricResult
    perturbed: SymmetricPairMetricResult
    geometry_effect: Tensor


def _as_mask(value: Tensor | None, reference: Tensor) -> Tensor:
    if value is None:
        return torch.ones_like(reference, dtype=torch.bool)
    return value.to(device=reference.device, dtype=torch.bool)


def _pixel_grid(batch: int, height: int, width: int, reference: Tensor) -> Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=0)
    return pixels.reshape(1, 3, -1).expand(batch, -1, -1)


def _sample(values: Tensor, grid: Tensor, *, mode: str = "bilinear") -> Tensor:
    return F.grid_sample(
        values,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def score_known_camera_pair(
    target_features: Tensor,
    source_features: Tensor,
    target_depth: Tensor,
    source_depth: Tensor,
    target_intrinsics: Tensor,
    source_intrinsics: Tensor,
    target_extrinsics_w2c: Tensor,
    source_extrinsics_w2c: Tensor,
    *,
    target_mask: Tensor | None = None,
    source_mask: Tensor | None = None,
    config: KnownCameraConfig | None = None,
) -> PairMetricResult:
    """Compare a feature map after target-depth reprojection into the source view."""

    config = config or KnownCameraConfig()
    batch, _, height, width = target_features.shape
    if source_features.shape[0] != batch:
        raise ValueError("target and source feature batches must match")
    if target_depth.shape != (batch, height, width):
        raise ValueError("target_depth must match the target feature resolution")
    if source_depth.shape != (batch, height, width):
        raise ValueError("source_depth must match the source feature resolution")

    pixels = _pixel_grid(batch, height, width, target_features)
    target_xyz = torch.linalg.solve(target_intrinsics, pixels)
    target_xyz = target_xyz * target_depth.reshape(batch, 1, -1)
    target_h = torch.cat((target_xyz, torch.ones_like(target_xyz[:, :1])), dim=1)
    world_h = torch.linalg.solve(target_extrinsics_w2c, target_h)
    source_xyz = (source_extrinsics_w2c @ world_h)[:, :3]

    projected = source_intrinsics @ source_xyz
    projected_z = projected[:, 2]
    projected_xy = projected[:, :2] / projected_z.clamp_min(config.eps).unsqueeze(1)
    x = projected_xy[:, 0]
    y = projected_xy[:, 1]
    normalized_x = 2 * x / max(width - 1, 1) - 1
    normalized_y = 2 * y / max(height - 1, 1) - 1
    grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(batch, height, width, 2)

    sampled_features = _sample(source_features, grid)
    sampled_depth = _sample(source_depth[:, None], grid)[:, 0]
    sampled_source_mask = _sample(
        _as_mask(source_mask, source_depth)[:, None].to(target_features.dtype),
        grid,
        mode="nearest",
    )[:, 0] > 0.5

    source_pixels = torch.cat((projected_xy, torch.ones_like(projected_xy[:, :1])), dim=1)
    sampled_source_xyz = torch.linalg.solve(source_intrinsics, source_pixels)
    sampled_source_xyz = sampled_source_xyz * sampled_depth.reshape(batch, 1, -1)
    sampled_source_h = torch.cat(
        (sampled_source_xyz, torch.ones_like(sampled_source_xyz[:, :1])),
        dim=1,
    )
    sampled_world_h = torch.linalg.solve(source_extrinsics_w2c, sampled_source_h)
    cycled_target_xyz = (target_extrinsics_w2c @ sampled_world_h)[:, :3]
    cycled_target_pixels = target_intrinsics @ cycled_target_xyz
    cycled_target_xy = cycled_target_pixels[:, :2] / cycled_target_pixels[:, 2:].clamp_min(
        config.eps
    )
    cycle_error_px = torch.linalg.vector_norm(cycled_target_xy - pixels[:, :2], dim=1).reshape(
        batch, height, width
    )

    target_valid = _as_mask(target_mask, target_depth)
    finite_depth = torch.isfinite(target_depth) & torch.isfinite(sampled_depth)
    positive_depth = (target_depth > 0) & (sampled_depth > 0) & (source_xyz[:, 2] > 0).reshape(
        batch, height, width
    )
    in_bounds = (
        (normalized_x >= -1)
        & (normalized_x <= 1)
        & (normalized_y >= -1)
        & (normalized_y <= 1)
    ).reshape(batch, height, width)
    source_z = source_xyz[:, 2].reshape(batch, height, width)
    relative_depth_error = (sampled_depth - source_z).abs() / sampled_depth.abs().clamp_min(
        config.eps
    )
    depth_consistent = relative_depth_error <= config.relative_depth_tolerance
    valid_mask = (
        target_valid
        & sampled_source_mask
        & finite_depth
        & positive_depth
        & in_bounds
        & depth_consistent
    )

    score_map = 1 - F.cosine_similarity(target_features, sampled_features, dim=1, eps=config.eps)
    valid_count = valid_mask.flatten(1).sum(dim=1)
    foreground_count = target_valid.flatten(1).sum(dim=1).clamp_min(1)
    overlap_fraction = valid_count / foreground_count
    abstained = (valid_count == 0) | (
        overlap_fraction < config.min_overlap_fraction
    )
    weighted_sum = (score_map * valid_mask).flatten(1).sum(dim=1)
    score = weighted_sum / valid_count.clamp_min(1)
    score = torch.where(abstained, torch.full_like(score, torch.nan), score)

    return PairMetricResult(
        score=score,
        score_map=score_map,
        valid_mask=valid_mask,
        overlap_fraction=overlap_fraction,
        abstained=abstained,
        projected_xy=projected_xy.reshape(batch, 2, height, width),
        relative_depth_error=relative_depth_error,
        cycle_error_px=cycle_error_px,
    )


def score_symmetric_known_camera_pair(
    first_features: Tensor,
    second_features: Tensor,
    first_depth: Tensor,
    second_depth: Tensor,
    first_intrinsics: Tensor,
    second_intrinsics: Tensor,
    first_extrinsics_w2c: Tensor,
    second_extrinsics_w2c: Tensor,
    *,
    first_mask: Tensor | None = None,
    second_mask: Tensor | None = None,
    config: KnownCameraConfig | None = None,
) -> SymmetricPairMetricResult:
    """Average both reprojection directions, abstaining if either direction is invalid."""

    first_to_second = score_known_camera_pair(
        first_features,
        second_features,
        first_depth,
        second_depth,
        first_intrinsics,
        second_intrinsics,
        first_extrinsics_w2c,
        second_extrinsics_w2c,
        target_mask=first_mask,
        source_mask=second_mask,
        config=config,
    )
    second_to_first = score_known_camera_pair(
        second_features,
        first_features,
        second_depth,
        first_depth,
        second_intrinsics,
        first_intrinsics,
        second_extrinsics_w2c,
        first_extrinsics_w2c,
        target_mask=second_mask,
        source_mask=first_mask,
        config=config,
    )
    abstained = first_to_second.abstained | second_to_first.abstained
    score = (first_to_second.score + second_to_first.score) / 2
    score = torch.where(abstained, torch.full_like(score, torch.nan), score)
    return SymmetricPairMetricResult(
        score=score,
        first_to_second=first_to_second,
        second_to_first=second_to_first,
        abstained=abstained,
    )


def _restrict_directional_result(
    result: PairMetricResult,
    common_mask: Tensor,
    target_mask: Tensor,
    config: KnownCameraConfig,
) -> PairMetricResult:
    valid_count = common_mask.flatten(1).sum(dim=1)
    foreground_count = target_mask.flatten(1).sum(dim=1).clamp_min(1)
    overlap_fraction = valid_count / foreground_count
    abstained = (valid_count == 0) | (
        overlap_fraction < config.min_overlap_fraction
    )
    score = (result.score_map * common_mask).flatten(1).sum(dim=1) / valid_count.clamp_min(1)
    score = torch.where(abstained, torch.full_like(score, torch.nan), score)
    return PairMetricResult(
        score=score,
        score_map=result.score_map,
        valid_mask=common_mask,
        overlap_fraction=overlap_fraction,
        abstained=abstained,
        projected_xy=result.projected_xy,
        relative_depth_error=result.relative_depth_error,
        cycle_error_px=result.cycle_error_px,
    )


def _restrict_symmetric_result(
    result: SymmetricPairMetricResult,
    first_common_mask: Tensor,
    second_common_mask: Tensor,
    first_mask: Tensor,
    second_mask: Tensor,
    config: KnownCameraConfig,
) -> SymmetricPairMetricResult:
    first_to_second = _restrict_directional_result(
        result.first_to_second,
        first_common_mask,
        first_mask,
        config,
    )
    second_to_first = _restrict_directional_result(
        result.second_to_first,
        second_common_mask,
        second_mask,
        config,
    )
    abstained = first_to_second.abstained | second_to_first.abstained
    score = (first_to_second.score + second_to_first.score) / 2
    score = torch.where(abstained, torch.full_like(score, torch.nan), score)
    return SymmetricPairMetricResult(
        score=score,
        first_to_second=first_to_second,
        second_to_first=second_to_first,
        abstained=abstained,
    )


def score_factorial_known_camera_pair(
    first_clean_features: Tensor,
    second_clean_features: Tensor,
    first_changed_features: Tensor,
    second_changed_features: Tensor,
    first_base_depth: Tensor,
    second_base_depth: Tensor,
    first_adaptive_depth: Tensor,
    second_adaptive_depth: Tensor,
    first_intrinsics: Tensor,
    second_intrinsics: Tensor,
    first_extrinsics_w2c: Tensor,
    second_extrinsics_w2c: Tensor,
    *,
    first_mask: Tensor | None = None,
    second_mask: Tensor | None = None,
    config: KnownCameraConfig | None = None,
) -> FactorialMetricResult:
    """Compute common-support clean/changed feature and base/adaptive depth scores."""

    config = config or KnownCameraConfig()
    first_mask = _as_mask(first_mask, first_base_depth)
    second_mask = _as_mask(second_mask, second_base_depth)
    shared = dict(
        first_intrinsics=first_intrinsics,
        second_intrinsics=second_intrinsics,
        first_extrinsics_w2c=first_extrinsics_w2c,
        second_extrinsics_w2c=second_extrinsics_w2c,
        first_mask=first_mask,
        second_mask=second_mask,
        config=config,
    )
    raw_s00 = score_symmetric_known_camera_pair(
        first_clean_features,
        second_clean_features,
        first_base_depth,
        second_base_depth,
        **shared,
    )
    raw_s10 = score_symmetric_known_camera_pair(
        first_changed_features,
        second_changed_features,
        first_base_depth,
        second_base_depth,
        **shared,
    )
    raw_s01 = score_symmetric_known_camera_pair(
        first_clean_features,
        second_clean_features,
        first_adaptive_depth,
        second_adaptive_depth,
        **shared,
    )
    raw_s11 = score_symmetric_known_camera_pair(
        first_changed_features,
        second_changed_features,
        first_adaptive_depth,
        second_adaptive_depth,
        **shared,
    )
    raw_results = (raw_s00, raw_s10, raw_s01, raw_s11)
    first_common_mask = torch.stack(
        [item.first_to_second.valid_mask for item in raw_results]
    ).all(dim=0)
    second_common_mask = torch.stack(
        [item.second_to_first.valid_mask for item in raw_results]
    ).all(dim=0)
    s00, s10, s01, s11 = (
        _restrict_symmetric_result(
            item,
            first_common_mask,
            second_common_mask,
            first_mask,
            second_mask,
            config,
        )
        for item in raw_results
    )

    feature_effect = s10.score - s00.score
    geometry_effect = s01.score - s00.score
    interaction = s11.score - s10.score - s01.score + s00.score
    total = s11.score - s00.score
    return FactorialMetricResult(
        s00=s00,
        s10=s10,
        s01=s01,
        s11=s11,
        feature_effect=feature_effect,
        geometry_effect=geometry_effect,
        interaction=interaction,
        total=total,
    )


def score_geometry_sensitivity_pair(
    first_features: Tensor,
    second_features: Tensor,
    first_base_depth: Tensor,
    second_base_depth: Tensor,
    first_perturbed_depth: Tensor,
    second_perturbed_depth: Tensor,
    first_base_intrinsics: Tensor,
    second_base_intrinsics: Tensor,
    first_perturbed_intrinsics: Tensor,
    second_perturbed_intrinsics: Tensor,
    first_base_extrinsics_w2c: Tensor,
    second_base_extrinsics_w2c: Tensor,
    first_perturbed_extrinsics_w2c: Tensor,
    second_perturbed_extrinsics_w2c: Tensor,
    *,
    first_mask: Tensor | None = None,
    second_mask: Tensor | None = None,
    config: KnownCameraConfig | None = None,
) -> GeometrySensitivityResult:
    """Compare base and known-perturbed geometry on identical features/support."""

    config = config or KnownCameraConfig()
    first_mask = _as_mask(first_mask, first_base_depth)
    second_mask = _as_mask(second_mask, second_base_depth)
    baseline = score_symmetric_known_camera_pair(
        first_features,
        second_features,
        first_base_depth,
        second_base_depth,
        first_base_intrinsics,
        second_base_intrinsics,
        first_base_extrinsics_w2c,
        second_base_extrinsics_w2c,
        first_mask=first_mask,
        second_mask=second_mask,
        config=config,
    )
    perturbed = score_symmetric_known_camera_pair(
        first_features,
        second_features,
        first_perturbed_depth,
        second_perturbed_depth,
        first_perturbed_intrinsics,
        second_perturbed_intrinsics,
        first_perturbed_extrinsics_w2c,
        second_perturbed_extrinsics_w2c,
        first_mask=first_mask,
        second_mask=second_mask,
        config=config,
    )
    first_common = (
        baseline.first_to_second.valid_mask
        & perturbed.first_to_second.valid_mask
    )
    second_common = (
        baseline.second_to_first.valid_mask
        & perturbed.second_to_first.valid_mask
    )
    baseline = _restrict_symmetric_result(
        baseline,
        first_common,
        second_common,
        first_mask,
        second_mask,
        config,
    )
    perturbed = _restrict_symmetric_result(
        perturbed,
        first_common,
        second_common,
        first_mask,
        second_mask,
        config,
    )
    return GeometrySensitivityResult(
        baseline=baseline,
        perturbed=perturbed,
        geometry_effect=perturbed.score - baseline.score,
    )
