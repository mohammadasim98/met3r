from __future__ import annotations

import torch
import pytest

from met3r.known_camera import (
    KnownCameraConfig,
    score_factorial_known_camera_pair,
    score_known_camera_pair,
    score_symmetric_geometry_support,
    score_symmetric_known_camera_pair,
)


def test_known_camera_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="relative_depth_tolerance"):
        KnownCameraConfig(relative_depth_tolerance=-0.1)
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        KnownCameraConfig(min_overlap_fraction=1.1)


def test_zero_valid_pixels_always_abstains_even_with_zero_overlap_threshold() -> None:
    features = torch.ones(1, 3, 2, 2)
    depth = torch.ones(1, 2, 2)
    intrinsics = torch.eye(3).unsqueeze(0)
    extrinsics = torch.eye(4).unsqueeze(0)
    empty_mask = torch.zeros(1, 2, 2, dtype=torch.bool)

    result = score_known_camera_pair(
        features,
        features,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        target_mask=empty_mask,
        source_mask=empty_mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    assert result.abstained.item()
    assert torch.isnan(result.score).item()


def _identity_camera(batch: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [
            [4.0, 0.0, (width - 1) / 2],
            [0.0, 4.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]
    ).expand(batch, -1, -1)
    extrinsics = torch.eye(4).expand(batch, -1, -1)
    return intrinsics, extrinsics


def test_identical_features_and_geometry_have_zero_distance() -> None:
    height = width = 5
    features = torch.randn(1, 8, height, width)
    depth = torch.ones(1, height, width)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(1, height, width)

    result = score_known_camera_pair(
        features,
        features,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        target_mask=mask,
        source_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    assert result.abstained.tolist() == [False]
    torch.testing.assert_close(result.score, torch.zeros_like(result.score), atol=1e-6, rtol=0)
    assert result.valid_mask.sum().item() == height * width
    torch.testing.assert_close(
        result.cycle_error_px[result.valid_mask],
        torch.zeros(height * width),
        atol=1e-5,
        rtol=0,
    )


def test_translated_planar_camera_has_exact_projection_and_occlusion_support() -> None:
    height = width = 5
    target = torch.zeros(1, width, height, width)
    for x in range(width):
        target[:, x, :, x] = 1
    source = torch.zeros_like(target)
    source[:, :, :, : width - 1] = target[:, :, :, 1:]
    target_depth = torch.ones(1, height, width)
    source_depth = torch.ones(1, height, width)
    source_depth[:, :, 1] = 0.5
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]]
    )
    target_extrinsics = torch.eye(4).unsqueeze(0)
    source_extrinsics = torch.eye(4).unsqueeze(0)
    source_extrinsics[:, 0, 3] = -1.0

    result = score_known_camera_pair(
        target,
        source,
        target_depth,
        source_depth,
        intrinsics,
        intrinsics,
        target_extrinsics,
        source_extrinsics,
        target_mask=mask,
        source_mask=mask,
        config=KnownCameraConfig(
            relative_depth_tolerance=0.01,
            min_overlap_fraction=0.0,
        ),
    )

    assert result.valid_mask.sum().item() == height * 3
    assert not result.valid_mask[0, :, 0].any()
    assert not result.valid_mask[0, :, 2].any()
    torch.testing.assert_close(
        result.projected_xy[0, 0, 2, 3],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(result.score, torch.tensor([0.0]), atol=1e-6, rtol=0)


def test_symmetric_score_is_invariant_to_view_order() -> None:
    height = width = 5
    first = torch.randn(1, 8, height, width)
    second = torch.randn(1, 8, height, width)
    depth = torch.ones(1, height, width)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(1, height, width)

    forward = score_symmetric_known_camera_pair(
        first,
        second,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )
    reverse = score_symmetric_known_camera_pair(
        second,
        first,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    torch.testing.assert_close(forward.score, reverse.score)
    torch.testing.assert_close(forward.first_to_second.score, reverse.second_to_first.score)
    torch.testing.assert_close(forward.second_to_first.score, reverse.first_to_second.score)


def test_geometry_support_matches_symmetric_feature_score_validity_fields() -> None:
    height = width = 5
    first_features = torch.randn(1, 8, height, width)
    second_features = torch.randn(1, 8, height, width)
    depth = torch.ones(1, height, width)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(1, height, width)
    config = KnownCameraConfig(min_overlap_fraction=0.0)

    support = score_symmetric_geometry_support(
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=config,
    )
    feature_score = score_symmetric_known_camera_pair(
        first_features,
        second_features,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=config,
    )

    assert torch.equal(support.abstained, feature_score.abstained)
    for support_direction, feature_direction in (
        (support.first_to_second, feature_score.first_to_second),
        (support.second_to_first, feature_score.second_to_first),
    ):
        assert torch.equal(support_direction.valid_mask, feature_direction.valid_mask)
        assert torch.equal(support_direction.abstained, feature_direction.abstained)
        torch.testing.assert_close(
            support_direction.overlap_fraction,
            feature_direction.overlap_fraction,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            support_direction.projected_xy,
            feature_direction.projected_xy,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            support_direction.relative_depth_error,
            feature_direction.relative_depth_error,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            support_direction.cycle_error_px,
            feature_direction.cycle_error_px,
            rtol=0,
            atol=0,
        )


def test_feature_changes_cannot_change_geometry_support() -> None:
    height = width = 5
    depth = torch.ones(1, height, width)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(1, height, width)
    config = KnownCameraConfig(min_overlap_fraction=0.0)
    support = score_symmetric_geometry_support(
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=config,
    )

    for first_features, second_features in (
        (torch.randn(1, 8, height, width), torch.randn(1, 8, height, width)),
        (torch.zeros(1, 8, height, width), torch.ones(1, 8, height, width)),
    ):
        scored = score_symmetric_known_camera_pair(
            first_features,
            second_features,
            depth,
            depth,
            intrinsics,
            intrinsics,
            extrinsics,
            extrinsics,
            first_mask=mask,
            second_mask=mask,
            config=config,
        )
        assert torch.equal(support.abstained, scored.abstained)
        assert torch.equal(
            support.first_to_second.valid_mask,
            scored.first_to_second.valid_mask,
        )
        assert torch.equal(
            support.second_to_first.valid_mask,
            scored.second_to_first.valid_mask,
        )


def test_factorial_score_separates_feature_and_geometry_effects() -> None:
    height = width = 5
    clean = torch.randn(1, 8, height, width)
    changed = -clean
    depth = torch.ones(1, height, width)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(1, height, width)

    result = score_factorial_known_camera_pair(
        clean,
        clean,
        changed,
        clean,
        depth,
        depth,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    torch.testing.assert_close(result.s00.score, torch.tensor([0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.feature_effect, torch.tensor([2.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(result.geometry_effect, torch.tensor([0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.interaction, torch.tensor([0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        result.total,
        result.feature_effect + result.geometry_effect + result.interaction,
    )
