from __future__ import annotations

from pathlib import Path

import pytest
import torch

import met3r.adjacent16_gate as adjacent16_gate
from met3r.adjacent16_gate import validate_adjacent16_geometry_preflight
from met3r.known_camera import KnownCameraConfig, score_symmetric_geometry_support


def _identity_camera(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [
            [4.0, 0.0, (width - 1) / 2],
            [0.0, 4.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    return intrinsics, torch.eye(4).unsqueeze(0)


def test_geometry_support_uses_directional_target_foreground_and_symmetric_abstention() -> None:
    height = width = 4
    depth = torch.ones(1, height, width)
    first_mask = torch.ones(1, height, width, dtype=torch.bool)
    second_mask = torch.zeros(1, height, width, dtype=torch.bool)
    second_mask[0, 0, 0] = True
    intrinsics, extrinsics = _identity_camera(height, width)

    support = score_symmetric_geometry_support(
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=first_mask,
        second_mask=second_mask,
        config=KnownCameraConfig(min_overlap_fraction=0.08),
    )

    torch.testing.assert_close(
        support.first_to_second.overlap_fraction,
        torch.tensor([1 / 16]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        support.second_to_first.overlap_fraction,
        torch.tensor([1.0]),
        rtol=0,
        atol=0,
    )
    assert support.first_to_second.abstained.tolist() == [True]
    assert support.second_to_first.abstained.tolist() == [False]
    assert support.abstained.tolist() == [True]


def test_geometry_support_diagnostics_are_finite_for_valid_correspondences() -> None:
    height = width = 4
    first_depth = torch.ones(1, height, width)
    second_depth = torch.ones(1, height, width)
    second_depth[0, 0, 0] = 0.0
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(height, width)

    support = score_symmetric_geometry_support(
        first_depth,
        second_depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    for direction in (support.first_to_second, support.second_to_first):
        assert torch.isfinite(direction.relative_depth_error[direction.valid_mask]).all()
        assert torch.isfinite(direction.cycle_error_px[direction.valid_mask]).all()


def test_geometry_preflight_fails_closed_when_the_frozen_run_is_missing(
    tmp_path: Path,
) -> None:
    """Catch a gate that treats a missing preflight input as passing evidence."""

    report = validate_adjacent16_geometry_preflight(tmp_path / "missing")

    assert report.passed is False
    assert report.pair_units == 0
    assert "input_validation" in report.failed_checks


def test_preflight_pair_uses_confidence_percentile_and_the_eight_percent_pass_boundary() -> None:
    """Catch applying the 20% median gate as an individual pair threshold."""

    depth = torch.ones((2, 4, 4))
    confidence = torch.stack((
        torch.linspace(1.0, 2.0, 16).reshape(4, 4),
        torch.ones((4, 4)),
    ))
    mask = torch.stack((
        torch.ones((4, 4), dtype=torch.bool),
        torch.tensor([[False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, True, True]]),
    ))
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]] * 2
    )
    extrinsics = torch.eye(4).repeat(2, 1, 1)

    scored = adjacent16_gate._score_preflight_pair(
        depth, confidence, mask, intrinsics, extrinsics, 0, 1,
    )

    assert scored["minimum_directional_overlap"] == pytest.approx(2 / 14)
    assert scored["symmetric_pass"] is True
    assert scored["minimum_directional_overlap"] < 0.20
