from __future__ import annotations

import math

import torch
import pytest

from met3r.known_camera import score_factorial_known_camera_pair
from met3r.results import flatten_factorial_result


def test_flatten_factorial_result_contains_scores_effects_and_diagnostics() -> None:
    feature = torch.ones(1, 3, 3, 3)
    changed = feature.clone()
    changed[:, 0, 1, 1] = -1
    depth = torch.ones(1, 3, 3)
    intrinsics = torch.eye(3).unsqueeze(0)
    extrinsics = torch.eye(4).unsqueeze(0)
    result = score_factorial_known_camera_pair(
        feature,
        feature,
        changed,
        feature,
        depth,
        depth,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
    )

    row = flatten_factorial_result(result)

    assert set(("s00", "s10", "s01", "s11", "feature_effect", "geometry_effect")) <= row.keys()
    assert {
        f"{cell}_{direction}_{field}"
        for cell in ("s00", "s10", "s01", "s11")
        for direction in ("first_to_second", "second_to_first")
        for field in ("score", "overlap")
    } <= row.keys()
    assert row["s00"] == pytest.approx(0.0, abs=1e-6)
    assert row["feature_effect"] > 0
    assert row["geometry_effect"] == pytest.approx(0.0, abs=1e-6)
    assert row["abstained"] is False
    assert row["overlap"] == 1.0
    assert row["s11_first_to_second_overlap"] == 1.0
    assert row["s11_second_to_first_overlap"] == 1.0
    assert row["cycle_error_px_median"] == 0.0


def test_flatten_factorial_result_keeps_missing_diagnostics_as_nan() -> None:
    feature = torch.ones(1, 3, 2, 2)
    depth = torch.ones(1, 2, 2)
    intrinsics = torch.eye(3).unsqueeze(0)
    extrinsics = torch.eye(4).unsqueeze(0)
    empty_mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    result = score_factorial_known_camera_pair(
        feature,
        feature,
        feature,
        feature,
        depth,
        depth,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=empty_mask,
        second_mask=empty_mask,
    )

    row = flatten_factorial_result(result)

    assert row["abstained"] is True
    assert math.isnan(row["cycle_error_px_median"])
    assert math.isnan(row["relative_depth_error_median"])
