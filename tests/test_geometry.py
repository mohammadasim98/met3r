from __future__ import annotations

import torch

from met3r.geometry import confidence_percentile_mask, resize_camera_geometry


def test_resize_camera_geometry_scales_intrinsics_and_spatial_inputs() -> None:
    depth = torch.ones(1, 4, 6)
    mask = torch.ones(1, 4, 6, dtype=torch.bool)
    confidence = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]
    )

    resized = resize_camera_geometry(
        depth,
        mask,
        confidence,
        intrinsics,
        output_size=(8, 12),
    )

    assert resized.depth.shape == (1, 8, 12)
    assert resized.mask.shape == (1, 8, 12)
    assert resized.confidence.shape == (1, 8, 12)
    torch.testing.assert_close(
        resized.intrinsics,
        torch.tensor([[[4.0, 0.0, 6.5], [0.0, 8.0, 4.5], [0.0, 0.0, 1.0]]]),
    )


def test_confidence_percentile_mask_filters_low_foreground_values_per_image() -> None:
    foreground = torch.ones(1, 2, 5, dtype=torch.bool)
    confidence = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)

    filtered, thresholds = confidence_percentile_mask(
        foreground,
        confidence,
        percentile=20.0,
    )

    torch.testing.assert_close(thresholds, torch.tensor([1.8]))
    assert filtered.sum().item() == 8
    assert filtered[0, 0, :2].tolist() == [False, False]


def test_geometry_downsampling_selects_pixel_center_nearest_samples() -> None:
    depth = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    mask = torch.ones_like(depth, dtype=torch.bool)
    intrinsics = torch.eye(3).unsqueeze(0)

    resized = resize_camera_geometry(
        depth,
        mask,
        None,
        intrinsics,
        output_size=(2, 3),
    )

    torch.testing.assert_close(
        resized.depth,
        torch.tensor([[[7.0, 9.0, 11.0], [19.0, 21.0, 23.0]]]),
    )
