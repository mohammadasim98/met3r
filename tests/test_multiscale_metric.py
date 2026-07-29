from __future__ import annotations

import torch

from met3r.experiment import ViewGeometry, score_multiscale_factorial
from met3r.known_camera import KnownCameraConfig


def _geometry(size: int) -> ViewGeometry:
    intrinsics = torch.tensor(
        [[[4.0, 0.0, (size - 1) / 2], [0.0, 4.0, (size - 1) / 2], [0.0, 0.0, 1.0]]]
    )
    return ViewGeometry(
        depth=torch.ones(1, size, size),
        confidence=torch.ones(1, size, size),
        mask=torch.ones(1, size, size, dtype=torch.bool),
        intrinsics=intrinsics,
        extrinsics_w2c=torch.eye(4).unsqueeze(0),
    )


def test_multiscale_factorial_scores_every_matching_feature_key() -> None:
    clean = {
        "native": torch.randn(1, 4, 4, 4),
        "bilinear_8": torch.randn(1, 4, 8, 8),
    }
    first_changed = {name: -value for name, value in clean.items()}
    geometry = _geometry(8)

    results = score_multiscale_factorial(
        clean,
        clean,
        first_changed,
        clean,
        geometry,
        geometry,
        geometry,
        geometry,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
        confidence_percentile=0.0,
    )

    assert set(results) == {"native", "bilinear_8"}
    for result in results.values():
        torch.testing.assert_close(
            result.feature_effect,
            torch.tensor([2.0]),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            result.geometry_effect,
            torch.tensor([0.0]),
            atol=1e-6,
            rtol=0,
        )
