from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from met3r.known_camera import score_known_camera_pair
from met3r.visualization import save_direction_diagnostics


def test_direction_diagnostic_figure_is_written(tmp_path: Path) -> None:
    features = torch.ones(1, 3, 4, 4)
    depth = torch.ones(1, 4, 4)
    intrinsics = torch.eye(3).unsqueeze(0)
    extrinsics = torch.eye(4).unsqueeze(0)
    result = score_known_camera_pair(
        features,
        features,
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
    )
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    output = tmp_path / "diagnostic.png"

    save_direction_diagnostics(image, image, result, output)

    assert output.is_file()
    assert output.stat().st_size > 0
