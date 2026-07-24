from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from met3r.geometry_perturbations import (
    perturb_depth,
    perturb_extrinsics,
    write_geometry_calibration_artifacts,
)


def test_pose_perturbation_has_requested_rotation_and_translation() -> None:
    extrinsics = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    extrinsics[1, 0, 3] = -2.0
    original_center = np.linalg.inv(extrinsics[1])[:3, 3]

    changed = perturb_extrinsics(
        extrinsics,
        view_index=1,
        rotation_degrees=10.0,
        translation_fraction=0.05,
        axis="y",
    )

    np.testing.assert_array_equal(changed[0], extrinsics[0])
    relative = changed[1, :3, :3] @ extrinsics[1, :3, :3].T
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    np.testing.assert_allclose(angle, 10.0, atol=1e-6)
    changed_center = np.linalg.inv(changed[1])[:3, 3]
    np.testing.assert_allclose(
        np.linalg.norm(changed_center - original_center),
        0.05,
        atol=1e-6,
    )


def test_rotation_without_translation_preserves_nonzero_camera_center() -> None:
    extrinsics = np.eye(4)[None]
    extrinsics[0, :3, 3] = [0.4, -0.2, 1.3]
    original_center = np.linalg.inv(extrinsics[0])[:3, 3]

    changed = perturb_extrinsics(
        extrinsics,
        view_index=0,
        rotation_degrees=5.0,
        scene_scale=1.0,
    )

    np.testing.assert_allclose(
        np.linalg.inv(changed[0])[:3, 3],
        original_center,
        atol=1e-7,
    )


def test_depth_perturbation_is_seeded_and_mask_limited() -> None:
    depth = np.ones((2, 4, 4), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[:, 1:3, 1:3] = True

    first = perturb_depth(depth, mask, scale=1.1, noise_std=0.05, seed=9)
    second = perturb_depth(depth, mask, scale=1.1, noise_std=0.05, seed=9)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[~mask], depth[~mask])
    assert not np.array_equal(first[mask], depth[mask])


def test_geometry_calibration_writer_materializes_relative_sweeps(tmp_path: Path) -> None:
    extrinsics = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    extrinsics[1, 0, 3] = -2.0
    source = tmp_path / "base.npz"
    np.savez(
        source,
        depth=np.ones((2, 3, 3), dtype=np.float32),
        mask=np.ones((2, 3, 3), dtype=bool),
        confidence=np.ones((2, 3, 3), dtype=np.float32),
        intrinsics=np.broadcast_to(np.eye(3), (2, 3, 3)),
        extrinsics=extrinsics,
    )

    manifest_path = write_geometry_calibration_artifacts(
        source,
        tmp_path / "sweep",
        view_index=1,
        rotation_degrees=(5.0,),
        translation_fractions=(0.02,),
        depth_scales=(1.05,),
        depth_noise_stds=(0.01,),
        seed=7,
    )

    manifest = json.loads(manifest_path.read_text())
    assert {item["kind"] for item in manifest["artifacts"]} == {
        "rotation",
        "relative_translation",
        "depth_scale",
        "relative_depth_noise",
    }
    assert manifest["scene_scale"] == 1.0
    assert all(Path(item["path"]).is_file() for item in manifest["artifacts"])
    depth_scale_path = next(
        Path(item["path"]) for item in manifest["artifacts"] if item["kind"] == "depth_scale"
    )
    scaled = np.load(depth_scale_path)["depth"]
    np.testing.assert_array_equal(scaled[0], np.ones((3, 3), dtype=np.float32))
    np.testing.assert_allclose(scaled[1], 1.05)
