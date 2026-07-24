from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from met3r.runner import (
    score_feature_geometry_artifacts,
    score_geometry_sensitivity_artifacts,
)


def test_artifact_runner_scores_all_feature_scales(tmp_path: Path) -> None:
    clean = np.ones((2, 3, 3, 3), dtype=np.float32)
    changed = clean.copy()
    changed[0, 0, 1, 1] = -1
    clean_path = tmp_path / "clean.npz"
    changed_path = tmp_path / "changed.npz"
    np.savez(clean_path, feature__native=clean, feature__bilinear_6=np.repeat(np.repeat(clean, 2, 2), 2, 3))
    np.savez(
        changed_path,
        feature__native=changed,
        feature__bilinear_6=np.repeat(np.repeat(changed, 2, 2), 2, 3),
    )

    extrinsics = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    geometry_path = tmp_path / "geometry.npz"
    np.savez(
        geometry_path,
        depth=np.ones((2, 3, 3), dtype=np.float32),
        confidence=np.ones((2, 3, 3), dtype=np.float32),
        mask=np.ones((2, 3, 3), dtype=bool),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    rows = score_feature_geometry_artifacts(
        clean_path,
        changed_path,
        geometry_path,
        geometry_path,
        first_index=0,
        second_index=1,
        metadata={"shoe_id": "test", "backbone": "fake"},
    )

    assert {row["resolution"] for row in rows} == {"native", "bilinear_6"}
    assert all(row["shoe_id"] == "test" for row in rows)
    assert all(row["feature_effect"] > 0 for row in rows)
    assert all(row["geometry_effect"] == 0 for row in rows)


def test_geometry_sensitivity_runner_allows_explicit_pose_perturbation(
    tmp_path: Path,
) -> None:
    features = np.zeros((2, 3, 4, 4), dtype=np.float32)
    features[:, 0] = 1
    features[:, 1] = np.linspace(0, 1, 4)[None, None, :]
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, feature__native=features)
    extrinsics = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    intrinsics = np.broadcast_to(
        np.array([[2.0, 0, 1.5], [0, 2.0, 1.5], [0, 0, 1.0]]),
        (2, 3, 3),
    ).copy()
    base = tmp_path / "base.npz"
    np.savez(
        base,
        depth=np.ones((2, 4, 4), dtype=np.float32),
        mask=np.ones((2, 4, 4), dtype=bool),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    perturbed_extrinsics = extrinsics.copy()
    perturbed_extrinsics[1, 0, 3] = 0.1
    perturbed = tmp_path / "translation.npz"
    np.savez(
        perturbed,
        depth=np.ones((2, 4, 4), dtype=np.float32),
        mask=np.ones((2, 4, 4), dtype=bool),
        intrinsics=intrinsics,
        extrinsics=perturbed_extrinsics,
    )

    rows = score_geometry_sensitivity_artifacts(
        feature_path,
        base,
        perturbed,
        first_index=0,
        second_index=1,
        metadata={"kind": "relative_translation", "magnitude": 0.01},
    )

    assert len(rows) == 1
    assert rows[0]["kind"] == "relative_translation"
    assert np.isfinite(rows[0]["geometry_effect"])


def test_artifact_runner_rejects_mismatched_backbone_metadata(tmp_path: Path) -> None:
    feature = np.ones((2, 3, 2, 2), dtype=np.float32)
    clean = tmp_path / "clean.npz"
    changed = tmp_path / "changed.npz"
    np.savez(
        clean,
        feature__native=feature,
        meta__backbone_name=np.asarray("dino"),
        meta__backbone_revision=np.asarray("aaa"),
        meta__view_ids=np.asarray(["left", "right"]),
    )
    np.savez(
        changed,
        feature__native=feature,
        meta__backbone_name=np.asarray("dinov2"),
        meta__backbone_revision=np.asarray("bbb"),
        meta__view_ids=np.asarray(["left", "right"]),
    )
    geometry = tmp_path / "geometry.npz"
    np.savez(
        geometry,
        depth=np.ones((2, 2, 2), dtype=np.float32),
        mask=np.ones((2, 2, 2), dtype=bool),
        intrinsics=np.broadcast_to(np.eye(3), (2, 3, 3)),
        extrinsics=np.broadcast_to(np.eye(4), (2, 4, 4)),
        meta__view_ids=np.asarray(["left", "right"]),
        meta__camera_groups=np.asarray(["upper", "upper"]),
        meta__analysis_tier=np.asarray("primary"),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
    )

    with pytest.raises(ValueError, match="backbone"):
        score_feature_geometry_artifacts(
            clean,
            changed,
            geometry,
            geometry,
            first_index=0,
            second_index=1,
        )


def test_strict_artifact_runner_requires_content_identity(tmp_path: Path) -> None:
    feature = np.ones((2, 3, 2, 2), dtype=np.float32)
    feature_path = tmp_path / "features.npz"
    np.savez(
        feature_path,
        feature__native=feature,
        meta__backbone_name=np.asarray("dino"),
        meta__backbone_revision=np.asarray("revision"),
        meta__input_size=np.asarray(32),
        meta__view_ids=np.asarray(["left", "right"]),
        meta__camera_groups=np.asarray(["upper", "upper"]),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
        meta__analysis_tier=np.asarray("primary"),
    )
    geometry_path = tmp_path / "geometry.npz"
    np.savez(
        geometry_path,
        depth=np.ones((2, 2, 2), dtype=np.float32),
        mask=np.ones((2, 2, 2), dtype=bool),
        intrinsics=np.broadcast_to(np.eye(3), (2, 3, 3)),
        extrinsics=np.broadcast_to(np.eye(4), (2, 4, 4)),
        meta__view_ids=np.asarray(["left", "right"]),
        meta__camera_groups=np.asarray(["upper", "upper"]),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
        meta__analysis_tier=np.asarray("primary"),
    )

    with pytest.raises(ValueError, match="image_set_id"):
        score_feature_geometry_artifacts(
            feature_path,
            feature_path,
            geometry_path,
            geometry_path,
            first_index=0,
            second_index=1,
            strict_metadata=True,
        )
