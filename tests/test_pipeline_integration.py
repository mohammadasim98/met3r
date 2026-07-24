from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from met3r.analysis import summarize_attribution
from met3r.corruptions import CorruptionSpec, apply_corruption_with_mask
from met3r.da3_depth import run_fixed_camera_depth
from met3r.identity import image_set_id
from met3r.runner import score_feature_geometry_artifacts


class _FakeDepthModel:
    def inference(self, **kwargs: object) -> SimpleNamespace:
        extrinsics = np.asarray(kwargs["extrinsics"])
        intrinsics = np.asarray(kwargs["intrinsics"])
        count = extrinsics.shape[0]
        return SimpleNamespace(
            depth=np.ones((count, 4, 4), dtype=np.float32),
            conf=np.ones((count, 4, 4), dtype=np.float32),
            processed_images=np.zeros((count, 4, 4, 3), dtype=np.uint8),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )


def _write_feature_artifact(
    path: Path,
    features: np.ndarray,
    images: list[Path],
) -> None:
    np.savez_compressed(
        path,
        feature__native=features,
        meta__artifact_type=np.asarray("dino_features"),
        meta__backbone_name=np.asarray("fake_dino"),
        meta__backbone_revision=np.asarray("revision"),
        meta__input_size=np.asarray(4),
        meta__view_ids=np.asarray(["left", "right"]),
        meta__camera_groups=np.asarray(["upper", "upper"]),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
        meta__analysis_tier=np.asarray("primary"),
        meta__image_set_id=np.asarray(image_set_id(images)),
        meta__bilinear_scales=np.asarray([], dtype=np.int64),
        meta__naf_scales=np.asarray([], dtype=np.int64),
        meta__naf_revision=np.asarray(""),
        meta__naf_guide_ablations=np.asarray(False),
        meta__runtime_fingerprint=np.asarray("feature-runtime"),
    )


def test_corruption_mask_depth_strict_score_and_analysis_pipeline(
    tmp_path: Path,
) -> None:
    clean_array = np.full((4, 4, 3), 255, dtype=np.uint8)
    clean_array[1:3, 1:3] = [50, 100, 200]
    clean_mask = np.zeros((4, 4), dtype=bool)
    clean_mask[1:3, 1:3] = True
    clean_paths = []
    for name in ("left.png", "right.png"):
        path = tmp_path / "clean" / name
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(clean_array).save(path)
        clean_paths.append(path)
    changed_array, changed_mask = apply_corruption_with_mask(
        clean_array,
        clean_mask,
        CorruptionSpec("translation", 0.25),
    )
    changed_paths = [tmp_path / "changed" / "left.png", tmp_path / "changed" / "right.png"]
    changed_paths[0].parent.mkdir(exist_ok=True)
    Image.fromarray(clean_array).save(changed_paths[0])
    Image.fromarray(changed_array).save(changed_paths[1])

    extrinsics = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    intrinsics = np.broadcast_to(
        np.array([[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]]),
        (2, 3, 3),
    ).copy()
    base_masks = np.stack((clean_mask, clean_mask))
    adaptive_masks = np.stack((clean_mask, changed_mask))
    provenance = {
        "revision": "source-revision",
        "dirty": False,
        "dirty_fingerprint": None,
    }
    runtime = {"python": "test", "packages": {}}
    base = run_fixed_camera_depth(
        _FakeDepthModel(),
        clean_paths,
        extrinsics,
        intrinsics,
        tmp_path / "base",
        model_id="fake/da3",
        model_revision="weights-revision",
        camera_provenance="fixed_precomputed_da3",
        foreground_mask=base_masks,
        view_ids=("left", "right"),
        camera_groups=("upper", "upper"),
        analysis_tier="primary",
        source_state=provenance,
        runtime_state=runtime,
    )
    adaptive = run_fixed_camera_depth(
        _FakeDepthModel(),
        changed_paths,
        extrinsics,
        intrinsics,
        tmp_path / "adaptive",
        model_id="fake/da3",
        model_revision="weights-revision",
        camera_provenance="fixed_precomputed_da3",
        foreground_mask=adaptive_masks,
        view_ids=("left", "right"),
        camera_groups=("upper", "upper"),
        analysis_tier="primary",
        source_state=provenance,
        runtime_state=runtime,
    )

    clean_features = np.ones((2, 3, 4, 4), dtype=np.float32)
    changed_features = clean_features.copy()
    changed_features[1] *= -1
    clean_feature_path = tmp_path / "clean_features.npz"
    changed_feature_path = tmp_path / "changed_features.npz"
    _write_feature_artifact(clean_feature_path, clean_features, clean_paths)
    _write_feature_artifact(changed_feature_path, changed_features, changed_paths)

    rows = score_feature_geometry_artifacts(
        clean_feature_path,
        changed_feature_path,
        base.arrays_path,
        adaptive.arrays_path,
        first_index=0,
        second_index=1,
        metadata={
            "shoe_id": "shoe-a",
            "backbone": "fake_dino",
            "corruption": "translation",
            "severity": 0.25,
        },
        strict_metadata=True,
    )
    summary = summarize_attribution(pd.DataFrame(rows), bootstrap_repeats=0)

    assert len(rows) == 1
    assert not rows[0]["abstained"]
    assert rows[0]["feature_effect"] > 0
    assert summary.iloc[0]["feature_effect_mean"] > 0
