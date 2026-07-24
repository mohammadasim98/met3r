from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import met3r.bytesize_data as bytesize_data
from met3r.bytesize_data import (
    build_bytesize_manifest,
    recover_bottom_camera_alignment,
    validate_processed_image,
    write_camera_artifacts,
    write_upper_camera_artifacts,
)
from met3r.upper_data import CAMERA_DATASET_BY_SHOE, UPPER_VIEWS

def test_manifest_maps_zero_based_camera_to_one_based_source_and_mask(tmp_path: Path) -> None:
    shoe = "TEST SHOE"
    for group in ("upper", "bottom"):
        custom = tmp_path / "custom_dataset" / shoe / group
        custom.mkdir(parents=True)
        np.savez(
            custom / "processed_images.npz",
            processed_images=np.zeros((48, 2, 2, 4), dtype=np.uint8),
        )
        np.savez(
            custom / "extrinsics.npz",
            extrinsics=np.broadcast_to(np.eye(4)[:3], (48, 3, 4)),
        )
        np.savez(
            custom / "intrinsics.npz",
            intrinsics=np.broadcast_to(np.eye(3), (48, 3, 3)),
        )
        raw = tmp_path / "sample_result" / shoe / group / "glb_masked_white_bg"
        raw.joinpath("outputs").mkdir(parents=True)
        np.savez(
            raw / "outputs" / "extrinsics.npz",
            extrinsics=np.broadcast_to(np.eye(4)[:3], (48, 3, 4)),
        )
        for camera_index in range(48):
            for path in (
                raw / "masked_inputs" / f"{camera_index}.jpg",
                tmp_path / "sample_data" / shoe / group / f"{camera_index + 1}.jpg",
                tmp_path / "sample_result" / shoe / group / "mask" / f"{camera_index + 1}.jpg",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2)).save(path)

    manifest = build_bytesize_manifest(tmp_path)

    left = manifest.shoes[0].views[0]
    assert left.view_id == "left"
    assert left.camera_group == "upper"
    assert left.camera_index == 42
    assert left.image_path.name == "42.jpg"
    assert left.source_path.name == "43.jpg"
    assert left.mask_path.name == "43.jpg"
    assert manifest.source_image_size == (2, 2)
    assert manifest.processed_image_size == (2, 2)


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(np.eye(4), (extrinsics.shape[0], 4, 4)).copy()
    result[:, :3, :4] = extrinsics
    return result


def test_camera_alignment_recovers_documented_rotation_and_flags_cross_group_translation() -> None:
    upper = np.zeros((2, 3, 4), dtype=np.float64)
    bottom = np.zeros((2, 3, 4), dtype=np.float64)
    upper[:, :3, :3] = np.eye(3)
    bottom[:, :3, :3] = np.eye(3)
    upper[:, 2, 3] = 2.0
    bottom[:, 2, 3] = 3.0

    angle = np.deg2rad(30.0)
    exported_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    exported_transform = np.eye(4)
    exported_transform[:3, :3] = exported_rotation
    exported_transform[:3, 3] = [0.1, -0.2, 0.3]
    custom = (_homogeneous(bottom) @ exported_transform)[:, :3, :4]

    correction = recover_bottom_camera_alignment(upper, bottom, custom)

    axis_flip = np.diag([1.0, -1.0, -1.0])
    expected = (axis_flip @ upper[0, :3, :3]).T @ exported_rotation @ (
        axis_flip @ bottom[0, :3, :3]
    )
    np.testing.assert_allclose(correction.bottom_to_upper_rotation, expected, atol=1e-7)
    np.testing.assert_allclose(
        correction.corrected_extrinsics[:, :3, :3],
        bottom[:, :3, :3] @ expected.T,
        atol=1e-7,
    )
    np.testing.assert_allclose(correction.corrected_extrinsics[:, :3, 3], custom[:, :3, 3])
    assert correction.transform_residual < 1e-12
    assert correction.translation_is_proxy
    assert correction.cross_group_provenance.endswith("translation_proxy")


def test_camera_alignment_rejects_nonconstant_export_transform() -> None:
    upper = np.tile(np.eye(4)[:3], (2, 1, 1))
    bottom = upper.copy()
    custom = upper.copy()
    custom[1, 0, 3] = 0.01

    with pytest.raises(ValueError, match="not constant"):
        recover_bottom_camera_alignment(upper, bottom, custom, transform_tolerance=1e-5)


def test_processed_image_validation_uses_grayscale_correlation(tmp_path: Path) -> None:
    ramp = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = np.repeat(ramp[..., None], 3, axis=-1)
    image_path = tmp_path / "image.png"
    Image.fromarray(rgb).save(image_path)

    processed = np.repeat(rgb[None], 2, axis=0)
    score = validate_processed_image(image_path, processed, camera_index=1)
    assert score == pytest.approx(1.0)

    processed[1] = processed[1, ::-1]
    with pytest.raises(ValueError, match="correlation"):
        validate_processed_image(image_path, processed, camera_index=1, min_correlation=0.98)


def test_camera_artifacts_separate_primary_groups_from_cross_group_proxy(
    tmp_path: Path,
) -> None:
    shoe = "TEST SHOE"
    extrinsics = np.broadcast_to(np.eye(4)[:3], (48, 3, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (48, 3, 3)).copy()
    processed = np.zeros((48, 4, 4, 4), dtype=np.uint8)
    processed[..., 3] = 255
    export_transform = np.eye(4)
    export_transform[:3, 3] = [0.1, 0.2, 0.3]
    for group in ("upper", "bottom"):
        custom = tmp_path / "custom_dataset" / shoe / group
        custom.mkdir(parents=True)
        np.savez(custom / "processed_images.npz", processed_images=processed)
        custom_extrinsics = extrinsics
        if group == "bottom":
            custom_h = _homogeneous(extrinsics) @ export_transform
            custom_extrinsics = custom_h[:, :3]
        np.savez(custom / "extrinsics.npz", extrinsics=custom_extrinsics)
        np.savez(custom / "intrinsics.npz", intrinsics=intrinsics)
        raw = tmp_path / "sample_result" / shoe / group / "glb_masked_white_bg"
        raw.joinpath("outputs").mkdir(parents=True)
        np.savez(raw / "outputs" / "extrinsics.npz", extrinsics=extrinsics)
        for camera_index in range(48):
            for path in (
                raw / "masked_inputs" / f"{camera_index}.jpg",
                tmp_path / "sample_data" / shoe / group / f"{camera_index + 1}.jpg",
                tmp_path / "sample_result" / shoe / group / "mask" / f"{camera_index + 1}.jpg",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4)).save(path)

    manifest = build_bytesize_manifest(tmp_path)
    write_camera_artifacts(manifest, tmp_path / "artifacts")

    upper = np.load(tmp_path / "artifacts" / "primary" / shoe / "upper" / "cameras.npz")
    bottom = np.load(tmp_path / "artifacts" / "primary" / shoe / "bottom" / "cameras.npz")
    cross = np.load(tmp_path / "artifacts" / "sensitivity" / shoe / "cameras.npz")
    assert upper["extrinsics"].shape == (4, 4, 4)
    assert bottom["extrinsics"].shape == (2, 4, 4)
    assert cross["extrinsics"].shape == (6, 4, 4)
    assert upper["mask"].shape == (4, 4, 4)
    image_paths = json.loads(
        (tmp_path / "artifacts" / "primary" / shoe / "upper" / "images.json").read_text()
    )
    assert Image.open(image_paths[0]).size == (4, 4)
    metadata = json.loads(
        (tmp_path / "artifacts" / "sensitivity" / shoe / "metadata.json").read_text()
    )
    assert metadata["camera_mode"] == "fixed/precomputed DA3 cameras (not calibrated GT)"
    assert metadata["analysis_tier"] == "sensitivity_only"
    assert metadata["translation_is_proxy"] is True


def test_upper_camera_artifacts_have_only_four_504_views(tmp_path: Path) -> None:
    manifest = build_bytesize_manifest("/root/dev/ByteSize/Reconstruction")

    write_upper_camera_artifacts(
        manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )

    shoe_id = "air_huarache_run_ultra_111449"
    with np.load(tmp_path / shoe_id / "cameras.npz") as data:
        assert data["intrinsics"].shape == (4, 3, 3)
        assert data["extrinsics"].shape == (4, 4, 4)
        assert data["mask"].shape == (4, 504, 504)
        assert data["view_ids"].tolist() == list(UPPER_VIEWS)
        assert data["camera_groups"].tolist() == ["upper"] * 4
    assert len(list(tmp_path.glob("*/cameras.npz"))) == 4
    assert not list(tmp_path.rglob("bottom"))
    assert not list(tmp_path.rglob("sensitivity"))


def _write_real_upper_camera_artifacts(tmp_path: Path):
    manifest = build_bytesize_manifest("/root/dev/ByteSize/Reconstruction")
    write_upper_camera_artifacts(
        manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )
    return manifest, tmp_path / "air_huarache_run_ultra_111449"


def _camera_bundle_content(artifact_dir: Path) -> dict[str, object]:
    with np.load(artifact_dir / "cameras.npz") as artifact:
        arrays = {
            name: (
                str(np.asarray(artifact[name]).dtype),
                np.asarray(artifact[name]).shape,
                hashlib.sha256(np.asarray(artifact[name]).tobytes()).hexdigest(),
            )
            for name in artifact.files
        }
    image_paths = json.loads((artifact_dir / "images.json").read_text())
    return {
        "arrays": arrays,
        "metadata": json.loads((artifact_dir / "metadata.json").read_text()),
        "image_paths": image_paths,
        "images": {
            Path(path).name: hashlib.sha256(
                np.asarray(Image.open(path).convert("RGB")).tobytes()
            ).hexdigest()
            for path in image_paths
        },
        "files": sorted(
            str(path.relative_to(artifact_dir))
            for path in artifact_dir.rglob("*")
            if path.is_file()
        ),
    }


def test_upper_camera_resume_repairs_mutated_provenance_label(tmp_path: Path) -> None:
    manifest, artifact_dir = _write_real_upper_camera_artifacts(tmp_path)
    canonical = _camera_bundle_content(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["camera_mode"] = "calibrated GT"
    metadata_path.write_text(json.dumps(metadata))

    write_upper_camera_artifacts(
        manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )

    assert _camera_bundle_content(artifact_dir) == canonical
    repaired = json.loads(metadata_path.read_text())
    assert repaired["camera_mode"] == "fixed/precomputed DA3 cameras (not calibrated GT)"


@pytest.mark.parametrize("mutation", ("camera_array", "staged_image", "group_mapping"))
def test_upper_camera_resume_repairs_mutated_canonical_content(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest, artifact_dir = _write_real_upper_camera_artifacts(tmp_path)
    canonical = _camera_bundle_content(artifact_dir)
    if mutation == "camera_array":
        camera_path = artifact_dir / "cameras.npz"
        with np.load(camera_path) as artifact:
            arrays = {name: np.asarray(artifact[name]) for name in artifact.files}
        arrays["intrinsics"] = arrays["intrinsics"].copy()
        arrays["intrinsics"][0, 0, 0] += 1.0
        np.savez_compressed(camera_path, **arrays)
    elif mutation == "staged_image":
        Image.new("RGB", (504, 504), "black").save(
            artifact_dir / "images/left.png"
        )
    else:
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["views"][0]["camera_group"] = "bottom"
        metadata_path.write_text(json.dumps(metadata))

    write_upper_camera_artifacts(
        manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )

    assert _camera_bundle_content(artifact_dir) == canonical


def test_upper_camera_resume_replaces_partial_bundle_and_removes_junk(
    tmp_path: Path,
) -> None:
    manifest, artifact_dir = _write_real_upper_camera_artifacts(tmp_path)
    canonical = _camera_bundle_content(artifact_dir)
    (artifact_dir / "metadata.json").unlink()
    (artifact_dir / "junk.txt").write_text("partial write")

    write_upper_camera_artifacts(
        manifest,
        tmp_path,
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )

    assert _camera_bundle_content(artifact_dir) == canonical


def test_upper_camera_atomic_repair_preserves_old_bundle_when_staging_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest, artifact_dir = _write_real_upper_camera_artifacts(tmp_path)
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["camera_mode"] = "calibrated GT"
    metadata_path.write_text(json.dumps(metadata))
    calls = []

    def fail_staging(destination: Path, **_: object) -> None:
        calls.append(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "partial").write_text("incomplete")
        raise OSError("simulated interrupted staging")

    monkeypatch.setattr(
        bytesize_data,
        "_write_selected_camera_set",
        fail_staging,
    )

    with pytest.raises(OSError, match="interrupted staging"):
        write_upper_camera_artifacts(
            manifest,
            tmp_path,
            shoe_aliases=CAMERA_DATASET_BY_SHOE,
        )

    assert len(calls) == 1
    assert (artifact_dir / "cameras.npz").is_file()
    assert json.loads(metadata_path.read_text())["camera_mode"] == "calibrated GT"
    assert not list(tmp_path.glob(".air_huarache_run_ultra_111449.*"))
