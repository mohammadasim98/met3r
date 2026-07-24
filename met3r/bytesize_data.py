"""ByteSize six-view dataset discovery and validated manifest records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .camera_metadata import CAMERA_MODE_LABEL

VIEW_SPECS: tuple[tuple[str, str, int], ...] = (
    ("left", "upper", 42),
    ("back", "upper", 4),
    ("right", "upper", 17),
    ("front", "upper", 29),
    ("bottom_oblique", "bottom", 0),
    ("upper_oblique", "bottom", 17),
)


@dataclass(frozen=True)
class ViewRecord:
    shoe_id: str
    view_id: str
    camera_group: str
    camera_index: int
    source_path: Path
    image_path: Path
    mask_path: Path
    processed_images_path: Path
    custom_extrinsics_path: Path
    raw_extrinsics_path: Path
    intrinsics_path: Path
    deformed_path: Path | None = None


@dataclass(frozen=True)
class ShoeRecord:
    shoe_id: str
    views: tuple[ViewRecord, ...]


@dataclass(frozen=True)
class DatasetManifest:
    reconstruction_root: Path
    source_image_size: tuple[int, int]
    processed_image_size: tuple[int, int]
    camera_provenance: str
    shoes: tuple[ShoeRecord, ...]

    def to_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))  # type: ignore[return-value]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class CameraCorrection:
    """Recovered bottom-camera orientation with explicit translation uncertainty."""

    corrected_extrinsics: np.ndarray
    bottom_to_upper_rotation: np.ndarray
    exported_transform: np.ndarray
    transform_residual: float
    translation_is_proxy: bool = True
    intra_group_provenance: str = "fixed_precomputed_da3"
    cross_group_provenance: str = (
        "fixed_precomputed_da3_icp_rotation_corrected_translation_proxy"
    )


def build_bytesize_manifest(
    reconstruction_root: str | Path,
    *,
    deformed_root: str | Path | None = None,
) -> DatasetManifest:
    """Discover shoes and build the fixed six-view experiment manifest."""

    root = Path(reconstruction_root).expanduser().resolve()
    custom_root = root / "custom_dataset"
    if not custom_root.is_dir():
        raise FileNotFoundError(f"missing ByteSize custom_dataset: {custom_root}")
    deformed_root_path = (
        Path(deformed_root).expanduser().resolve() if deformed_root is not None else None
    )
    shoes = []
    for shoe_dir in sorted(path for path in custom_root.iterdir() if path.is_dir()):
        shoe_id = shoe_dir.name
        if not all((shoe_dir / group).is_dir() for group in ("upper", "bottom")):
            continue
        views = []
        for view_id, group, camera_index in VIEW_SPECS:
            source_number = camera_index + 1
            custom_group = custom_root / shoe_id / group
            raw_group = root / "sample_result" / shoe_id / group / "glb_masked_white_bg"
            deformed_path = None
            if deformed_root_path is not None:
                candidate = deformed_root_path / shoe_id / view_id / "deformed.png"
                if candidate.is_file():
                    deformed_path = candidate
            view = ViewRecord(
                shoe_id=shoe_id,
                view_id=view_id,
                camera_group=group,
                camera_index=camera_index,
                source_path=root / "sample_data" / shoe_id / group / f"{source_number}.jpg",
                image_path=raw_group / "masked_inputs" / f"{camera_index}.jpg",
                mask_path=root
                / "sample_result"
                / shoe_id
                / group
                / "mask"
                / f"{source_number}.jpg",
                processed_images_path=custom_group / "processed_images.npz",
                custom_extrinsics_path=custom_group / "extrinsics.npz",
                raw_extrinsics_path=raw_group / "outputs" / "extrinsics.npz",
                intrinsics_path=custom_group / "intrinsics.npz",
                deformed_path=deformed_path,
            )
            _validate_view_paths(view)
            views.append(view)
        shoes.append(ShoeRecord(shoe_id=shoe_id, views=tuple(views)))
    if not shoes:
        raise ValueError(f"no complete ByteSize shoes found under {custom_root}")
    source_sizes = set()
    processed_sizes = set()
    processed_cache: dict[Path, tuple[int, int]] = {}
    for shoe in shoes:
        for view in shoe.views:
            with Image.open(view.source_path) as image:
                source_sizes.add(image.size)
            if view.processed_images_path not in processed_cache:
                with np.load(view.processed_images_path) as artifact:
                    processed = np.asarray(artifact["processed_images"])
                if processed.ndim != 4 or processed.shape[-1] < 4:
                    raise ValueError(
                        f"invalid processed_images array: {view.processed_images_path}"
                    )
                processed_cache[view.processed_images_path] = (
                    int(processed.shape[2]),
                    int(processed.shape[1]),
                )
            processed_sizes.add(processed_cache[view.processed_images_path])
    if len(source_sizes) != 1 or len(processed_sizes) != 1:
        raise ValueError(
            "ByteSize source/processed image sizes must be consistent; "
            f"source={sorted(source_sizes)}, processed={sorted(processed_sizes)}"
        )
    return DatasetManifest(
        reconstruction_root=root,
        source_image_size=next(iter(source_sizes)),
        processed_image_size=next(iter(processed_sizes)),
        camera_provenance="fixed_precomputed_da3_icp",
        shoes=tuple(shoes),
    )


def _validate_view_paths(view: ViewRecord) -> None:
    for field_name in (
        "source_path",
        "image_path",
        "mask_path",
        "processed_images_path",
        "custom_extrinsics_path",
        "raw_extrinsics_path",
        "intrinsics_path",
    ):
        path = getattr(view, field_name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {field_name} for {view.shoe_id}/{view.view_id}: {path}")


def _as_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsics must have shape (N,3,4) or (N,4,4), got {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.copy()
    result = np.broadcast_to(np.eye(4), (extrinsics.shape[0], 4, 4)).copy()
    result[:, :3, :4] = extrinsics
    return result


def _proper_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation


def recover_bottom_camera_alignment(
    upper_extrinsics: np.ndarray,
    bottom_raw_extrinsics: np.ndarray,
    bottom_custom_extrinsics: np.ndarray,
    *,
    transform_tolerance: float = 1e-5,
) -> CameraCorrection:
    """Recover the documented upper/bottom rotation from saved DA3 camera arrays.

    The notebook applied an ICP transform in GLB coordinates directly to raw DA3
    extrinsics.  Its rotation can be converted back because the first-camera
    axis transforms are known.  GLB's point-cloud centering translation was not
    saved, so the custom translation is retained only as an explicitly labelled
    proxy for cross-group sensitivity analysis.
    """

    upper = _as_homogeneous(upper_extrinsics)
    bottom_raw = _as_homogeneous(bottom_raw_extrinsics)
    bottom_custom = _as_homogeneous(bottom_custom_extrinsics)
    if bottom_raw.shape != bottom_custom.shape:
        raise ValueError("raw and custom bottom extrinsics must have identical shapes")
    if upper.shape[0] == 0 or bottom_raw.shape[0] == 0:
        raise ValueError("camera arrays must not be empty")

    exported_transforms = np.linalg.inv(bottom_raw) @ bottom_custom
    transform_residual = float(
        np.max(np.abs(exported_transforms - exported_transforms[:1]))
    )
    if transform_residual > transform_tolerance:
        raise ValueError(
            "saved notebook export transform is not constant across bottom views: "
            f"residual={transform_residual:.3e}, tolerance={transform_tolerance:.3e}"
        )

    axis_flip = np.diag([1.0, -1.0, -1.0])
    upper_export_rotation = axis_flip @ upper[0, :3, :3]
    bottom_export_rotation = axis_flip @ bottom_raw[0, :3, :3]
    bottom_to_upper = (
        upper_export_rotation.T
        @ exported_transforms[0, :3, :3]
        @ bottom_export_rotation
    )
    bottom_to_upper = _proper_rotation(bottom_to_upper)

    corrected = bottom_custom.copy()
    corrected[:, :3, :3] = bottom_raw[:, :3, :3] @ bottom_to_upper.T
    return CameraCorrection(
        corrected_extrinsics=corrected[:, :3, :4],
        bottom_to_upper_rotation=bottom_to_upper,
        exported_transform=exported_transforms[0],
        transform_residual=transform_residual,
    )


def validate_processed_image(
    image_path: str | Path,
    processed_images: np.ndarray,
    *,
    camera_index: int,
    min_correlation: float = 0.98,
) -> float:
    """Verify that a manifest image and its indexed DA3 tensor refer to the same view."""

    processed = np.asarray(processed_images)
    if processed.ndim != 4 or processed.shape[-1] < 3:
        raise ValueError(f"processed_images must have shape (N,H,W,C>=3), got {processed.shape}")
    if not 0 <= camera_index < processed.shape[0]:
        raise IndexError(f"camera_index {camera_index} outside {processed.shape[0]} images")

    target = processed[camera_index, ..., :3].astype(np.float64)
    with Image.open(image_path) as image:
        source = image.convert("RGB").resize(
            (target.shape[1], target.shape[0]),
            resample=Image.Resampling.BILINEAR,
        )
        source_array = np.asarray(source, dtype=np.float64)
    source_gray = source_array.mean(axis=-1).reshape(-1)
    target_gray = target.mean(axis=-1).reshape(-1)
    if np.std(source_gray) == 0 or np.std(target_gray) == 0:
        correlation = float(np.allclose(source_gray, target_gray))
    else:
        correlation = float(np.corrcoef(source_gray, target_gray)[0, 1])
    if not np.isfinite(correlation) or correlation < min_correlation:
        raise ValueError(
            f"processed-image grayscale correlation {correlation:.6f} is below "
            f"{min_correlation:.6f} for camera index {camera_index}"
        )
    return correlation


def _write_selected_camera_set(
    destination: Path,
    *,
    views: tuple[ViewRecord, ...],
    extrinsics_by_group: dict[str, np.ndarray],
    provenance: str,
    analysis_tier: str,
    translation_is_proxy: bool,
) -> None:
    arrays, staged_images = _selected_camera_content(
        views=views,
        extrinsics_by_group=extrinsics_by_group,
    )
    destination.mkdir(parents=True, exist_ok=True)
    image_dir = destination / "images"
    image_dir.mkdir(exist_ok=True)
    image_paths = []
    for view, staged_rgb in zip(views, staged_images):
        staged_path = image_dir / f"{view.view_id}.png"
        Image.fromarray(staged_rgb).save(staged_path)
        image_paths.append(str(staged_path))

    np.savez_compressed(destination / "cameras.npz", **arrays)
    (destination / "images.json").write_text(
        json.dumps(image_paths, indent=2),
        encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            _camera_metadata(
                views=views,
                provenance=provenance,
                analysis_tier=analysis_tier,
                translation_is_proxy=translation_is_proxy,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def _selected_camera_content(
    *,
    views: tuple[ViewRecord, ...],
    extrinsics_by_group: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    intrinsics = []
    extrinsics = []
    masks = []
    staged_images = []
    for view in views:
        with np.load(view.intrinsics_path) as artifact:
            intrinsics.append(np.asarray(artifact["intrinsics"])[view.camera_index])
        extrinsics.append(extrinsics_by_group[view.camera_group][view.camera_index])
        with np.load(view.processed_images_path) as artifact:
            processed = np.asarray(artifact["processed_images"])[view.camera_index]
        if processed.shape[-1] < 4:
            raise ValueError(
                f"processed image lacks alpha mask for {view.shoe_id}/{view.view_id}"
            )
        foreground = processed[..., 3] > 127
        masks.append(foreground)
        staged_rgb = np.where(
            foreground[..., None],
            processed[..., :3],
            np.asarray([255, 255, 255], dtype=processed.dtype),
        ).astype(np.uint8)
        staged_images.append(staged_rgb)
    return (
        {
            "intrinsics": np.stack(intrinsics),
            "extrinsics": np.stack(extrinsics),
            "mask": np.stack(masks),
            "view_ids": np.asarray([view.view_id for view in views]),
            "camera_groups": np.asarray([view.camera_group for view in views]),
        },
        tuple(staged_images),
    )


def _camera_metadata(
    *,
    views: tuple[ViewRecord, ...],
    provenance: str,
    analysis_tier: str,
    translation_is_proxy: bool,
) -> dict[str, object]:
    return {
        "camera_mode": CAMERA_MODE_LABEL,
        "camera_provenance": provenance,
        "analysis_tier": analysis_tier,
        "translation_is_proxy": translation_is_proxy,
        "views": [
            {
                "view_id": view.view_id,
                "camera_group": view.camera_group,
                "camera_index": view.camera_index,
            }
            for view in views
        ],
    }


def _selected_camera_set_matches(
    destination: Path,
    *,
    views: tuple[ViewRecord, ...],
    extrinsics_by_group: Mapping[str, np.ndarray],
    provenance: str,
    analysis_tier: str,
    translation_is_proxy: bool,
) -> bool:
    try:
        expected_arrays, expected_images = _selected_camera_content(
            views=views,
            extrinsics_by_group=extrinsics_by_group,
        )
        with np.load(destination / "cameras.npz") as artifact:
            if set(artifact.files) != set(expected_arrays):
                return False
            if any(
                not np.array_equal(np.asarray(artifact[name]), expected)
                for name, expected in expected_arrays.items()
            ):
                return False
        expected_paths = [
            str(destination / "images" / f"{view.view_id}.png")
            for view in views
        ]
        if json.loads(
            (destination / "images.json").read_text(encoding="utf-8")
        ) != expected_paths:
            return False
        if json.loads(
            (destination / "metadata.json").read_text(encoding="utf-8")
        ) != _camera_metadata(
            views=views,
            provenance=provenance,
            analysis_tier=analysis_tier,
            translation_is_proxy=translation_is_proxy,
        ):
            return False
        for path, expected in zip(expected_paths, expected_images):
            with Image.open(path) as image:
                actual = np.asarray(image.convert("RGB"))
            if not np.array_equal(actual, expected):
                return False
        return True
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def write_camera_artifacts(
    manifest: DatasetManifest,
    output_dir: str | Path,
) -> None:
    """Write primary per-group poses and a separately labelled cross-group proxy."""

    destination = Path(output_dir).expanduser().resolve()
    for shoe in manifest.shoes:
        group_extrinsics: dict[str, np.ndarray] = {}
        for group in ("upper", "bottom"):
            example = next(view for view in shoe.views if view.camera_group == group)
            with np.load(example.raw_extrinsics_path) as artifact:
                group_extrinsics[group] = _as_homogeneous(
                    np.asarray(artifact["extrinsics"])
                )
            group_views = tuple(
                view for view in shoe.views if view.camera_group == group
            )
            _write_selected_camera_set(
                destination / "primary" / shoe.shoe_id / group,
                views=group_views,
                extrinsics_by_group=group_extrinsics,
                provenance="fixed_precomputed_da3",
                analysis_tier="primary",
                translation_is_proxy=False,
            )

        upper = next(view for view in shoe.views if view.camera_group == "upper")
        bottom = next(view for view in shoe.views if view.camera_group == "bottom")
        with np.load(upper.raw_extrinsics_path) as artifact:
            upper_raw = np.asarray(artifact["extrinsics"])
        with np.load(bottom.raw_extrinsics_path) as artifact:
            bottom_raw = np.asarray(artifact["extrinsics"])
        with np.load(bottom.custom_extrinsics_path) as artifact:
            bottom_custom = np.asarray(artifact["extrinsics"])
        correction = recover_bottom_camera_alignment(
            upper_raw,
            bottom_raw,
            bottom_custom,
        )
        proxy_extrinsics = {
            "upper": _as_homogeneous(upper_raw),
            "bottom": _as_homogeneous(correction.corrected_extrinsics),
        }
        _write_selected_camera_set(
            destination / "sensitivity" / shoe.shoe_id,
            views=shoe.views,
            extrinsics_by_group=proxy_extrinsics,
            provenance=correction.cross_group_provenance,
            analysis_tier="sensitivity_only",
            translation_is_proxy=True,
        )


def write_upper_camera_artifacts(
    manifest: DatasetManifest,
    output_dir: str | Path,
    shoe_aliases: Mapping[str, str],
) -> None:
    """Write exactly one four-view upper camera artifact per deformation slug."""

    destination = Path(output_dir).expanduser().resolve()
    shoes_by_camera_id = {shoe.shoe_id: shoe for shoe in manifest.shoes}
    if len(shoe_aliases) != len(set(shoe_aliases.values())):
        raise ValueError("upper camera aliases must be one-to-one")
    for deformation_slug, camera_dataset_id in shoe_aliases.items():
        shoe = shoes_by_camera_id.get(camera_dataset_id)
        if shoe is None:
            raise ValueError(
                f"camera dataset {camera_dataset_id!r} for {deformation_slug!r} "
                "is absent from the ByteSize manifest"
            )
        upper_views = tuple(
            view for view in shoe.views if view.camera_group == "upper"
        )
        if tuple(view.view_id for view in upper_views) != (
            "left",
            "back",
            "right",
            "front",
        ):
            raise ValueError(
                f"camera dataset {camera_dataset_id!r} lacks the canonical upper views"
            )
        artifact_dir = destination / deformation_slug
        camera_path = artifact_dir / "cameras.npz"
        example = upper_views[0]
        with np.load(example.raw_extrinsics_path) as artifact:
            upper_extrinsics = _as_homogeneous(np.asarray(artifact["extrinsics"]))
        if camera_path.is_file():
            if _selected_camera_set_matches(
                artifact_dir,
                views=upper_views,
                extrinsics_by_group={"upper": upper_extrinsics},
                provenance="fixed_precomputed_da3",
                analysis_tier="primary",
                translation_is_proxy=False,
            ):
                continue
            raise ValueError(f"invalid existing upper camera artifact: {artifact_dir}")
        _write_selected_camera_set(
            artifact_dir,
            views=upper_views,
            extrinsics_by_group={"upper": upper_extrinsics},
            provenance="fixed_precomputed_da3",
            analysis_tier="primary",
            translation_is_proxy=False,
        )
