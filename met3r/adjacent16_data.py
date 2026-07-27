"""Validated dense source records for the frozen adjacent 16-view experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .adjacent16_contract import Adjacent16Contract, load_adjacent16_contract
from .bytesize_data import validate_processed_image
from .deformation_audit import (
    DeformationQuality,
    _load_json,
    _quality,
    _require_file,
    _selected_attempt,
    _validate_metrics,
)


_VIEW_COUNT = 48


@dataclass(frozen=True)
class AdjacentViewRecord:
    shoe_id: str
    camera_dataset_id: str
    quartet_id: str
    view_id: str
    orbit_position: int
    source_path: Path
    source_mask_path: Path
    stress_path: Path | None
    stress_mask_path: Path | None
    deformation_json_path: Path | None
    metrics_json_path: Path | None
    task_json_path: Path | None
    processed_image_correlation: float
    camera_azimuth_deg: float
    source_sha256: str
    source_mask_sha256: str
    stress_sha256: str | None
    stress_mask_sha256: str | None
    quality: DeformationQuality | None


@dataclass(frozen=True)
class AdjacentShoeRecord:
    shoe_id: str
    camera_dataset_id: str
    views: tuple[AdjacentViewRecord, ...]


@dataclass(frozen=True)
class Adjacent16Manifest:
    reconstruction_root: Path
    stress_root: Path
    shoes: tuple[AdjacentShoeRecord, ...]


def camera_azimuths(extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Return circular camera-center-bearing azimuths in degrees.

    DA3's translation scale is not a stable orbit signal; the camera's world
    forward axis supplies the corresponding center bearing on the X/Z plane.
    """

    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("extrinsics must have shape (N,3,4) or (N,4,4)")
    if not np.isfinite(extrinsics).all():
        raise ValueError("extrinsics contain non-finite values")
    forward = np.einsum("nji,j->ni", extrinsics[:, :3, :3], np.array([0.0, 0.0, 1.0]))
    if not np.isfinite(forward).all():
        raise ValueError("camera forward vectors contain non-finite values")
    return np.mod(np.degrees(np.arctan2(forward[:, 2], forward[:, 0])), 360.0)


def pair_baseline(
    first: AdjacentViewRecord,
    second: AdjacentViewRecord,
) -> tuple[int, float]:
    """Return orbit-step distance and absolute angular separation."""

    steps = abs(first.orbit_position - second.orbit_position)
    steps = min(steps, _VIEW_COUNT - steps)
    angle = abs(first.camera_azimuth_deg - second.camera_azimuth_deg)
    return steps, min(angle, 360.0 - angle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order_from_azimuths(azimuths: np.ndarray, *, first_view_id: str) -> tuple[str, ...]:
    ordered = [str(index) for index in np.argsort(azimuths, kind="stable")]
    try:
        first = ordered.index(first_view_id)
    except ValueError as error:
        raise ValueError(f"camera azimuths do not contain view {first_view_id}") from error
    return tuple(ordered[first:] + ordered[:first])


def _require_exact_files(directory: Path, *, names: set[str], label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing {label} directory: {directory}")
    found = {path.name for path in directory.iterdir() if path.is_file()}
    if found != names:
        raise ValueError(
            f"{label} entries must be exactly 48 expected files: "
            f"missing={sorted(names - found)}, unexpected={sorted(found - names)}"
        )


def _source_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (1080, 1080):
            raise ValueError(f"source image must be RGB 1080x1080: {path}")


def _mask_image(path: Path, *, source_path: Path) -> None:
    with Image.open(source_path) as source, Image.open(path) as mask:
        if mask.size != source.size:
            raise ValueError(f"source and mask dimensions differ: {source_path}, {path}")


def _load_cameras(outputs: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"processed_images.npz", "intrinsics.npz", "extrinsics.npz"}
    _require_exact_files(outputs, names=required, label="camera output")
    arrays: list[np.ndarray] = []
    for filename, key in (
        ("processed_images.npz", "processed_images"),
        ("intrinsics.npz", "intrinsics"),
        ("extrinsics.npz", "extrinsics"),
    ):
        path = outputs / filename
        with np.load(path) as artifact:
            if set(artifact.files) != {key}:
                raise ValueError(f"camera artifact must contain only {key}: {path}")
            value = np.asarray(artifact[key])
        if not np.isfinite(value).all():
            raise ValueError(f"camera values must be finite: {path}")
        arrays.append(value)
    processed, intrinsics, extrinsics = arrays
    if processed.ndim != 4 or processed.shape[0] != _VIEW_COUNT or processed.shape[-1] < 3:
        raise ValueError("processed_images must contain 48 RGB images")
    if intrinsics.shape != (_VIEW_COUNT, 3, 3):
        raise ValueError("intrinsics must have shape (48,3,3)")
    if extrinsics.shape not in {(_VIEW_COUNT, 3, 4), (_VIEW_COUNT, 4, 4)}:
        raise ValueError("extrinsics must have shape (48,3,4) or (48,4,4)")
    return processed, intrinsics, extrinsics


def _stress_record(
    root: Path,
    *,
    shoe_id: str,
    quartet_id: str,
    view_id: str,
    require_stress: bool,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, str | None, str | None, DeformationQuality | None]:
    deformation_root = root / shoe_id / quartet_id / f"view_{view_id}" / "deformation"
    if not deformation_root.exists() and not require_stress:
        return (None,) * 8
    stress_path = _require_file(deformation_root / "deformed.png", label="stress image")
    stress_mask_path = _require_file(deformation_root / "deformed_mask.png", label="stress mask")
    deformation_json_path = _require_file(deformation_root / "deformation.json", label="stress deformation metadata")
    metrics_json_path = _require_file(deformation_root / "metrics.json", label="stress deformation metrics")
    task_json_path = _require_file(deformation_root / "task.json", label="stress deformation task")
    _source_image(stress_path)
    _mask_image(stress_mask_path, source_path=stress_path)
    deformation = _load_json(deformation_json_path, label="stress deformation metadata")
    task = _load_json(task_json_path, label="stress deformation task")
    selected = _selected_attempt(deformation)
    _validate_metrics(_load_json(metrics_json_path, label="stress deformation metrics"), deformation, selected)
    return (
        stress_path,
        stress_mask_path,
        deformation_json_path,
        metrics_json_path,
        task_json_path,
        _sha256(stress_path),
        _sha256(stress_mask_path),
        _quality(deformation, task),
    )


def build_adjacent16_manifest(
    contract: Adjacent16Contract,
    *,
    require_stress: bool,
) -> Adjacent16Manifest:
    """Validate all 48 source records, then retain the frozen selected 16."""

    root = contract.reconstruction_root.expanduser().resolve()
    stress_root = contract.stress_root.expanduser().resolve()
    source_root = root / "sample_result"
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing adjacent16 source root: {source_root}")
    expected_images = {f"{index}.jpg" for index in range(_VIEW_COUNT)}
    expected_masks = {f"{index}.jpg" for index in range(1, _VIEW_COUNT + 1)}
    expected_shoes = contract.raw["shoes"]
    if not isinstance(expected_shoes, Mapping):
        raise ValueError("adjacent16 contract shoes must be a mapping")
    if {path.name for path in source_root.iterdir() if path.is_dir()} != set(expected_shoes.values()):
        raise ValueError("source display-name shoes do not match the frozen contract")

    orbit_by_view = {view_id: position for position, view_id in enumerate(contract.camera_order)}
    selected = {
        view_id: quartet.quartet_id
        for quartet in contract.quartets
        for view_id in quartet.view_ids
    }
    shoes = []
    for shoe_id, camera_dataset_id in expected_shoes.items():
        if not isinstance(shoe_id, str) or not isinstance(camera_dataset_id, str):
            raise ValueError("adjacent16 contract shoe records must be strings")
        group = source_root / camera_dataset_id / "upper"
        inputs = group / "glb_masked_white_bg" / "masked_inputs"
        masks = group / "mask"
        outputs = group / "glb_masked_white_bg" / "outputs"
        _require_exact_files(inputs, names=expected_images, label="source image")
        _require_exact_files(masks, names=expected_masks, label="source mask")
        processed, _intrinsics, extrinsics = _load_cameras(outputs)
        azimuths = camera_azimuths(extrinsics)
        derived_order = _order_from_azimuths(
            azimuths, first_view_id=contract.camera_order[0]
        )
        if derived_order != contract.camera_order:
            raise ValueError("derived camera azimuth order does not match the frozen contract")
        ordered_gaps = np.diff(np.r_[np.sort(azimuths), np.sort(azimuths)[0] + 360.0])
        if not np.all((ordered_gaps >= 5.0) & (ordered_gaps <= 10.0)):
            raise ValueError("adjacent camera azimuth gaps must be within 5 to 10 degrees")

        all_records: dict[str, dict[str, object]] = {}
        for view_id in contract.camera_order:
            index = int(view_id)
            source_path = inputs / f"{index}.jpg"
            source_mask_path = masks / f"{index + 1}.jpg"
            _source_image(source_path)
            _mask_image(source_mask_path, source_path=source_path)
            all_records[view_id] = {
                "source_path": source_path,
                "source_mask_path": source_mask_path,
                "processed_image_correlation": validate_processed_image(
                    source_path, processed, camera_index=index, min_correlation=0.99
                ),
                "camera_azimuth_deg": float(azimuths[index]),
                "source_sha256": _sha256(source_path),
                "source_mask_sha256": _sha256(source_mask_path),
            }

        views = []
        for quartet in contract.quartets:
            for view_id in quartet.view_ids:
                record = all_records[view_id]
                stress = _stress_record(
                    stress_root, shoe_id=shoe_id, quartet_id=quartet.quartet_id,
                    view_id=view_id, require_stress=require_stress
                )
                views.append(
                    AdjacentViewRecord(
                        shoe_id=shoe_id,
                        camera_dataset_id=camera_dataset_id,
                        quartet_id=quartet.quartet_id,
                        view_id=view_id,
                        orbit_position=orbit_by_view[view_id],
                        source_path=record["source_path"],  # type: ignore[arg-type]
                        source_mask_path=record["source_mask_path"],  # type: ignore[arg-type]
                        stress_path=stress[0], stress_mask_path=stress[1], deformation_json_path=stress[2],
                        metrics_json_path=stress[3], task_json_path=stress[4],
                        processed_image_correlation=record["processed_image_correlation"],  # type: ignore[arg-type]
                        camera_azimuth_deg=record["camera_azimuth_deg"],  # type: ignore[arg-type]
                        source_sha256=record["source_sha256"],  # type: ignore[arg-type]
                        source_mask_sha256=record["source_mask_sha256"],  # type: ignore[arg-type]
                        stress_sha256=stress[5], stress_mask_sha256=stress[6], quality=stress[7],
                    )
                )
        shoes.append(AdjacentShoeRecord(shoe_id, camera_dataset_id, tuple(views)))
    return Adjacent16Manifest(root, stress_root, tuple(shoes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--clean-only", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = build_adjacent16_manifest(
        load_adjacent16_contract(arguments.contract), require_stress=not arguments.clean_only
    )
    print(json.dumps({
        "all_source_views": len(manifest.shoes) * _VIEW_COUNT,
        "selected_views": sum(len(shoe.views) for shoe in manifest.shoes),
        "selected_pair_units": len(manifest.shoes) * len(manifest.shoes[0].views) * 3 // 2,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
