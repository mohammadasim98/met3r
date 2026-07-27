"""Validated input records for the fixed four-view upper-shoe experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .bytesize_data import validate_processed_image
from .deformation_audit import (
    DeformationQuality,
    _load_json,
    _quality,
    _required_bool,
    _required_mapping,
    _required_nonempty_string,
    _require_file,
    _selected_attempt,
    _strict_json_equal,
    _validate_metrics,
)


EXPECTED_SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_nrg_dc6083_500",
    "air_max_90_surplus_cq7743_300",
    "air_tuned_max_cv6984_001",
)
UPPER_VIEWS = ("left", "back", "right", "front")
CAMERA_DATASET_BY_SHOE = {
    "air_huarache_run_ultra_111449": "AIR HUARACHE RUN ULTRA(111449)",
    "air_max_90_nrg_dc6083_500": "AIR MAX 90 NRG(DC6083-500)",
    "air_max_90_surplus_cq7743_300": "AIR MAX 90 SURPLUS(CQ7743-300)",
    "air_tuned_max_cv6984_001": "AIR TUNED MAX(CV6984-001)",
}

_UPPER_CAMERA_INDEX_BY_VIEW = {
    "left": 42,
    "back": 4,
    "right": 17,
    "front": 29,
}


@dataclass(frozen=True)
class ImageInfo:
    size: tuple[int, int]
    mode: str
    sha256: str


@dataclass(frozen=True)
class UpperViewRecord:
    shoe_id: str
    camera_dataset_id: str
    view_id: str
    source_path: Path
    source_mask_path: Path
    deformed_path: Path
    deformed_mask_path: Path
    deformation_json_path: Path
    metrics_json_path: Path
    task_json_path: Path
    source_image: ImageInfo
    deformed_image: ImageInfo
    processed_image_correlation: float
    quality: DeformationQuality


@dataclass(frozen=True)
class UpperShoeRecord:
    shoe_id: str
    views: tuple[UpperViewRecord, ...]


@dataclass(frozen=True)
class UpperManifest:
    reconstruction_root: Path
    shoes: tuple[UpperShoeRecord, ...]

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict_paths(self), indent=2),
            encoding="utf-8",
        )
        return destination


def asdict_paths(value: object) -> object:
    """Convert a record graph into JSON-compatible scalar and absolute-path values."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict_paths(asdict(value))
    if isinstance(value, Path):
        if not value.is_absolute():
            raise ValueError(f"manifest path must be absolute: {value}")
        return str(value)
    if isinstance(value, tuple):
        return [asdict_paths(item) for item in value]
    if isinstance(value, list):
        return [asdict_paths(item) for item in value]
    if isinstance(value, dict):
        return {str(key): asdict_paths(item) for key, item in value.items()}
    return value


def _image_info(path: Path, *, label: str) -> ImageInfo:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{label} must be RGB: {path}")
        size = image.size
    return ImageInfo(
        size=size,
        mode="RGB",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _validate_image_and_masks(
    source_path: Path,
    source_mask_path: Path,
    deformed_path: Path,
    deformed_mask_path: Path,
) -> tuple[ImageInfo, ImageInfo]:
    source = _image_info(source_path, label="source image")
    deformed = _image_info(deformed_path, label="deformed image")
    if source.size != (1080, 1080) or deformed.size != (1080, 1080):
        raise ValueError(
            "source and deformed images must be 1080x1080: "
            f"source={source.size}, deformed={deformed.size}"
        )
    with Image.open(source_mask_path) as source_mask, Image.open(deformed_mask_path) as deformed_mask:
        sizes = {source.size, deformed.size, source_mask.size, deformed_mask.size}
    if len(sizes) != 1:
        raise ValueError(
            "source, deformed, and mask images must have identical dimensions: "
            f"source={source.size}, deformed={deformed.size}, "
            f"source_mask={source_mask.size}, deformed_mask={deformed_mask.size}"
        )
    return source, deformed


def build_upper_manifest(reconstruction_root: str | Path) -> UpperManifest:
    """Discover the frozen upper-view deformation inputs and their DA3 records."""

    root = Path(reconstruction_root).expanduser().resolve()
    multiview_root = root / "six_view_experiment_result_precomputed" / "multiview"
    if not multiview_root.is_dir():
        raise FileNotFoundError(f"missing upper-view multiview root: {multiview_root}")
    discovered_shoes = {path.name for path in multiview_root.iterdir() if path.is_dir()}
    expected_shoes = set(EXPECTED_SHOES)
    if discovered_shoes != expected_shoes:
        raise ValueError(
            "upper-view shoe IDs do not match the frozen contract: "
            f"missing={sorted(expected_shoes - discovered_shoes)}, "
            f"unexpected={sorted(discovered_shoes - expected_shoes)}"
        )

    shoes = []
    for shoe_id in EXPECTED_SHOES:
        camera_dataset_id = CAMERA_DATASET_BY_SHOE[shoe_id]
        processed_path = _require_file(
            root / "custom_dataset" / camera_dataset_id / "upper" / "processed_images.npz",
            label=f"processed camera images for {shoe_id}",
        )
        with np.load(processed_path) as artifact:
            if "processed_images" not in artifact:
                raise ValueError(f"missing processed_images array: {processed_path}")
            processed_images = np.asarray(artifact["processed_images"])

        views = []
        for view_id in UPPER_VIEWS:
            deformation_root = multiview_root / shoe_id / view_id / "deformation"
            source_path = _require_file(deformation_root / "source.png", label="source image")
            source_mask_path = _require_file(
                deformation_root / "source_mask.png", label="source mask"
            )
            deformed_path = _require_file(deformation_root / "deformed.png", label="deformed image")
            deformed_mask_path = _require_file(
                deformation_root / "deformed_mask.png", label="deformed mask"
            )
            deformation_json_path = _require_file(
                deformation_root / "deformation.json", label="deformation metadata"
            )
            metrics_json_path = _require_file(
                deformation_root / "metrics.json", label="deformation metrics"
            )
            task_json_path = _require_file(deformation_root / "task.json", label="deformation task"
            )
            source_image, deformed_image = _validate_image_and_masks(
                source_path,
                source_mask_path,
                deformed_path,
                deformed_mask_path,
            )
            correlation = validate_processed_image(
                source_path,
                processed_images,
                camera_index=_UPPER_CAMERA_INDEX_BY_VIEW[view_id],
                min_correlation=0.99,
            )
            deformation = _load_json(deformation_json_path, label="deformation metadata")
            task = _load_json(task_json_path, label="deformation task")
            selected = _selected_attempt(deformation)
            _validate_metrics(
                _load_json(metrics_json_path, label="deformation metrics"),
                deformation,
                selected,
            )
            views.append(
                UpperViewRecord(
                    shoe_id=shoe_id,
                    camera_dataset_id=camera_dataset_id,
                    view_id=view_id,
                    source_path=source_path,
                    source_mask_path=source_mask_path,
                    deformed_path=deformed_path,
                    deformed_mask_path=deformed_mask_path,
                    deformation_json_path=deformation_json_path,
                    metrics_json_path=metrics_json_path,
                    task_json_path=task_json_path,
                    source_image=source_image,
                    deformed_image=deformed_image,
                    processed_image_correlation=correlation,
                    quality=_quality(deformation, task),
                )
            )
        shoes.append(UpperShoeRecord(shoe_id=shoe_id, views=tuple(views)))
    return UpperManifest(reconstruction_root=root, shoes=tuple(shoes))
