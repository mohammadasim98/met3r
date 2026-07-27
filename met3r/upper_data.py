"""Validated input records for the fixed four-view upper-shoe experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .bytesize_data import validate_processed_image


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


@dataclass(frozen=True)
class DeformationQuality:
    status: str
    selected_attempt_id: str
    semantic_gate: bool
    preservation_gate: bool
    camera_preserved: bool
    outsole_preserved: bool
    fit_relation: str
    task_status: str


def _required_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


_required_nonempty_string = _required_string


def _required_bool(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _required_mapping(
    payload: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _selected_attempt(payload: Mapping[str, object]) -> Mapping[str, object]:
    selected = _required_string(payload, "selected_attempt_id", "selected attempt ID")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("deformation attempts must be a list")
    matches = [
        item for item in attempts
        if isinstance(item, Mapping) and item.get("attempt_id") == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"selected attempt {selected!r} is not unique")
    return matches[0]


def _quality(
    deformation: Mapping[str, object], task: Mapping[str, object]
) -> DeformationQuality:
    status = _required_string(deformation, "status", "deformation status")
    selected_attempt_id = _required_string(deformation, "selected_attempt_id", "selected attempt ID")
    selected = _selected_attempt(deformation)
    semantic = _required_mapping(selected, "semantic", "selected semantic audit")
    preservation = _required_mapping(selected, "preservation", "selected preservation audit")
    comparison = _required_mapping(selected, "comparison", "selected comparison audit")
    fit_relation = _required_string(comparison, "relation", "fit relation")
    if fit_relation not in {"improved", "equivalent", "worsened"}:
        raise ValueError(f"fit relation is invalid: {fit_relation!r}")
    return DeformationQuality(
        status=status, selected_attempt_id=selected_attempt_id,
        semantic_gate=_required_bool(semantic, "passes_gate", "semantic gate"),
        preservation_gate=_required_bool(preservation, "passes_gate", "preservation gate"),
        camera_preserved=_required_bool(semantic, "camera_preserved", "camera preservation"),
        outsole_preserved=_required_bool(semantic, "outsole_preserved", "outsole preservation"),
        fit_relation=fit_relation,
        task_status=_required_string(task, "status", "task status"),
    )


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            _strict_json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _strict_json_equal(first, second) for first, second in zip(left, right)
        )
    return left == right


def _validate_metrics(
    metrics: Mapping[str, object],
    deformation: Mapping[str, object],
    selected: Mapping[str, object],
) -> None:
    selected_metrics = _required_mapping(selected, "metrics", "selected metrics")
    selected_comparison = _required_mapping(selected, "comparison", "selected comparison")
    selected_preservation = _required_mapping(selected, "preservation", "selected preservation")
    status = _required_string(deformation, "status", "deformation status")
    if not _strict_json_equal(_required_mapping(metrics, "after", "metrics after"), selected_metrics):
        raise ValueError("metrics after does not match the selected attempt")
    if not _strict_json_equal(_required_mapping(metrics, "comparison", "metrics comparison"), selected_comparison):
        raise ValueError("metrics comparison does not match the selected attempt")
    if not _strict_json_equal(_required_mapping(metrics, "preservation", "metrics preservation"), selected_preservation):
        raise ValueError("metrics preservation does not match the selected attempt")
    if _required_string(metrics, "verdict", "metrics verdict") != status:
        raise ValueError("metrics verdict does not match deformation status")


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


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
