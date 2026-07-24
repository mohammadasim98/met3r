"""Depth Anything 3 adapter that preserves fixed/precomputed camera provenance."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .identity import image_set_id
from .provenance import provenance_fingerprint, runtime_environment

DEFAULT_DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"
DEFAULT_DA3_REVISION = "8615eefb62f2db4f8d6ebaa59160086981672829"


@dataclass(frozen=True)
class DepthArtifact:
    arrays_path: Path
    metadata_path: Path


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(extrinsics, dtype=np.float64)
    if value.ndim != 3 or value.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsics must be (N,3,4) or (N,4,4), got {value.shape}")
    if value.shape[-2:] == (4, 4):
        return value.copy()
    result = np.broadcast_to(np.eye(4), (value.shape[0], 4, 4)).copy()
    result[:, :3, :4] = value
    return result


def load_da3_model(
    model_id: str = DEFAULT_DA3_MODEL_ID,
    *,
    revision: str = DEFAULT_DA3_REVISION,
    repository_root: str | Path | None = None,
    device: str | None = None,
) -> Any:
    """Load DA3 lazily, optionally from a local source checkout."""

    if repository_root is not None:
        source = str(Path(repository_root).expanduser().resolve() / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise ImportError(
            "Depth Anything 3 is not installed; pass repository_root pointing to "
            "the Depth-Anything-3 checkout"
        ) from error
    model = DepthAnything3.from_pretrained(model_id, revision=revision)
    if device is not None:
        model = model.to(device)
    return model


def run_fixed_camera_depth(
    model: Any,
    images: Sequence[str | Path | np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: str | Path,
    *,
    model_id: str,
    model_revision: str = DEFAULT_DA3_REVISION,
    camera_provenance: str,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    pose_tolerance: float = 1e-5,
    intrinsics_tolerance: float = 1e-5,
    foreground_mask: np.ndarray | None = None,
    view_ids: Sequence[str] | None = None,
    camera_groups: Sequence[str] | None = None,
    analysis_tier: str | None = None,
    source_state: Mapping[str, object] | None = None,
    runtime_state: Mapping[str, object] | None = None,
) -> DepthArtifact:
    """Infer depth with fixed cameras and verify the aligned-output camera contract."""

    fixed_extrinsics = _homogeneous(extrinsics)
    content_id = image_set_id(images)
    fixed_intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if fixed_intrinsics.shape != (len(images), 3, 3):
        raise ValueError(
            f"intrinsics must have shape ({len(images)},3,3), got {fixed_intrinsics.shape}"
        )
    if fixed_extrinsics.shape[0] != len(images):
        raise ValueError("image and extrinsics counts do not match")
    if view_ids is not None and len(view_ids) != len(images):
        raise ValueError("view_ids must contain one value per image")
    if camera_groups is not None and len(camera_groups) != len(images):
        raise ValueError("camera_groups must contain one value per image")

    prediction = model.inference(
        image=[str(value) if isinstance(value, Path) else value for value in images],
        extrinsics=fixed_extrinsics,
        intrinsics=fixed_intrinsics,
        align_to_input_ext_scale=True,
        process_res=process_res,
        process_res_method=process_res_method,
        export_dir=None,
    )
    returned_extrinsics = _homogeneous(np.asarray(prediction.extrinsics))
    pose_error = float(np.max(np.abs(returned_extrinsics - fixed_extrinsics)))
    if pose_error > pose_tolerance:
        raise ValueError(
            "DA3 returned-camera contract mismatch for supplied poses: "
            f"max_abs_error={pose_error:.3e}, tolerance={pose_tolerance:.3e}"
        )
    returned_intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    if returned_intrinsics.shape != fixed_intrinsics.shape:
        raise ValueError(
            "DA3 returned intrinsics with an unexpected shape: "
            f"{returned_intrinsics.shape} != {fixed_intrinsics.shape}"
        )
    intrinsics_error = float(
        np.max(np.abs(returned_intrinsics - fixed_intrinsics))
    )
    if intrinsics_error > intrinsics_tolerance:
        raise ValueError(
            "DA3 returned-camera contract mismatch for supplied intrinsics: "
            f"max_abs_error={intrinsics_error:.3e}, "
            f"tolerance={intrinsics_tolerance:.3e}"
        )

    depth = np.asarray(prediction.depth)
    if foreground_mask is None:
        metric_mask = np.isfinite(depth) & (depth > 0)
        mask_source = "positive_finite_depth_fallback"
    else:
        metric_mask = np.asarray(foreground_mask, dtype=bool)
        if metric_mask.shape != depth.shape:
            raise ValueError(
                f"foreground_mask shape {metric_mask.shape} does not match depth {depth.shape}"
            )
        mask_source = "provided_foreground_mask"

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(runtime_state or runtime_environment())
    runtime_fingerprint = provenance_fingerprint(environment)
    source_revision = (
        str(source_state.get("revision", ""))
        if source_state is not None
        else ""
    )
    source_dirty = (
        bool(source_state.get("dirty", False))
        if source_state is not None
        else False
    )
    source_dirty_fingerprint = (
        str(source_state.get("dirty_fingerprint") or "")
        if source_state is not None
        else ""
    )
    arrays_path = destination / "depth_geometry.npz"
    metadata_path = destination / "depth_geometry.json"
    saved_arrays = dict(
        depth=depth,
        confidence=np.asarray(prediction.conf),
        mask=metric_mask,
        processed_images=np.asarray(prediction.processed_images),
        extrinsics=fixed_extrinsics,
        intrinsics=returned_intrinsics,
        meta__artifact_type=np.asarray("fixed_camera_depth"),
        meta__model_id=np.asarray(model_id),
        meta__model_revision=np.asarray(model_revision),
        meta__runtime_fingerprint=np.asarray(runtime_fingerprint),
        meta__source_revision=np.asarray(source_revision),
        meta__source_dirty=np.asarray(source_dirty),
        meta__source_dirty_fingerprint=np.asarray(source_dirty_fingerprint),
        meta__camera_provenance=np.asarray(camera_provenance),
        meta__image_set_id=np.asarray(content_id),
    )
    if view_ids is not None:
        saved_arrays["meta__view_ids"] = np.asarray(view_ids)
    if camera_groups is not None:
        saved_arrays["meta__camera_groups"] = np.asarray(camera_groups)
    if analysis_tier is not None:
        saved_arrays["meta__analysis_tier"] = np.asarray(analysis_tier)
    np.savez_compressed(arrays_path, **saved_arrays)
    metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "source_state": dict(source_state) if source_state is not None else None,
        "runtime_environment": environment,
        "runtime_fingerprint": runtime_fingerprint,
        "camera_mode": "fixed/precomputed (not calibrated GT)",
        "camera_provenance": camera_provenance,
        "process_res": process_res,
        "process_res_method": process_res_method,
        "pose_max_abs_error": pose_error,
        "pose_tolerance": pose_tolerance,
        "intrinsics_max_abs_error": intrinsics_error,
        "intrinsics_tolerance": intrinsics_tolerance,
        "mask_source": mask_source,
        "view_count": len(images),
        "image_set_id": content_id,
        "view_ids": list(view_ids) if view_ids is not None else None,
        "camera_groups": list(camera_groups) if camera_groups is not None else None,
        "analysis_tier": analysis_tier,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return DepthArtifact(arrays_path=arrays_path, metadata_path=metadata_path)
