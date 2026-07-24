"""Known-magnitude camera/depth perturbations for geometry sensitivity calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _axis_rotation(axis: str, radians: float) -> np.ndarray:
    cosine, sine = np.cos(radians), np.sin(radians)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=float)
    if axis == "y":
        return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=float)
    if axis == "z":
        return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float)
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def perturb_extrinsics(
    extrinsics_w2c: np.ndarray,
    *,
    view_index: int,
    rotation_degrees: float = 0.0,
    translation_fraction: float = 0.0,
    scene_scale: float | None = None,
    axis: str = "y",
    translation_axis: str = "x",
) -> np.ndarray:
    """Perturb one w2c pose while keeping rotation and translation controls distinct.

    The rotation left-multiplies the complete ``[R|t]`` transform, preserving
    camera center. Translation magnitude is a fraction of the median camera
    radius, so its units are comparable across shoes.
    """

    value = np.asarray(extrinsics_w2c)
    if value.ndim != 3 or value.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape (N,3,4) or (N,4,4)")
    if not 0 <= view_index < value.shape[0]:
        raise IndexError("view_index is out of range")
    result = value.copy()
    delta = _axis_rotation(axis, np.deg2rad(rotation_degrees)).astype(result.dtype)
    result[view_index, :3, :4] = delta @ result[view_index, :3, :4]
    if translation_fraction:
        scale = (
            float(scene_scale)
            if scene_scale is not None
            else estimate_camera_scene_scale(value)
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scene_scale must be positive for relative translation")
        translation_axis_index = {"x": 0, "y": 1, "z": 2}.get(translation_axis)
        if translation_axis_index is None:
            raise ValueError("translation_axis must be one of 'x', 'y', or 'z'")
        result[view_index, translation_axis_index, 3] += translation_fraction * scale
    return result


def estimate_camera_scene_scale(extrinsics_w2c: np.ndarray) -> float:
    """Estimate a robust camera-baseline radius used for relative translations."""

    value = np.asarray(extrinsics_w2c, dtype=np.float64)
    if value.ndim != 3 or value.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape (N,3,4) or (N,4,4)")
    if value.shape[-2:] == (3, 4):
        homogeneous = np.broadcast_to(np.eye(4), (value.shape[0], 4, 4)).copy()
        homogeneous[:, :3, :4] = value
    else:
        homogeneous = value
    centers = np.linalg.inv(homogeneous)[:, :3, 3]
    center = np.mean(centers, axis=0)
    radii = np.linalg.norm(centers - center, axis=1)
    positive = radii[radii > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError("cannot estimate scene scale from coincident camera centers")
    return float(np.median(positive))


def perturb_depth(
    depth: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    scale: float = 1.0,
    bias: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Perturb only foreground depths; noise_std is relative to median foreground depth."""

    value = np.asarray(depth)
    mask = np.asarray(foreground_mask, dtype=bool)
    if value.shape != mask.shape:
        raise ValueError("depth and foreground_mask must have identical shapes")
    if scale <= 0 or noise_std < 0:
        raise ValueError("scale must be positive and noise_std must be non-negative")
    result = value.copy()
    foreground = value[mask]
    if foreground.size == 0:
        return result
    median_depth = float(np.median(foreground))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std * median_depth, size=foreground.shape)
    result[mask] = np.maximum(foreground * scale + bias + noise, np.finfo(value.dtype).eps)
    return result


def write_geometry_calibration_artifacts(
    base_geometry_path: str | Path,
    output_dir: str | Path,
    *,
    view_index: int,
    rotation_degrees: Sequence[float] = (),
    translation_fractions: Sequence[float] = (),
    depth_scales: Sequence[float] = (),
    depth_noise_stds: Sequence[float] = (),
    seed: int = 42,
) -> Path:
    """Materialize one-factor-at-a-time geometry variants for score calibration."""

    with np.load(base_geometry_path) as source:
        base = {key: np.asarray(source[key]) for key in source.files}
    required = {"depth", "intrinsics", "extrinsics"}
    missing = required - set(base)
    if missing:
        raise ValueError(f"base geometry is missing arrays: {sorted(missing)}")
    mask = np.asarray(
        base.get("mask", np.isfinite(base["depth"]) & (base["depth"] > 0)),
        dtype=bool,
    )
    if not 0 <= view_index < base["depth"].shape[0]:
        raise IndexError("view_index is out of range")
    scene_scale = estimate_camera_scene_scale(base["extrinsics"])
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def write_variant(kind: str, magnitude: float, arrays: dict[str, np.ndarray]) -> None:
        condition_id = f"{kind}_{magnitude:.6g}".replace("-", "neg").replace(".", "p")
        condition_dir = destination / condition_id
        condition_dir.mkdir(exist_ok=True)
        path = condition_dir / "geometry.npz"
        np.savez_compressed(path, **arrays)
        records.append(
            {
                "condition_id": condition_id,
                "kind": kind,
                "magnitude": float(magnitude),
                "path": str(path),
                "view_index": view_index,
            }
        )

    for magnitude in rotation_degrees:
        arrays = dict(base)
        arrays["extrinsics"] = perturb_extrinsics(
            base["extrinsics"],
            view_index=view_index,
            rotation_degrees=float(magnitude),
            scene_scale=scene_scale,
        )
        write_variant("rotation", float(magnitude), arrays)
    for magnitude in translation_fractions:
        arrays = dict(base)
        arrays["extrinsics"] = perturb_extrinsics(
            base["extrinsics"],
            view_index=view_index,
            translation_fraction=float(magnitude),
            scene_scale=scene_scale,
        )
        write_variant("relative_translation", float(magnitude), arrays)
    for magnitude in depth_scales:
        arrays = dict(base)
        changed_depth = base["depth"].copy()
        changed_depth[view_index : view_index + 1] = perturb_depth(
            base["depth"][view_index : view_index + 1],
            mask[view_index : view_index + 1],
            scale=float(magnitude),
            seed=seed,
        )
        arrays["depth"] = changed_depth
        write_variant("depth_scale", float(magnitude), arrays)
    for magnitude in depth_noise_stds:
        arrays = dict(base)
        changed_depth = base["depth"].copy()
        changed_depth[view_index : view_index + 1] = perturb_depth(
            base["depth"][view_index : view_index + 1],
            mask[view_index : view_index + 1],
            noise_std=float(magnitude),
            seed=seed,
        )
        arrays["depth"] = changed_depth
        write_variant("relative_depth_noise", float(magnitude), arrays)

    manifest_path = destination / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_geometry": str(Path(base_geometry_path).resolve()),
                "scene_scale": scene_scale,
                "translation_units": "fraction_of_median_camera_radius",
                "artifacts": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
