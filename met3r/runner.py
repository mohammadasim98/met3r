"""Artifact-oriented runners for resumable feature extraction and metric scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .experiment import ViewGeometry, score_multiscale_factorial
from .features import (
    BACKBONE_SPECS,
    MultiScaleFeatureExtractor,
    NAF_REVISION,
    build_naf_guides,
    load_feature_backbone,
    load_naf_upsampler,
)
from .known_camera import KnownCameraConfig
from .known_camera import score_geometry_sensitivity_pair
from .geometry import confidence_percentile_mask, resize_camera_geometry
from .identity import image_set_id
from .provenance import provenance_fingerprint, runtime_environment
from .results import flatten_factorial_result


@dataclass(frozen=True)
class FeatureArtifact:
    arrays_path: Path
    metadata_path: Path


def extract_feature_artifact(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    backbone_name: str,
    input_size: int = 592,
    bilinear_scales: Sequence[int] = (),
    naf_scales: Sequence[int] = (),
    naf_guide_ablations: bool = True,
    device: str = "cuda",
    view_ids: Sequence[str] | None = None,
    camera_groups: Sequence[str] | None = None,
    camera_provenance: str | None = None,
    analysis_tier: str | None = None,
) -> FeatureArtifact:
    """Extract pinned DINO-family features and write all resolutions to one NPZ."""

    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"unknown backbone {backbone_name!r}")
    spec = BACKBONE_SPECS[backbone_name]
    if input_size % spec.patch_size:
        raise ValueError(
            f"input_size {input_size} must be divisible by patch size {spec.patch_size}"
        )
    arrays = []
    for path in image_paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(
                (input_size, input_size), resample=Image.Resampling.LANCZOS
            )
            arrays.append(np.asarray(resized, dtype=np.float32) / 255.0)
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
    content_id = image_set_id(image_paths)
    if view_ids is not None and len(view_ids) != len(image_paths):
        raise ValueError("view_ids must contain one value per image")
    if camera_groups is not None and len(camera_groups) != len(image_paths):
        raise ValueError("camera_groups must contain one value per image")
    backbone = load_feature_backbone(backbone_name).to(device)
    naf = load_naf_upsampler(device=device).to(device) if naf_scales else None
    extractor = MultiScaleFeatureExtractor(backbone, naf_upsampler=naf).to(device)
    guides = None
    if naf_scales:
        guides = (
            build_naf_guides(images)
            if naf_guide_ablations
            else {"normal": images}
        )
    features = extractor.extract(
        images,
        bilinear_scales=bilinear_scales,
        naf_scales=naf_scales,
        naf_guides=guides,
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment()
    runtime_fingerprint = provenance_fingerprint(environment)
    arrays_path = destination / "features.npz"
    metadata_path = destination / "features.json"
    saved_arrays: dict[str, np.ndarray] = {
        f"feature__{name}": tensor.detach().float().cpu().numpy()
        for name, tensor in features.items()
    }
    saved_arrays.update(
        {
            "meta__artifact_type": np.asarray("dino_features"),
            "meta__backbone_name": np.asarray(backbone_name),
            "meta__backbone_revision": np.asarray(spec.revision or ""),
            "meta__input_size": np.asarray(input_size),
            "meta__image_set_id": np.asarray(content_id),
            "meta__bilinear_scales": np.asarray(bilinear_scales, dtype=np.int64),
            "meta__naf_scales": np.asarray(naf_scales, dtype=np.int64),
            "meta__naf_revision": np.asarray(NAF_REVISION if naf_scales else ""),
            "meta__naf_guide_ablations": np.asarray(naf_guide_ablations),
            "meta__runtime_fingerprint": np.asarray(runtime_fingerprint),
        }
    )
    if view_ids is not None:
        saved_arrays["meta__view_ids"] = np.asarray(view_ids)
    if camera_groups is not None:
        saved_arrays["meta__camera_groups"] = np.asarray(camera_groups)
    if camera_provenance is not None:
        saved_arrays["meta__camera_provenance"] = np.asarray(camera_provenance)
    if analysis_tier is not None:
        saved_arrays["meta__analysis_tier"] = np.asarray(analysis_tier)
    np.savez_compressed(arrays_path, **saved_arrays)
    native_shape = list(features["native"].shape[-2:])
    metadata_path.write_text(
        json.dumps(
            {
                "backbone_name": backbone_name,
                "backbone": {
                    "family": spec.family,
                    "source": spec.source,
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "patch_size": spec.patch_size,
                },
                "input_size": input_size,
                "native_resolution": native_shape,
                "bilinear_scales": list(bilinear_scales),
                "naf_scales": list(naf_scales),
                "naf_revision": NAF_REVISION if naf_scales else None,
                "naf_guide_ablations": naf_guide_ablations,
                "image_paths": [str(Path(path).resolve()) for path in image_paths],
                "image_set_id": content_id,
                "view_ids": list(view_ids) if view_ids is not None else None,
                "camera_groups": (
                    list(camera_groups) if camera_groups is not None else None
                ),
                "camera_provenance": camera_provenance,
                "analysis_tier": analysis_tier,
                "runtime_environment": environment,
                "runtime_fingerprint": runtime_fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FeatureArtifact(arrays_path=arrays_path, metadata_path=metadata_path)


def _load_features(path: str | Path, device: str) -> dict[str, torch.Tensor]:
    with np.load(path) as artifact:
        features = {
            key.removeprefix("feature__"): torch.as_tensor(
                artifact[key], device=device
            ).float()
            for key in artifact.files
            if key.startswith("feature__")
        }
    if not features:
        raise ValueError(f"no feature__ arrays found in {path}")
    return features


def _metadata_value(value: np.ndarray) -> object:
    converted = value.tolist()
    return converted


def _load_npz_metadata(path: str | Path) -> dict[str, object]:
    with np.load(path) as artifact:
        return {
            key.removeprefix("meta__"): _metadata_value(np.asarray(artifact[key]))
            for key in artifact.files
            if key.startswith("meta__")
        }


def _assert_matching_metadata(
    first: Mapping[str, object],
    second: Mapping[str, object],
    *,
    fields: Sequence[str],
    label: str,
) -> None:
    for field in fields:
        if field in first and field in second and first[field] != second[field]:
            raise ValueError(
                f"{label} {field} mismatch: {first[field]!r} != {second[field]!r}"
            )


def _require_metadata(
    metadata: Mapping[str, object],
    *,
    fields: Sequence[str],
    label: str,
) -> None:
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(f"{label} artifact is missing metadata: {missing}")


def _validate_indices(count: int, first_index: int, second_index: int) -> None:
    if first_index == second_index:
        raise ValueError("first_index and second_index must identify different views")
    for index in (first_index, second_index):
        if not 0 <= index < count:
            raise IndexError(f"view index {index} is outside artifact count {count}")


def _resolved_metadata(
    caller: Mapping[str, object] | None,
    feature_metadata: Mapping[str, object],
    geometry_metadata: Mapping[str, object],
    *,
    first_index: int,
    second_index: int,
) -> dict[str, object]:
    result = dict(caller or {})
    view_ids = geometry_metadata.get("view_ids", feature_metadata.get("view_ids"))
    camera_groups = geometry_metadata.get("camera_groups")
    if isinstance(view_ids, list):
        result["first_view"] = view_ids[first_index]
        result["second_view"] = view_ids[second_index]
        result["pair_id"] = f"{view_ids[first_index]}__{view_ids[second_index]}"
    else:
        result["pair_id"] = f"index_{first_index}__index_{second_index}"
    if isinstance(camera_groups, list):
        first_group = camera_groups[first_index]
        second_group = camera_groups[second_index]
        result["pair_scope"] = (
            "intra_group" if first_group == second_group else "cross_group"
        )
        result["camera_group"] = (
            first_group if first_group == second_group else "cross_group"
        )
    for field in (
        "analysis_tier",
        "camera_provenance",
    ):
        if field in geometry_metadata:
            result[field] = geometry_metadata[field]
    for field in ("backbone_name", "backbone_revision", "input_size"):
        if field in feature_metadata:
            result[field] = feature_metadata[field]
    return result


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"invalid extrinsics shape {extrinsics.shape}")
    result = np.broadcast_to(np.eye(4), (*extrinsics.shape[:-2], 4, 4)).copy()
    result[..., :3, :4] = extrinsics
    return result


def _load_geometry(path: str | Path, device: str) -> ViewGeometry:
    with np.load(path) as artifact:
        depth = np.asarray(artifact["depth"])
        mask = (
            np.asarray(artifact["mask"], dtype=bool)
            if "mask" in artifact
            else np.isfinite(depth) & (depth > 0)
        )
        confidence = (
            torch.as_tensor(np.asarray(artifact["confidence"]), device=device).float()
            if "confidence" in artifact
            else None
        )
        return ViewGeometry(
            depth=torch.as_tensor(depth, device=device).float(),
            mask=torch.as_tensor(mask, device=device),
            confidence=confidence,
            intrinsics=torch.as_tensor(
                np.asarray(artifact["intrinsics"]), device=device
            ).float(),
            extrinsics_w2c=torch.as_tensor(
                _homogeneous(np.asarray(artifact["extrinsics"])), device=device
            ).float(),
        )


def _select_geometry(geometry: ViewGeometry, index: int) -> ViewGeometry:
    return ViewGeometry(
        depth=geometry.depth[index : index + 1],
        mask=geometry.mask[index : index + 1],
        confidence=(
            geometry.confidence[index : index + 1]
            if geometry.confidence is not None
            else None
        ),
        intrinsics=geometry.intrinsics[index : index + 1],
        extrinsics_w2c=geometry.extrinsics_w2c[index : index + 1],
    )


def score_feature_geometry_artifacts(
    clean_features_path: str | Path,
    changed_features_path: str | Path,
    base_geometry_path: str | Path,
    adaptive_geometry_path: str | Path,
    *,
    first_index: int,
    second_index: int,
    metadata: Mapping[str, object] | None = None,
    device: str = "cpu",
    config: KnownCameraConfig | None = None,
    confidence_percentile: float = 10.0,
    strict_metadata: bool = False,
) -> list[dict[str, object]]:
    """Score one indexed pair across every feature resolution in saved artifacts."""

    clean = _load_features(clean_features_path, device)
    changed = _load_features(changed_features_path, device)
    clean_metadata = _load_npz_metadata(clean_features_path)
    changed_metadata = _load_npz_metadata(changed_features_path)
    _assert_matching_metadata(
        clean_metadata,
        changed_metadata,
        fields=(
            "artifact_type",
            "backbone_name",
            "backbone_revision",
            "input_size",
            "view_ids",
            "camera_groups",
            "camera_provenance",
            "analysis_tier",
            "bilinear_scales",
            "naf_scales",
            "naf_revision",
            "naf_guide_ablations",
            "runtime_fingerprint",
        ),
        label="clean/changed feature",
    )
    base = _load_geometry(base_geometry_path, device)
    adaptive = _load_geometry(adaptive_geometry_path, device)
    base_metadata = _load_npz_metadata(base_geometry_path)
    adaptive_metadata = _load_npz_metadata(adaptive_geometry_path)
    if strict_metadata:
        _require_metadata(
            clean_metadata,
            fields=(
                "backbone_name",
                "artifact_type",
                "backbone_revision",
                "input_size",
                "view_ids",
                "camera_groups",
                "camera_provenance",
                "analysis_tier",
                "image_set_id",
                "bilinear_scales",
                "naf_scales",
                "naf_revision",
                "naf_guide_ablations",
                "runtime_fingerprint",
            ),
            label="feature",
        )
        _require_metadata(
            changed_metadata,
            fields=tuple(clean_metadata.keys()),
            label="changed feature",
        )
        _require_metadata(
            base_metadata,
            fields=(
                "artifact_type",
                "view_ids",
                "camera_groups",
                "camera_provenance",
                "analysis_tier",
                "image_set_id",
                "model_id",
                "model_revision",
                "runtime_fingerprint",
                "source_revision",
                "source_dirty",
                "source_dirty_fingerprint",
            ),
            label="geometry",
        )
        _require_metadata(
            adaptive_metadata,
            fields=tuple(base_metadata.keys()),
            label="adaptive geometry",
        )
    _assert_matching_metadata(
        base_metadata,
        adaptive_metadata,
        fields=(
            "artifact_type",
            "view_ids",
            "camera_groups",
            "camera_provenance",
            "analysis_tier",
            "model_id",
            "model_revision",
            "runtime_fingerprint",
            "source_revision",
            "source_dirty",
            "source_dirty_fingerprint",
        ),
        label="base/adaptive geometry",
    )
    _assert_matching_metadata(
        clean_metadata,
        base_metadata,
        fields=(
            "view_ids",
            "camera_groups",
            "camera_provenance",
            "analysis_tier",
        ),
        label="feature/geometry",
    )
    _assert_matching_metadata(
        clean_metadata,
        base_metadata,
        fields=("image_set_id",),
        label="clean-feature/base-geometry content",
    )
    _assert_matching_metadata(
        changed_metadata,
        adaptive_metadata,
        fields=("image_set_id",),
        label="changed-feature/adaptive-geometry content",
    )
    count = next(iter(clean.values())).shape[0]
    if any(value.shape[0] != count for value in (*clean.values(), *changed.values())):
        raise ValueError("all feature arrays must have the same view count")
    if base.depth.shape[0] != count or adaptive.depth.shape[0] != count:
        raise ValueError("feature and geometry view counts do not match")
    _validate_indices(count, first_index, second_index)

    first_clean = {key: value[first_index : first_index + 1] for key, value in clean.items()}
    second_clean = {key: value[second_index : second_index + 1] for key, value in clean.items()}
    first_changed = {
        key: value[first_index : first_index + 1] for key, value in changed.items()
    }
    second_changed = {
        key: value[second_index : second_index + 1] for key, value in changed.items()
    }
    results = score_multiscale_factorial(
        first_clean,
        second_clean,
        first_changed,
        second_changed,
        _select_geometry(base, first_index),
        _select_geometry(base, second_index),
        _select_geometry(adaptive, first_index),
        _select_geometry(adaptive, second_index),
        config=config,
        confidence_percentile=confidence_percentile,
    )
    shared = _resolved_metadata(
        metadata,
        clean_metadata,
        base_metadata,
        first_index=first_index,
        second_index=second_index,
    )
    return [
        {
            **shared,
            "first_index": first_index,
            "second_index": second_index,
            "resolution": resolution,
            **flatten_factorial_result(result),
        }
        for resolution, result in results.items()
    ]


def _filtered_mask(
    geometry: ViewGeometry,
    confidence_percentile: float,
) -> torch.Tensor:
    if geometry.confidence is None:
        return geometry.mask
    return confidence_percentile_mask(
        geometry.mask,
        geometry.confidence,
        percentile=confidence_percentile,
    )[0]


def score_geometry_sensitivity_artifacts(
    features_path: str | Path,
    base_geometry_path: str | Path,
    perturbed_geometry_path: str | Path,
    *,
    first_index: int,
    second_index: int,
    metadata: Mapping[str, object] | None = None,
    device: str = "cpu",
    config: KnownCameraConfig | None = None,
    confidence_percentile: float = 10.0,
    strict_metadata: bool = False,
) -> list[dict[str, object]]:
    """Score explicit pose/depth perturbations without the fixed-pose factorial guard."""

    features = _load_features(features_path, device)
    feature_metadata = _load_npz_metadata(features_path)
    base = _load_geometry(base_geometry_path, device)
    perturbed = _load_geometry(perturbed_geometry_path, device)
    base_metadata = _load_npz_metadata(base_geometry_path)
    perturbed_metadata = _load_npz_metadata(perturbed_geometry_path)
    if strict_metadata:
        _require_metadata(
            feature_metadata,
            fields=(
                "artifact_type",
                "backbone_name",
                "backbone_revision",
                "input_size",
                "view_ids",
                "camera_groups",
                "camera_provenance",
                "analysis_tier",
                "image_set_id",
                "bilinear_scales",
                "naf_scales",
                "naf_revision",
                "naf_guide_ablations",
                "runtime_fingerprint",
            ),
            label="feature",
        )
        _require_metadata(
            base_metadata,
            fields=(
                "artifact_type",
                "view_ids",
                "camera_groups",
                "camera_provenance",
                "analysis_tier",
                "image_set_id",
                "model_id",
                "model_revision",
                "runtime_fingerprint",
                "source_revision",
                "source_dirty",
                "source_dirty_fingerprint",
            ),
            label="geometry",
        )
        _require_metadata(
            perturbed_metadata,
            fields=tuple(base_metadata.keys()),
            label="perturbed geometry",
        )
    _assert_matching_metadata(
        base_metadata,
        perturbed_metadata,
        fields=(
            "artifact_type",
            "view_ids",
            "camera_groups",
            "camera_provenance",
            "analysis_tier",
            "model_id",
            "model_revision",
            "runtime_fingerprint",
            "source_revision",
            "source_dirty",
            "source_dirty_fingerprint",
        ),
        label="base/perturbed geometry",
    )
    _assert_matching_metadata(
        feature_metadata,
        base_metadata,
        fields=(
            "view_ids",
            "camera_groups",
            "camera_provenance",
            "analysis_tier",
        ),
        label="feature/geometry",
    )
    _assert_matching_metadata(
        feature_metadata,
        base_metadata,
        fields=("image_set_id",),
        label="feature/geometry content",
    )
    count = next(iter(features.values())).shape[0]
    if any(value.shape[0] != count for value in features.values()):
        raise ValueError("all feature arrays must have the same view count")
    if base.depth.shape[0] != count or perturbed.depth.shape[0] != count:
        raise ValueError("feature and geometry view counts do not match")
    _validate_indices(count, first_index, second_index)
    first_base = _select_geometry(base, first_index)
    second_base = _select_geometry(base, second_index)
    first_perturbed = _select_geometry(perturbed, first_index)
    second_perturbed = _select_geometry(perturbed, second_index)
    first_mask = _filtered_mask(first_base, confidence_percentile) & _filtered_mask(
        first_perturbed, confidence_percentile
    )
    second_mask = _filtered_mask(second_base, confidence_percentile) & _filtered_mask(
        second_perturbed, confidence_percentile
    )
    shared = _resolved_metadata(
        metadata,
        feature_metadata,
        base_metadata,
        first_index=first_index,
        second_index=second_index,
    )
    rows = []
    for resolution, feature_tensor in features.items():
        first_features = feature_tensor[first_index : first_index + 1]
        second_features = feature_tensor[second_index : second_index + 1]
        output_size = tuple(first_features.shape[-2:])
        first_base_scaled = resize_camera_geometry(
            first_base.depth,
            first_mask,
            first_base.confidence,
            first_base.intrinsics,
            output_size=output_size,
        )
        second_base_scaled = resize_camera_geometry(
            second_base.depth,
            second_mask,
            second_base.confidence,
            second_base.intrinsics,
            output_size=output_size,
        )
        first_perturbed_scaled = resize_camera_geometry(
            first_perturbed.depth,
            first_mask,
            first_perturbed.confidence,
            first_perturbed.intrinsics,
            output_size=output_size,
        )
        second_perturbed_scaled = resize_camera_geometry(
            second_perturbed.depth,
            second_mask,
            second_perturbed.confidence,
            second_perturbed.intrinsics,
            output_size=output_size,
        )
        result = score_geometry_sensitivity_pair(
            first_features,
            second_features,
            first_base_scaled.depth,
            second_base_scaled.depth,
            first_perturbed_scaled.depth,
            second_perturbed_scaled.depth,
            first_base_scaled.intrinsics,
            second_base_scaled.intrinsics,
            first_perturbed_scaled.intrinsics,
            second_perturbed_scaled.intrinsics,
            first_base.extrinsics_w2c,
            second_base.extrinsics_w2c,
            first_perturbed.extrinsics_w2c,
            second_perturbed.extrinsics_w2c,
            first_mask=first_base_scaled.mask,
            second_mask=second_base_scaled.mask,
            config=config,
        )
        overlap = min(
            float(result.perturbed.first_to_second.overlap_fraction.item()),
            float(result.perturbed.second_to_first.overlap_fraction.item()),
        )
        rows.append(
            {
                **shared,
                "first_index": first_index,
                "second_index": second_index,
                "resolution": resolution,
                "base_score": float(result.baseline.score.item()),
                "perturbed_score": float(result.perturbed.score.item()),
                "score": float(result.perturbed.score.item()),
                "geometry_effect": float(result.geometry_effect.item()),
                "overlap": overlap,
                "abstained": bool(result.perturbed.abstained.item()),
            }
        )
    return rows
