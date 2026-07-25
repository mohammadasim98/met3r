"""Long-lived upper-view feature worker with atomic metric persistence."""

from __future__ import annotations

import errno
import gc
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import traceback
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor

from .camera_metadata import CAMERA_MODE_LABEL
from .experiment import ViewGeometry, score_multiscale_factorial
from .features import (
    BACKBONE_SPECS,
    NAF_REVISION,
    MultiScaleFeatureExtractor,
    load_feature_backbone,
    load_naf_upsampler,
)
from .geometry import confidence_percentile_mask, resize_camera_geometry
from .job_registry import ClaimedJob, JobRegistry, WorkerSummary, sha256_file
from .known_camera import (
    GeometrySensitivityResult,
    KnownCameraConfig,
    PairMetricResult,
    score_geometry_sensitivity_pair,
)
from .pca_features import PCABasis, fit_pca_basis, render_pca_png
from .results import flatten_factorial_result
from .streaming_features import iter_feature_variants
from .upper_data import UPPER_VIEWS
from .upper_design import FeatureVariant
from .visualization import save_direction_diagnostics


_STALE_SECONDS = 21_600
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.EBUSY,
        errno.EINTR,
        errno.ETIMEDOUT,
    }
)
_QUALITY_FIELDS = (
    "semantic_gate",
    "preservation_gate",
    "camera_preserved",
    "outsole_preserved",
    "fit_relation",
)
_MIDDLE_DIAGNOSTIC_SEVERITY = {
    "translation": 0.12,
    "thin_plate_warp": 0.50,
    "exposure": 0.50,
    "blur": 0.70,
    "jpeg": 0.75,
}


@dataclass(frozen=True)
class _RunContract:
    root: Path
    config: Mapping[str, object]
    config_sha256: str
    scope: str
    metric_config: KnownCameraConfig
    confidence_percentile: float
    backbone_revision: str


@dataclass(frozen=True)
class _StateRecord:
    state_id: str
    track: str
    shoe_id: str
    changed_view: str | None
    corruption: str
    severity: float
    state_dir: Path


@dataclass(frozen=True)
class _ImageBatch:
    tensor: Tensor
    arrays: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _InstalledOutput:
    destination: Path
    backup: Path | None
    empty_marker: Path | None


@dataclass
class _CleanFeatureCache:
    key: tuple[str, str, int] | None = None
    images: _ImageBatch | None = None
    native: Tensor | None = None

    def release(self) -> None:
        self.key = None
        self.images = None
        self.native = None


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _path_within(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be inside {root}: {resolved}") from error
    return resolved


def _load_run_contract(root: Path, backbone: str) -> _RunContract:
    config_path = root / "config" / "frozen_config.json"
    content = config_path.read_bytes()
    config_sha256 = hashlib.sha256(content).hexdigest()
    config = json.loads(content)
    if not isinstance(config, Mapping):
        raise ValueError("frozen config must contain an object")
    run_manifest = _json_object(root / "run_manifest.json")
    if run_manifest.get("config_sha256") != config_sha256:
        raise ValueError("run manifest config hash mismatch")
    scope = run_manifest.get("scope")
    if scope not in {"pilot", "full"}:
        raise ValueError("run manifest scope must be 'pilot' or 'full'")
    if config.get("stage_size") != 504:
        raise ValueError("frozen stage_size must be exactly 504")
    if tuple(config.get("views", ())) != UPPER_VIEWS:
        raise ValueError(f"frozen views must be exactly {UPPER_VIEWS}")
    if config.get("camera_label") != CAMERA_MODE_LABEL:
        raise ValueError("frozen camera label does not match the report contract")
    if config.get("seed") != 42:
        raise ValueError("frozen feature seed must be exactly 42")
    metric = config.get("metric")
    if not isinstance(metric, Mapping):
        raise ValueError("frozen config requires a metric object")
    tolerance = metric.get("relative_depth_tolerance")
    overlap = metric.get("min_overlap_fraction")
    percentile = metric.get("confidence_percentile")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or float(tolerance) != 0.03
    ):
        raise ValueError("relative_depth_tolerance must be exactly 0.03")
    if (
        not isinstance(overlap, (int, float))
        or isinstance(overlap, bool)
        or float(overlap) != 0.08
    ):
        raise ValueError("min_overlap_fraction must be exactly 0.08")
    if (
        not isinstance(percentile, (int, float))
        or isinstance(percentile, bool)
        or float(percentile) != 10.0
    ):
        raise ValueError("confidence_percentile must be exactly 10.0")
    models = config.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("frozen config requires a models object")
    model = models.get(backbone)
    if not isinstance(model, Mapping):
        raise ValueError(f"frozen config has no model record for {backbone!r}")
    revision = model.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"frozen model {backbone!r} requires a revision")
    specification = BACKBONE_SPECS.get(backbone)
    if specification is not None and specification.revision != revision:
        raise ValueError("backbone revision does not match the pinned implementation")
    if models.get("naf_revision") != NAF_REVISION:
        raise ValueError("NAF revision does not match the pinned implementation")
    return _RunContract(
        root=root,
        config=config,
        config_sha256=config_sha256,
        scope=str(scope),
        metric_config=KnownCameraConfig(
            relative_depth_tolerance=float(tolerance),
            min_overlap_fraction=float(overlap),
        ),
        confidence_percentile=float(percentile),
        backbone_revision=revision,
    )


def _load_variants(root: Path, backbone: str) -> tuple[FeatureVariant, ...]:
    raw = json.loads(
        (root / "manifests" / "feature_variants.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, list):
        raise ValueError("feature variant manifest must contain a list")
    variants: list[FeatureVariant] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("feature variant records must be objects")
        try:
            variant = FeatureVariant(
                feature_track=str(item["feature_track"]),
                backbone=str(item["backbone"]),
                input_size=int(item["input_size"]),
                feature_mode=str(item["feature_mode"]),
                resolution=int(item["resolution"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("feature variant record is malformed") from error
        if variant.backbone == backbone:
            variants.append(variant)
    if not variants:
        raise ValueError(f"no feature variants are assigned to {backbone!r}")
    keys = [variant.key for variant in variants]
    if len(keys) != len(set(keys)):
        raise ValueError("feature variant keys must be unique")
    return tuple(variants)


def _variants_by_track(
    variants: Sequence[FeatureVariant],
) -> dict[str, tuple[FeatureVariant, ...]]:
    result: dict[str, list[FeatureVariant]] = {}
    for variant in variants:
        result.setdefault(variant.feature_track, []).append(variant)
    return {key: tuple(value) for key, value in sorted(result.items())}


def _pca_fit_paths(root: Path) -> tuple[Path, ...]:
    fit_root = root / "pca_fit_images"
    shoe_dirs = sorted(path for path in fit_root.iterdir() if path.is_dir())
    if len(shoe_dirs) != 4:
        raise ValueError("PCA fitting requires exactly four staged shoe directories")
    paths: list[Path] = []
    for shoe_dir in shoe_dirs:
        by_name = {path.stem: path for path in shoe_dir.glob("*.png")}
        if tuple(name for name in UPPER_VIEWS if name in by_name) != UPPER_VIEWS:
            raise ValueError(f"PCA fitting shoe is missing upper views: {shoe_dir}")
        if set(by_name) != set(UPPER_VIEWS):
            raise ValueError(f"PCA fitting shoe contains non-upper views: {shoe_dir}")
        paths.extend(by_name[view_id] for view_id in UPPER_VIEWS)
    if len(paths) != 16:
        raise ValueError("PCA fitting requires exactly all 16 staged upper images")
    return tuple(paths)


def _load_image_batch(
    paths: Sequence[Path],
    *,
    input_size: int,
    device: torch.device,
) -> _ImageBatch:
    arrays: list[np.ndarray] = []
    tensors: list[Tensor] = []
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (504, 504):
                raise ValueError(f"staged PCA/state image must be RGB 504x504: {path}")
            value = np.asarray(
                image.resize(
                    (input_size, input_size),
                    Image.Resampling.LANCZOS,
                )
            ).copy()
        arrays.append(value)
        tensors.append(
            torch.from_numpy(value)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
    return _ImageBatch(
        tensor=torch.stack(tensors).to(device),
        arrays=tuple(arrays),
    )


def _validate_native_shape(
    native: Tensor,
    variants: Sequence[FeatureVariant],
) -> None:
    native_variants = [
        variant for variant in variants if variant.feature_mode == "native"
    ]
    if len(native_variants) != 1:
        raise ValueError("each feature track/input must have exactly one native variant")
    expected = native_variants[0].resolution
    if native.ndim != 4 or tuple(native.shape[-2:]) != (expected, expected):
        raise ValueError(
            "backbone native grid does not match the feature variant contract: "
            f"{tuple(native.shape[-2:])} != {(expected, expected)}"
        )


def _pca_manifest_expected(
    contract: _RunContract,
    backbone: str,
    paths: Sequence[Path],
    input_sizes: Sequence[int],
) -> dict[str, object]:
    return {
        "artifact_type": "fixed_pca_basis",
        "backbone": backbone,
        "backbone_revision": contract.backbone_revision,
        "config_sha256": contract.config_sha256,
        "seed": 42,
        "fit_image_count": len(paths),
        "fit_sample_count": len(paths) * len(input_sizes),
        "input_sizes": list(input_sizes),
        "fit_images": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _install_path(staging: Path, destination: Path) -> _InstalledOutput:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    empty_marker = destination.with_name(
        f".{destination.name}.empty.{uuid.uuid4().hex}"
    )
    if destination.exists():
        destination.rename(backup)
        empty_marker = None
    else:
        backup = None
        empty_marker.touch()
    try:
        staging.rename(destination)
    except Exception:
        if destination.exists():
            _remove_path(destination)
        if backup is not None and backup.exists():
            backup.rename(destination)
        if empty_marker is not None and empty_marker.exists():
            empty_marker.unlink()
        raise
    return _InstalledOutput(destination, backup, empty_marker)


def _rollback_outputs(outputs: Sequence[_InstalledOutput]) -> None:
    for output in reversed(outputs):
        if output.destination.exists():
            _remove_path(output.destination)
        if output.backup is not None and output.backup.exists():
            output.backup.rename(output.destination)
        if output.empty_marker is not None and output.empty_marker.exists():
            output.empty_marker.unlink()


def _commit_outputs(outputs: Sequence[_InstalledOutput]) -> None:
    for output in outputs:
        if output.backup is not None and output.backup.exists():
            _remove_path(output.backup)
        if output.empty_marker is not None and output.empty_marker.exists():
            output.empty_marker.unlink()


def _fit_or_load_pca(
    contract: _RunContract,
    backbone_name: str,
    extractor: MultiScaleFeatureExtractor,
    variants: Sequence[FeatureVariant],
    device: torch.device,
) -> PCABasis:
    paths = _pca_fit_paths(contract.root)
    input_sizes = tuple(sorted({variant.input_size for variant in variants}))
    expected = _pca_manifest_expected(
        contract,
        backbone_name,
        paths,
        input_sizes,
    )
    destination_dir = contract.root / "pca" / backbone_name
    basis_path = destination_dir / "basis.npz"
    manifest_path = destination_dir / "basis.json"
    if basis_path.is_file() and manifest_path.is_file():
        try:
            basis = PCABasis.load(basis_path)
            manifest = _json_object(manifest_path)
            expected_existing = dict(expected)
            expected_existing["fingerprint"] = basis.fingerprint
            if dict(manifest) == expected_existing:
                return basis
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    batches: list[Tensor] = []
    try:
        for input_size in input_sizes:
            for offset in range(0, len(paths), 4):
                images = _load_image_batch(
                    paths[offset : offset + 4],
                    input_size=input_size,
                    device=device,
                )
                native = extractor.extract_native(images.tensor)
                _validate_native_shape(
                    native,
                    tuple(
                        variant
                        for variant in variants
                        if variant.input_size == input_size
                    ),
                )
                batches.append(native.detach().to(device="cpu"))
                del native, images
        basis = fit_pca_basis(batches, max_tokens=200_000, seed=42)
    finally:
        batches.clear()
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.staging.",
            dir=destination_dir.parent,
        )
    )
    installed: _InstalledOutput | None = None
    try:
        staged_basis = staging_dir / "basis.npz"
        basis.save(staged_basis)
        loaded = PCABasis.load(staged_basis)
        if loaded.fingerprint != basis.fingerprint:
            raise ValueError("staged PCA basis fingerprint mismatch")
        (staging_dir / "basis.json").write_text(
            json.dumps(
                {**expected, "fingerprint": basis.fingerprint},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        installed = _install_path(staging_dir, destination_dir)
        PCABasis.load(basis_path)
        if _json_object(manifest_path).get("fingerprint") != basis.fingerprint:
            raise ValueError("published PCA manifest fingerprint mismatch")
    except Exception:
        if installed is not None:
            _rollback_outputs((installed,))
        if staging_dir.exists():
            _remove_path(staging_dir)
        raise
    _commit_outputs((installed,))
    return basis


def _load_state_index(root: Path) -> dict[str, _StateRecord]:
    path = root / "manifests" / "image_states.jsonl"
    states: dict[str, _StateRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"image state line {line_number} is not an object")
        state_id = payload.get("state_id")
        track = payload.get("track")
        shoe_id = payload.get("shoe_id")
        corruption = payload.get("corruption")
        severity = payload.get("severity")
        changed_view = payload.get("changed_view")
        if (
            not isinstance(state_id, str)
            or not isinstance(track, str)
            or not isinstance(shoe_id, str)
            or not isinstance(corruption, str)
            or not isinstance(severity, (int, float))
            or isinstance(severity, bool)
            or (changed_view is not None and changed_view not in UPPER_VIEWS)
        ):
            raise ValueError(f"image state line {line_number} is malformed")
        if state_id in states:
            raise ValueError(f"duplicate image state ID: {state_id}")
        states[state_id] = _StateRecord(
            state_id=state_id,
            track=track,
            shoe_id=shoe_id,
            changed_view=changed_view,
            corruption=corruption,
            severity=float(severity),
            state_dir=(root / "states" / state_id).resolve(),
        )
    return states


def _clean_state_for_shoe(
    states: Mapping[str, _StateRecord],
    shoe_id: str,
) -> _StateRecord:
    matches = [
        state
        for state in states.values()
        if state.shoe_id == shoe_id
        and state.track == "controlled"
        and state.corruption == "clean"
    ]
    if len(matches) != 1:
        raise ValueError(f"shoe {shoe_id!r} must have exactly one clean state")
    return matches[0]


def _state_images(
    state: _StateRecord,
    input_size: int,
    device: torch.device,
) -> _ImageBatch:
    metadata = _json_object(state.state_dir / "state.json")
    raw_state = metadata.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"state metadata is malformed: {state.state_dir}")
    if raw_state.get("state_id") != state.state_id:
        raise ValueError("state metadata ID does not match its manifest")
    if raw_state.get("shoe_id") != state.shoe_id:
        raise ValueError("state metadata shoe does not match its manifest")
    if raw_state.get("track") != state.track:
        raise ValueError("state metadata track does not match its manifest")
    if metadata.get("stage_size") != 504:
        raise ValueError("materialized feature state must be staged at 504")
    paths = tuple(
        state.state_dir / "images" / f"{view_id}.png" for view_id in UPPER_VIEWS
    )
    return _load_image_batch(paths, input_size=input_size, device=device)


def _clean_features(
    cache: _CleanFeatureCache,
    state: _StateRecord,
    feature_track: str,
    input_size: int,
    extractor: MultiScaleFeatureExtractor,
    device: torch.device,
) -> tuple[_ImageBatch, Tensor]:
    key = (state.shoe_id, feature_track, input_size)
    if cache.key != key:
        cache.release()
        images = _state_images(state, input_size, device)
        native = extractor.extract_native(images.tensor)
        cache.key = key
        cache.images = images
        cache.native = native
    if cache.images is None or cache.native is None:
        raise RuntimeError("clean feature cache failed to initialize")
    return cache.images, cache.native


def _load_geometry(
    path: Path,
    device: torch.device,
    *,
    validate_metadata: bool,
) -> ViewGeometry:
    if validate_metadata:
        metadata = _json_object(path.with_name("depth_geometry.json"))
        if metadata.get("camera_mode") != CAMERA_MODE_LABEL:
            raise ValueError("geometry camera report label mismatch")
        if metadata.get("camera_provenance") != "fixed_precomputed_da3":
            raise ValueError("geometry camera provenance mismatch")
        if metadata.get("process_res") != 504:
            raise ValueError("geometry process resolution must be 504")
        if tuple(metadata.get("view_ids", ())) != UPPER_VIEWS:
            raise ValueError("geometry metadata view order mismatch")
        if tuple(metadata.get("camera_groups", ())) != ("upper",) * 4:
            raise ValueError("geometry metadata contains a non-upper view")
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "depth",
            "mask",
            "intrinsics",
            "extrinsics",
            "meta__view_ids",
            "meta__camera_groups",
            "meta__camera_provenance",
        }
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"geometry artifact is missing arrays: {sorted(missing)}")
        depth = np.asarray(artifact["depth"])
        mask = np.asarray(artifact["mask"])
        confidence = (
            np.asarray(artifact["confidence"])
            if "confidence" in artifact.files
            else None
        )
        intrinsics = np.asarray(artifact["intrinsics"])
        extrinsics = np.asarray(artifact["extrinsics"])
        views = tuple(str(value) for value in artifact["meta__view_ids"].tolist())
        groups = tuple(
            str(value) for value in artifact["meta__camera_groups"].tolist()
        )
        provenance = str(artifact["meta__camera_provenance"].item())
    if depth.shape != (4, 504, 504) or mask.shape != (4, 504, 504):
        raise ValueError("geometry must contain four 504x504 upper views")
    if confidence is not None and confidence.shape != depth.shape:
        raise ValueError("geometry confidence shape mismatch")
    if intrinsics.shape != (4, 3, 3) or extrinsics.shape != (4, 4, 4):
        raise ValueError("geometry camera arrays have unexpected shapes")
    if views != UPPER_VIEWS or groups != ("upper",) * 4:
        raise ValueError("geometry must contain only canonical upper views")
    if provenance != "fixed_precomputed_da3":
        raise ValueError("geometry artifact camera provenance mismatch")
    return ViewGeometry(
        depth=torch.as_tensor(depth, dtype=torch.float32, device=device),
        mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
        confidence=(
            torch.as_tensor(confidence, dtype=torch.float32, device=device)
            if confidence is not None
            else None
        ),
        intrinsics=torch.as_tensor(
            intrinsics,
            dtype=torch.float32,
            device=device,
        ),
        extrinsics_w2c=torch.as_tensor(
            extrinsics,
            dtype=torch.float32,
            device=device,
        ),
    )


def select_geometry(geometry: ViewGeometry, index: int) -> ViewGeometry:
    """Select one canonical upper view while retaining a batch dimension."""

    if index not in range(len(UPPER_VIEWS)):
        raise IndexError("upper geometry index is out of range")
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


def _pair_rows(root: Path) -> tuple[dict[str, object], ...]:
    frame = pd.read_parquet(root / "manifests" / "pair_conditions.parquet")
    required = {
        "design_id",
        "track",
        "shoe_id",
        "pair_id",
        "first_view",
        "second_view",
        "image_state_id",
        "corruption",
        "severity",
    }
    if not required <= set(frame):
        raise ValueError(f"pair manifest is missing columns: {sorted(required - set(frame))}")
    rows = tuple(frame.to_dict(orient="records"))
    for row in rows:
        first = row.get("first_view")
        second = row.get("second_view")
        if (
            first not in UPPER_VIEWS
            or second not in UPPER_VIEWS
            or first == second
            or row.get("pair_id") != f"{first}__{second}"
        ):
            raise ValueError("pair manifest contains a non-canonical upper pair")
    return rows


def _quality_index(root: Path) -> dict[tuple[str, str], Mapping[str, object]]:
    manifest = _json_object(root / "manifests" / "upper_manifest.json")
    shoes = manifest.get("shoes")
    if not isinstance(shoes, list):
        raise ValueError("upper manifest requires a shoes list")
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for shoe in shoes:
        if not isinstance(shoe, Mapping) or not isinstance(shoe.get("shoe_id"), str):
            raise ValueError("upper manifest shoe is malformed")
        views = shoe.get("views")
        if not isinstance(views, list):
            raise ValueError("upper manifest shoe requires views")
        for view in views:
            if not isinstance(view, Mapping) or view.get("view_id") not in UPPER_VIEWS:
                raise ValueError("upper manifest view is malformed")
            quality = view.get("quality")
            if not isinstance(quality, Mapping):
                raise ValueError("upper manifest view requires quality metadata")
            for field in _QUALITY_FIELDS:
                if field not in quality:
                    raise ValueError(f"upper manifest quality is missing {field}")
            result[(str(shoe["shoe_id"]), str(view["view_id"]))] = quality
    return result


def _quality_columns(
    qualities: Mapping[tuple[str, str], Mapping[str, object]],
    shoe_id: str,
    first_view: str,
    second_view: str,
    *,
    stress: bool,
) -> dict[str, object]:
    if not stress:
        return {
            f"{prefix}_{field}": None
            for prefix in ("first", "second")
            for field in _QUALITY_FIELDS
        }
    result: dict[str, object] = {}
    for prefix, view_id in (("first", first_view), ("second", second_view)):
        try:
            quality = qualities[(shoe_id, view_id)]
        except KeyError as error:
            raise ValueError(
                f"stress quality is missing for {shoe_id}/{view_id}"
            ) from error
        for field in _QUALITY_FIELDS:
            result[f"{prefix}_{field}"] = quality[field]
    return result


def _row_provenance(
    contract: _RunContract,
    job: ClaimedJob,
    variant: FeatureVariant,
) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "analysis_track": job.payload["analysis_track"],
        "feature_track": variant.feature_track,
        "variant_key": variant.key,
        "backbone": variant.backbone,
        "backbone_revision": contract.backbone_revision,
        "input_size": variant.input_size,
        "feature_mode": variant.feature_mode,
        "resolution": variant.resolution,
        "config_sha256": contract.config_sha256,
        "camera_provenance": "fixed_precomputed_da3",
        "camera_label": CAMERA_MODE_LABEL,
    }


def _condition_columns(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "design_id": row["design_id"],
        "track": row["track"],
        "shoe_id": row["shoe_id"],
        "pair_id": row["pair_id"],
        "first_view": row["first_view"],
        "second_view": row["second_view"],
        "changed_view": (
            None
            if row.get("changed_view") is None
            or (
                isinstance(row.get("changed_view"), float)
                and math.isnan(float(row["changed_view"]))
            )
            else row["changed_view"]
        ),
        "image_state_id": row["image_state_id"],
        "corruption": row["corruption"],
        "severity": float(row["severity"]),
    }


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-+" else "_"
        for character in value
    )


def _maximum_severity(contract: _RunContract, corruption: str) -> float | None:
    corruptions = contract.config.get("corruptions")
    if not isinstance(corruptions, Mapping):
        return None
    raw = corruptions.get(corruption)
    if not isinstance(raw, list) or not raw:
        return None
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in raw
    ):
        raise ValueError(f"corruption severities are malformed for {corruption!r}")
    return max(float(value) for value in raw)


def _retain_diagnostic(
    contract: _RunContract,
    condition: Mapping[str, object],
    *,
    abstained: bool,
) -> bool:
    if contract.scope == "pilot" or abstained:
        return True
    corruption = str(condition["corruption"])
    severity = float(condition["severity"])
    if corruption == "clean":
        return True
    middle = _MIDDLE_DIAGNOSTIC_SEVERITY.get(corruption)
    if middle is not None and math.isclose(severity, middle):
        return True
    maximum = _maximum_severity(contract, corruption)
    return maximum is not None and math.isclose(severity, maximum)


def _write_manifest(directory: Path, name: str, files: Sequence[Path]) -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": str(file.relative_to(directory)),
                        "sha256": sha256_file(file),
                    }
                    for file in sorted(files)
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _render_pca_variant(
    feature: Tensor,
    basis: PCABasis,
    variant: FeatureVariant,
    output_dir: Path,
    *,
    track_max_resolution: int,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, view_id in enumerate(UPPER_VIEWS):
        path = output_dir / f"{_safe_name(variant.key)}__{view_id}.png"
        rendered = render_pca_png(
            feature[index : index + 1],
            basis,
            variant,
            path,
            track_max_resolution=track_max_resolution,
        )
        if rendered is not None:
            paths.append(rendered)
    return tuple(paths)


def _save_factorial_diagnostics(
    result: object,
    condition: Mapping[str, object],
    variant: FeatureVariant,
    changed_images: _ImageBatch,
    output_dir: Path,
) -> tuple[Path, ...]:
    first_view = str(condition["first_view"])
    second_view = str(condition["second_view"])
    first = UPPER_VIEWS.index(first_view)
    second = UPPER_VIEWS.index(second_view)
    design = _safe_name(str(condition["design_id"]))
    key = _safe_name(variant.key)
    first_path = output_dir / f"{design}__{key}__first_to_second.png"
    second_path = output_dir / f"{design}__{key}__second_to_first.png"
    save_direction_diagnostics(
        changed_images.arrays[first],
        changed_images.arrays[second],
        result.s11.first_to_second,
        first_path,
    )
    save_direction_diagnostics(
        changed_images.arrays[second],
        changed_images.arrays[first],
        result.s11.second_to_first,
        second_path,
    )
    return first_path, second_path


def _zip_variants(
    clean_native: Tensor,
    clean_images: Tensor,
    changed_native: Tensor,
    changed_images: Tensor,
    variants: Sequence[FeatureVariant],
    naf: torch.nn.Module,
):
    clean_iterator = iter_feature_variants(
        clean_native,
        clean_images,
        variants,
        naf,
    )
    changed_iterator = iter_feature_variants(
        changed_native,
        changed_images,
        variants,
        naf,
    )
    with closing(clean_iterator), closing(changed_iterator):
        for expected, clean_item, changed_item in zip(
            variants,
            clean_iterator,
            changed_iterator,
            strict=True,
        ):
            clean_variant, clean_feature = clean_item
            changed_variant, changed_feature = changed_item
            if (
                clean_variant.key != expected.key
                or changed_variant.key != expected.key
                or clean_variant.key != changed_variant.key
            ):
                raise ValueError("clean and changed feature variants are not exact-key zipped")
            yield expected, clean_feature, changed_feature


def _stage_job_directories(
    contract: _RunContract,
    job: ClaimedJob,
    feature_track: str,
) -> tuple[Path, Path, Path, Path]:
    pca_destination = (
        contract.root
        / "pca"
        / str(job.payload["backbone"])
        / feature_track
        / job.job_id
    )
    diagnostic_destination = contract.root / "visualizations" / job.job_id
    pca_destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_destination.parent.mkdir(parents=True, exist_ok=True)
    pca_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{pca_destination.name}.staging.",
            dir=pca_destination.parent,
        )
    )
    diagnostic_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{diagnostic_destination.name}.staging.",
            dir=diagnostic_destination.parent,
        )
    )
    return (
        pca_staging,
        pca_destination,
        diagnostic_staging,
        diagnostic_destination,
    )


def _stage_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    if frame.empty:
        raise ValueError("metric Parquet part must contain at least one row")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    if temporary.exists():
        _remove_path(temporary)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    try:
        pq.write_table(table, temporary)
        readback = pq.read_table(temporary)
        if readback.num_rows != len(frame):
            raise ValueError(
                f"Parquet row-count mismatch: {readback.num_rows} != {len(frame)}"
            )
        if not readback.schema.equals(table.schema, check_metadata=True):
            raise ValueError("Parquet schema changed during readback")
    except Exception:
        if temporary.exists():
            _remove_path(temporary)
        raise
    return temporary


def _write_parquet_atomic(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Write, read back, install, and hash one Parquet part atomically."""

    final = Path(destination)
    temporary = _stage_parquet(frame, final)
    installed: _InstalledOutput | None = None
    try:
        installed = _install_path(temporary, final)
        sha256_file(final)
    except Exception:
        if installed is not None:
            _rollback_outputs((installed,))
        if temporary.exists():
            _remove_path(temporary)
        raise
    _commit_outputs((installed,))
    return final


def _output_manifest(paths: Sequence[Path]) -> dict[str, object]:
    return {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(paths)
        ]
    }


def _metric_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    boolean_columns = {
        "abstained",
        "abstention",
        *(
            f"{prefix}_{field}"
            for prefix in ("first", "second")
            for field in (
                "semantic_gate",
                "preservation_gate",
                "camera_preserved",
                "outsole_preserved",
            )
        ),
    }
    string_columns = {
        "changed_view",
        "first_fit_relation",
        "second_fit_relation",
    }
    for column in boolean_columns & set(frame):
        frame[column] = frame[column].astype("boolean")
    for column in string_columns & set(frame):
        frame[column] = frame[column].astype("string")
    return frame


def _publish_job_outputs(
    registry: JobRegistry,
    job: ClaimedJob,
    frame: pd.DataFrame,
    part_path: Path,
    directory_pairs: Sequence[tuple[Path, Path]],
) -> None:
    staged_part = _stage_parquet(frame, part_path)
    installed: list[_InstalledOutput] = []
    try:
        for staging, destination in directory_pairs:
            installed.append(_install_path(staging, destination))
        installed.append(_install_path(staged_part, part_path))
        published_files = [part_path]
        for _, destination in directory_pairs:
            published_files.extend(
                path for path in destination.rglob("*") if path.is_file()
            )
        _assert_no_persistent_features(part_path.parents[3])
        manifest = _output_manifest(published_files)
        registry.complete(job.job_id, manifest)
    except Exception:
        _rollback_outputs(installed)
        if staged_part.exists():
            _remove_path(staged_part)
        for staging, _ in directory_pairs:
            if staging.exists():
                _remove_path(staging)
        raise
    _commit_outputs(installed)


def _factorial_rows(
    contract: _RunContract,
    job: ClaimedJob,
    basis: PCABasis,
    extractor: MultiScaleFeatureExtractor,
    naf: torch.nn.Module,
    variants: Sequence[FeatureVariant],
    state: _StateRecord,
    clean_state: _StateRecord,
    pairs: Sequence[Mapping[str, object]],
    qualities: Mapping[tuple[str, str], Mapping[str, object]],
    device: torch.device,
    clean_cache: _CleanFeatureCache,
    pca_staging: Path,
    diagnostic_staging: Path,
) -> list[dict[str, object]]:
    input_sizes = {variant.input_size for variant in variants}
    if len(input_sizes) != 1:
        raise ValueError("one feature track must use exactly one input size")
    input_size = next(iter(input_sizes))
    clean_images, clean_native = _clean_features(
        clean_cache,
        clean_state,
        variants[0].feature_track,
        input_size,
        extractor,
        device,
    )
    _validate_native_shape(clean_native, variants)
    changed_is_clean = state.state_id == clean_state.state_id
    if changed_is_clean:
        changed_images = clean_images
        changed_native = clean_native
    else:
        changed_images = _state_images(state, input_size, device)
        changed_native = extractor.extract_native(changed_images.tensor)
        _validate_native_shape(changed_native, variants)
    base_geometry = _load_geometry(
        contract.root / "geometry" / clean_state.state_id / "depth_geometry.npz",
        device,
        validate_metadata=True,
    )
    adaptive_geometry = _load_geometry(
        contract.root / "geometry" / state.state_id / "depth_geometry.npz",
        device,
        validate_metadata=True,
    )
    track_max = max(variant.resolution for variant in variants)
    rows: list[dict[str, object]] = []
    pca_paths: list[Path] = []
    diagnostic_paths: list[Path] = []
    try:
        for variant, clean_feature, changed_feature in _zip_variants(
            clean_native,
            clean_images.tensor,
            changed_native,
            changed_images.tensor,
            variants,
            naf,
        ):
            pca_paths.extend(
                _render_pca_variant(
                    changed_feature,
                    basis,
                    variant,
                    pca_staging,
                    track_max_resolution=track_max,
                )
            )
            for condition in pairs:
                first_view = str(condition["first_view"])
                second_view = str(condition["second_view"])
                first = UPPER_VIEWS.index(first_view)
                second = UPPER_VIEWS.index(second_view)
                result = score_multiscale_factorial(
                    {variant.key: clean_feature[first : first + 1]},
                    {variant.key: clean_feature[second : second + 1]},
                    {variant.key: changed_feature[first : first + 1]},
                    {variant.key: changed_feature[second : second + 1]},
                    select_geometry(base_geometry, first),
                    select_geometry(base_geometry, second),
                    select_geometry(adaptive_geometry, first),
                    select_geometry(adaptive_geometry, second),
                    config=contract.metric_config,
                    confidence_percentile=contract.confidence_percentile,
                )[variant.key]
                flat = flatten_factorial_result(result)
                row = {
                    **_condition_columns(condition),
                    **_row_provenance(contract, job, variant),
                    **flat,
                    **_quality_columns(
                        qualities,
                        state.shoe_id,
                        first_view,
                        second_view,
                        stress=job.payload["job_type"] == "stress",
                    ),
                }
                rows.append(row)
                if _retain_diagnostic(
                    contract,
                    condition,
                    abstained=bool(flat["abstained"]),
                ):
                    diagnostic_paths.extend(
                        _save_factorial_diagnostics(
                            result,
                            condition,
                            variant,
                            changed_images,
                            diagnostic_staging,
                        )
                    )
    finally:
        del clean_native, base_geometry, adaptive_geometry, clean_images
        if not changed_is_clean:
            del changed_native, changed_images
    _write_manifest(pca_staging, "manifest.json", pca_paths)
    _write_manifest(diagnostic_staging, "manifest.json", diagnostic_paths)
    return rows


def _masked_median(direction: PairMetricResult, name: str) -> float:
    values = getattr(direction, name)[0][direction.valid_mask[0]]
    if values.numel() == 0:
        return float("nan")
    return float(values.detach().float().median().cpu().item())


def _sensitivity_row(result: GeometrySensitivityResult) -> dict[str, object]:
    baseline = result.baseline
    perturbed = result.perturbed
    cycle = (
        _masked_median(perturbed.first_to_second, "cycle_error_px"),
        _masked_median(perturbed.second_to_first, "cycle_error_px"),
    )
    depth = (
        _masked_median(perturbed.first_to_second, "relative_depth_error"),
        _masked_median(perturbed.second_to_first, "relative_depth_error"),
    )
    finite_cycle = [value for value in cycle if math.isfinite(value)]
    finite_depth = [value for value in depth if math.isfinite(value)]
    return {
        "base_score": float(baseline.score[0].detach().cpu().item()),
        "perturbed_score": float(perturbed.score[0].detach().cpu().item()),
        "geometry_effect": float(result.geometry_effect[0].detach().cpu().item()),
        "base_first_to_second_score": float(
            baseline.first_to_second.score[0].detach().cpu().item()
        ),
        "base_second_to_first_score": float(
            baseline.second_to_first.score[0].detach().cpu().item()
        ),
        "perturbed_first_to_second_score": float(
            perturbed.first_to_second.score[0].detach().cpu().item()
        ),
        "perturbed_second_to_first_score": float(
            perturbed.second_to_first.score[0].detach().cpu().item()
        ),
        "base_first_to_second_overlap": float(
            baseline.first_to_second.overlap_fraction[0].detach().cpu().item()
        ),
        "base_second_to_first_overlap": float(
            baseline.second_to_first.overlap_fraction[0].detach().cpu().item()
        ),
        "perturbed_first_to_second_overlap": float(
            perturbed.first_to_second.overlap_fraction[0].detach().cpu().item()
        ),
        "perturbed_second_to_first_overlap": float(
            perturbed.second_to_first.overlap_fraction[0].detach().cpu().item()
        ),
        "overlap": min(
            float(
                perturbed.first_to_second.overlap_fraction[0].detach().cpu().item()
            ),
            float(
                perturbed.second_to_first.overlap_fraction[0].detach().cpu().item()
            ),
        ),
        "abstained": bool(perturbed.abstained[0].detach().cpu().item()),
        "abstention": bool(perturbed.abstained[0].detach().cpu().item()),
        "cycle_error_px_median": (
            sum(finite_cycle) / len(finite_cycle)
            if finite_cycle
            else float("nan")
        ),
        "relative_depth_error_median": (
            sum(finite_depth) / len(finite_depth)
            if finite_depth
            else float("nan")
        ),
    }


def _calibration_records(
    contract: _RunContract,
    clean_state: _StateRecord,
) -> tuple[Mapping[str, object], ...]:
    path = (
        contract.root
        / "geometry"
        / clean_state.state_id
        / "calibration"
        / "calibration_manifest.json"
    )
    manifest = _json_object(path)
    if manifest.get("artifact_type") != "upper_geometry_calibration":
        raise ValueError("geometry calibration artifact type mismatch")
    if manifest.get("config_sha256") != contract.config_sha256:
        raise ValueError("geometry calibration config hash mismatch")
    if manifest.get("seed") != 42:
        raise ValueError("geometry calibration seed mismatch")
    if tuple(manifest.get("views", ())) != UPPER_VIEWS:
        raise ValueError("geometry calibration view order mismatch")
    expected_base = (
        contract.root
        / "geometry"
        / clean_state.state_id
        / "depth_geometry.npz"
    ).resolve()
    if Path(str(manifest.get("base_geometry", ""))).resolve() != expected_base:
        raise ValueError("geometry calibration base path mismatch")
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list) or len(raw_records) != 68:
        raise ValueError("geometry calibration must contain exactly 68 artifacts")
    calibration = contract.config.get("geometry_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("frozen config requires geometry calibration values")
    expected_keys = {
        "rotation": "rotation_degrees",
        "relative_translation": "relative_translation",
        "depth_scale": "depth_scale",
        "relative_depth_noise": "relative_depth_noise_std",
    }
    expected_values: dict[str, set[float]] = {}
    for kind, config_key in expected_keys.items():
        values = calibration.get(config_key)
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
        ):
            raise ValueError(
                f"geometry_calibration.{config_key} must be a non-empty number list"
            )
        expected_values[kind] = {float(value) for value in values}
    if sum(len(values) for values in expected_values.values()) != 17:
        raise ValueError("frozen geometry calibration must define 17 perturbations")
    calibration_root = path.parent.resolve()
    records: list[Mapping[str, object]] = []
    counts = {view_id: 0 for view_id in UPPER_VIEWS}
    identities: set[tuple[str, str, float]] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("geometry calibration record is malformed")
        view_id = raw.get("view_id")
        kind = raw.get("kind")
        magnitude = raw.get("magnitude")
        view_index = raw.get("view_index")
        if (
            view_id not in UPPER_VIEWS
            or view_index != UPPER_VIEWS.index(str(view_id))
            or not isinstance(kind, str)
            or not isinstance(magnitude, (int, float))
            or isinstance(magnitude, bool)
            or float(magnitude) <= 0
        ):
            raise ValueError("geometry calibration identity is malformed")
        identity = (str(view_id), kind, float(magnitude))
        if identity in identities:
            raise ValueError("geometry calibration identities must be unique")
        identities.add(identity)
        artifact_path = _path_within(
            str(raw.get("path", "")),
            calibration_root,
            "calibration artifact",
        )
        expected_hash = raw.get("sha256")
        if (
            not artifact_path.is_file()
            or not isinstance(expected_hash, str)
            or sha256_file(artifact_path) != expected_hash
        ):
            raise ValueError("geometry calibration artifact hash mismatch")
        counts[str(view_id)] += 1
        records.append({**raw, "path": str(artifact_path)})
    if set(counts.values()) != {17}:
        raise ValueError("geometry calibration requires 17 perturbations per view")
    for view_id in UPPER_VIEWS:
        for kind, magnitudes in expected_values.items():
            actual = {
                float(record["magnitude"])
                for record in records
                if record["view_id"] == view_id and record["kind"] == kind
            }
            if actual != magnitudes:
                raise ValueError(
                    f"geometry calibration values mismatch for {view_id}/{kind}"
                )
    return tuple(records)


def _calibration_rows(
    contract: _RunContract,
    job: ClaimedJob,
    extractor: MultiScaleFeatureExtractor,
    naf: torch.nn.Module,
    variants: Sequence[FeatureVariant],
    clean_state: _StateRecord,
    pairs: Sequence[Mapping[str, object]],
    device: torch.device,
    clean_cache: _CleanFeatureCache,
    pca_staging: Path,
    diagnostic_staging: Path,
) -> list[dict[str, object]]:
    input_sizes = {variant.input_size for variant in variants}
    if len(input_sizes) != 1:
        raise ValueError("one feature track must use exactly one input size")
    input_size = next(iter(input_sizes))
    images, native = _clean_features(
        clean_cache,
        clean_state,
        variants[0].feature_track,
        input_size,
        extractor,
        device,
    )
    _validate_native_shape(native, variants)
    base_geometry = _load_geometry(
        contract.root / "geometry" / clean_state.state_id / "depth_geometry.npz",
        device,
        validate_metadata=True,
    )
    records = _calibration_records(contract, clean_state)
    rows: list[dict[str, object]] = []
    try:
        for variant, clean_feature, changed_feature in _zip_variants(
            native,
            images.tensor,
            native,
            images.tensor,
            variants,
            naf,
        ):
            if clean_feature.data_ptr() != changed_feature.data_ptr():
                if not torch.equal(clean_feature, changed_feature):
                    raise ValueError("calibration feature iterators diverged")
            for record in records:
                changed_view = str(record["view_id"])
                perturbed = _load_geometry(
                    Path(str(record["path"])),
                    device,
                    validate_metadata=False,
                )
                for condition in pairs:
                    if changed_view not in {
                        condition["first_view"],
                        condition["second_view"],
                    }:
                        continue
                    first = UPPER_VIEWS.index(str(condition["first_view"]))
                    second = UPPER_VIEWS.index(str(condition["second_view"]))
                    first_base = select_geometry(base_geometry, first)
                    second_base = select_geometry(base_geometry, second)
                    first_perturbed = select_geometry(perturbed, first)
                    second_perturbed = select_geometry(perturbed, second)
                    first_base_mask = first_base.mask
                    second_base_mask = second_base.mask
                    first_perturbed_mask = first_perturbed.mask
                    second_perturbed_mask = second_perturbed.mask
                    if first_base.confidence is not None:
                        first_base_mask, _ = confidence_percentile_mask(
                            first_base_mask,
                            first_base.confidence,
                            percentile=contract.confidence_percentile,
                        )
                    if second_base.confidence is not None:
                        second_base_mask, _ = confidence_percentile_mask(
                            second_base_mask,
                            second_base.confidence,
                            percentile=contract.confidence_percentile,
                        )
                    if first_perturbed.confidence is not None:
                        first_perturbed_mask, _ = confidence_percentile_mask(
                            first_perturbed_mask,
                            first_perturbed.confidence,
                            percentile=contract.confidence_percentile,
                        )
                    if second_perturbed.confidence is not None:
                        second_perturbed_mask, _ = confidence_percentile_mask(
                            second_perturbed_mask,
                            second_perturbed.confidence,
                            percentile=contract.confidence_percentile,
                        )
                    first_shared_mask = first_base_mask & first_perturbed_mask
                    second_shared_mask = second_base_mask & second_perturbed_mask
                    output_size = tuple(clean_feature.shape[-2:])
                    first_base_scaled = resize_camera_geometry(
                        first_base.depth,
                        first_shared_mask,
                        first_base.confidence,
                        first_base.intrinsics,
                        output_size=output_size,
                    )
                    second_base_scaled = resize_camera_geometry(
                        second_base.depth,
                        second_shared_mask,
                        second_base.confidence,
                        second_base.intrinsics,
                        output_size=output_size,
                    )
                    first_perturbed_scaled = resize_camera_geometry(
                        first_perturbed.depth,
                        first_shared_mask,
                        first_perturbed.confidence,
                        first_perturbed.intrinsics,
                        output_size=output_size,
                    )
                    second_perturbed_scaled = resize_camera_geometry(
                        second_perturbed.depth,
                        second_shared_mask,
                        second_perturbed.confidence,
                        second_perturbed.intrinsics,
                        output_size=output_size,
                    )
                    result = score_geometry_sensitivity_pair(
                        clean_feature[first : first + 1],
                        clean_feature[second : second + 1],
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
                        config=contract.metric_config,
                    )
                    rows.append(
                        {
                            **_condition_columns(condition),
                            **_row_provenance(contract, job, variant),
                            "changed_view": changed_view,
                            "perturbation_kind": record["kind"],
                            "magnitude": float(record["magnitude"]),
                            **_sensitivity_row(result),
                        }
                    )
                del perturbed
    finally:
        del native, images, base_geometry
    _write_manifest(pca_staging, "manifest.json", ())
    _write_manifest(diagnostic_staging, "manifest.json", ())
    return rows


def _part_path(contract: _RunContract, job: ClaimedJob) -> Path:
    job_type = job.payload.get("job_type")
    track = (
        "geometry_calibration"
        if job_type == "geometry_calibration"
        else str(job.payload["feature_track"])
    )
    return contract.root / "metrics" / "parts" / track / f"{job.job_id}.parquet"


def _job_destinations(
    contract: _RunContract,
    job_id: str,
    payload: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    fake = ClaimedJob(job_id, "feature", payload, 0)
    return (
        _part_path(contract, fake),
        contract.root
        / "pca"
        / str(payload["backbone"])
        / str(payload["feature_track"])
        / job_id,
        contract.root / "visualizations" / job_id,
    )


def _siblings(destination: Path, kind: str) -> list[Path]:
    return sorted(destination.parent.glob(f".{destination.name}.{kind}.*"))


def _recover_destination(destination: Path, *, committed: bool) -> None:
    backups = _siblings(destination, "backup")
    stagings = _siblings(destination, "staging")
    empty_markers = _siblings(destination, "empty")
    temporary = Path(f"{destination}.tmp")
    if len(backups) > 1 or len(empty_markers) > 1:
        raise ValueError(f"multiple interrupted install markers found for {destination}")
    if backups and empty_markers:
        raise ValueError(f"conflicting interrupted install markers found for {destination}")
    if committed:
        for path in (*backups, *empty_markers, *stagings):
            _remove_path(path)
        if temporary.exists():
            _remove_path(temporary)
        return
    if backups:
        if destination.exists():
            _remove_path(destination)
        backups[0].rename(destination)
    elif empty_markers:
        if destination.exists():
            _remove_path(destination)
        empty_markers[0].unlink()
    for path in stagings:
        _remove_path(path)
    if temporary.exists():
        _remove_path(temporary)


def _feature_job_rows(
    registry: JobRegistry,
    backbone: str,
) -> tuple[tuple[str, str, Mapping[str, object]], ...]:
    with sqlite3.connect(registry.path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT job_id, state, payload_json
            FROM jobs
            WHERE phase = 'feature'
            ORDER BY job_id
            """
        ).fetchall()
    result = []
    for job_id, state, payload_json in rows:
        payload = json.loads(payload_json)
        if not isinstance(payload, Mapping):
            raise ValueError(f"stored feature payload is malformed: {job_id}")
        if payload.get("backbone") == backbone:
            result.append((str(job_id), str(state), payload))
    return tuple(result)


def _recover_feature_jobs(
    contract: _RunContract,
    registry: JobRegistry,
    backbone: str,
) -> None:
    for job_id, state, payload in _feature_job_rows(registry, backbone):
        if state != "completed":
            continue
        validated = registry.validated_state(job_id)
        for destination in _job_destinations(contract, job_id, payload):
            _recover_destination(
                destination,
                committed=validated == "completed",
            )
    registry.requeue_stale(
        phase="feature",
        max_age_seconds=_STALE_SECONDS,
        now=datetime.now(timezone.utc),
    )
    for job_id, state, payload in _feature_job_rows(registry, backbone):
        if state == "pending":
            for destination in _job_destinations(contract, job_id, payload):
                _recover_destination(destination, committed=False)


def _assert_no_persistent_features(run_root: str | Path) -> None:
    """Reject persistent raw DINO tensors regardless of their filename."""

    root = Path(run_root)

    def handle_walk_error(error: OSError) -> None:
        if error.errno != errno.ENOENT:
            raise error

    for directory, _, filenames in os.walk(root, onerror=handle_walk_error):
        for filename in filenames:
            path = Path(directory) / filename
            if path.name == "features.npz":
                raise ValueError(f"persistent DINO feature artifact found: {path}")
            if path.suffix != ".npz":
                continue
            try:
                with np.load(path, allow_pickle=False) as artifact:
                    for key in ("meta__artifact_type", "artifact_type"):
                        if key in artifact.files:
                            value = str(np.asarray(artifact[key]).item())
                            if value == "dino_features":
                                raise ValueError(
                                    "persistent DINO feature artifact found: "
                                    f"{path}"
                                )
            except FileNotFoundError:
                continue


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS


def _run_claimed_job(
    contract: _RunContract,
    registry: JobRegistry,
    job: ClaimedJob,
    basis: PCABasis,
    extractor: MultiScaleFeatureExtractor,
    naf: torch.nn.Module,
    variants: Sequence[FeatureVariant],
    states: Mapping[str, _StateRecord],
    all_pairs: Sequence[Mapping[str, object]],
    qualities: Mapping[tuple[str, str], Mapping[str, object]],
    device: torch.device,
    clean_cache: _CleanFeatureCache,
) -> None:
    if job.payload.get("config_sha256") != contract.config_sha256:
        raise ValueError("feature job config hash mismatch")
    if job.payload.get("backbone") != variants[0].backbone:
        raise ValueError("feature job backbone mismatch")
    if job.payload.get("feature_track") != variants[0].feature_track:
        raise ValueError("feature job track mismatch")
    job_type = job.payload.get("job_type")
    if job_type not in {"feature", "stress", "geometry_calibration"}:
        raise ValueError(f"unexpected feature job type: {job_type}")
    state_id = job.payload.get("state_id")
    if not isinstance(state_id, str) or state_id not in states:
        raise ValueError("feature job references an unknown state")
    state = states[state_id]
    state_dir = _path_within(
        str(job.payload.get("state_dir", "")),
        contract.root / "states",
        "state_dir",
    )
    if state_dir != state.state_dir:
        raise ValueError("feature job state path does not match the manifest")
    if (
        job.payload.get("shoe_id") != state.shoe_id
        or job.payload.get("track") != state.track
    ):
        raise ValueError("feature job state identity mismatch")
    expected_analysis = {
        "feature": "controlled",
        "stress": "stress",
        "geometry_calibration": "geometry_calibration",
    }[str(job_type)]
    if job.payload.get("analysis_track") != expected_analysis:
        raise ValueError("feature job analysis-track mismatch")
    expected_track = "stress" if job_type == "stress" else "controlled"
    if state.track != expected_track:
        raise ValueError("feature job type does not match the state track")
    if job_type == "geometry_calibration" and state.corruption != "clean":
        raise ValueError("geometry calibration jobs require a clean state")
    clean_state = _clean_state_for_shoe(states, state.shoe_id)
    pairs = tuple(
        row for row in all_pairs if row.get("image_state_id") == state.state_id
    )
    if not pairs:
        raise ValueError("feature job has no attached pair conditions")
    expected_changed_view = (
        state.changed_view if job_type == "feature" else None
    )
    for pair in pairs:
        changed_view = _condition_columns(pair)["changed_view"]
        if (
            pair.get("shoe_id") != state.shoe_id
            or pair.get("track") != state.track
            or changed_view != expected_changed_view
            or (
                changed_view is not None
                and changed_view
                not in {pair.get("first_view"), pair.get("second_view")}
            )
        ):
            raise ValueError("pair condition does not match its image state")
    pca_staging, pca_destination, diagnostic_staging, diagnostic_destination = (
        _stage_job_directories(contract, job, variants[0].feature_track)
    )
    try:
        if job_type == "geometry_calibration":
            rows = _calibration_rows(
                contract,
                job,
                extractor,
                naf,
                variants,
                clean_state,
                pairs,
                device,
                clean_cache,
                pca_staging,
                diagnostic_staging,
            )
        else:
            rows = _factorial_rows(
                contract,
                job,
                basis,
                extractor,
                naf,
                variants,
                state,
                clean_state,
                pairs,
                qualities,
                device,
                clean_cache,
                pca_staging,
                diagnostic_staging,
            )
        _publish_job_outputs(
            registry,
            job,
            _metric_frame(rows),
            _part_path(contract, job),
            (
                (pca_staging, pca_destination),
                (diagnostic_staging, diagnostic_destination),
            ),
        )
    finally:
        if pca_staging.exists():
            _remove_path(pca_staging)
        if diagnostic_staging.exists():
            _remove_path(diagnostic_staging)


def run_feature_worker(
    run_root: str | Path,
    backbone: str,
    device: str,
) -> WorkerSummary:
    """Claim one backbone's feature jobs while retaining no raw feature artifact."""

    root = Path(run_root).expanduser().resolve()
    contract = _load_run_contract(root, backbone)
    variants = _load_variants(root, backbone)
    tracks = _variants_by_track(variants)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    _assert_no_persistent_features(root)
    _recover_feature_jobs(contract, registry, backbone)

    torch_device = torch.device(device)
    backbone_model = load_feature_backbone(backbone)
    backbone_model = backbone_model.to(torch_device).eval()
    naf = load_naf_upsampler(device=torch_device)
    naf = naf.to(torch_device).eval()
    extractor = MultiScaleFeatureExtractor(backbone_model)
    basis = _fit_or_load_pca(
        contract,
        backbone,
        extractor,
        variants,
        torch_device,
    )
    states = _load_state_index(root)
    pairs = _pair_rows(root)
    qualities = _quality_index(root)

    completed = 0
    failed = 0
    for feature_track, track_variants in tracks.items():
        clean_cache = _CleanFeatureCache()
        try:
            while True:
                job = registry.claim_next(
                    worker_id=f"feature:{backbone}",
                    phase="feature",
                    filters={
                        "backbone": backbone,
                        "feature_track": feature_track,
                    },
                )
                if job is None:
                    break
                try:
                    _run_claimed_job(
                        contract,
                        registry,
                        job,
                        basis,
                        extractor,
                        naf,
                        track_variants,
                        states,
                        pairs,
                        qualities,
                        torch_device,
                        clean_cache,
                    )
                    completed += 1
                except Exception as error:
                    trace = traceback.format_exc()
                    retryable = _is_transient(error)
                    error.__traceback__ = None
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        clean_cache.release()
                    gc.collect()
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    registry.fail(job.job_id, trace, retryable=retryable)
                    if not retryable:
                        raise
                    if job.attempts >= 3:
                        failed += 1
        finally:
            clean_cache.release()
    del basis, extractor, naf, backbone_model
    return WorkerSummary(completed=completed, abstained=0, failed=failed)
