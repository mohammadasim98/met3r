"""Fail-closed hard gates for the frozen adjacent-16 experiment."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from .adjacent16_contract import Adjacent16Contract, load_adjacent16_contract
from .adjacent16_data import (
    Adjacent16Manifest,
    AdjacentShoeRecord,
    AdjacentViewRecord,
    camera_azimuths,
)
from .adjacent16_design import Adjacent16Design, build_adjacent16_design
from .camera_metadata import CAMERA_MODE_LABEL
from .geometry import confidence_percentile_mask
from .job_registry import sha256_file
from .known_camera import KnownCameraConfig, score_symmetric_geometry_support


_PREFLIGHT_COLUMNS = (
    "shoe_id", "quartet_id", "pair_id", "first_view", "second_view",
    "baseline_steps", "angular_separation_deg", "first_to_second_overlap",
    "second_to_first_overlap", "minimum_directional_overlap",
    "overlap_asymmetry", "common_valid_fraction", "cycle_error_px_median",
    "relative_depth_error_median", "symmetric_pass",
)
_TRACK_FILES = {
    "controlled": "controlled_fidelity.parquet",
    "geometry_calibration": "geometry_calibration.parquet",
    "stress": "real_deformation_stress.parquet",
}
_EXPECTED_REGISTRY_COUNTS = {"geometry": 32, "feature": 3472}
_FEATURE_CLASS_COUNTS = {
    "dino_vitb16": {"clean": 16, "changed": 832, "stress": 16, "calibration": 0},
    "dinov2_vitb14": {"clean": 16, "changed": 832, "stress": 16, "calibration": 0},
    "dinov3_vitb16": {"clean": 16, "changed": 832, "stress": 16, "calibration": 0},
    "dinov3_vitl16_pixal3d": {"clean": 16, "changed": 832, "stress": 16, "calibration": 16},
}
_ADJACENT_MANIFEST_KEYS = frozenset(
    {"reconstruction_root", "stress_root", "shoes"}
)
_ADJACENT_SHOE_KEYS = frozenset({"shoe_id", "camera_dataset_id", "views"})
_ADJACENT_VIEW_KEYS = frozenset(
    {
        "shoe_id",
        "camera_dataset_id",
        "quartet_id",
        "view_id",
        "orbit_position",
        "source_path",
        "source_mask_path",
        "stress_path",
        "stress_mask_path",
        "deformation_json_path",
        "metrics_json_path",
        "task_json_path",
        "processed_image_correlation",
        "camera_azimuth_deg",
        "source_sha256",
        "source_mask_sha256",
        "stress_sha256",
        "stress_mask_sha256",
        "quality",
    }
)
_CAMERA_NPZ_KEYS = frozenset(
    {"extrinsics", "intrinsics", "mask", "view_ids", "camera_groups"}
)
_FIXED_GEOMETRY_NPZ_KEYS = frozenset(
    {
        "depth",
        "confidence",
        "mask",
        "extrinsics",
        "intrinsics",
        "meta__artifact_type",
        "meta__model_id",
        "meta__model_revision",
        "meta__runtime_fingerprint",
        "meta__source_revision",
        "meta__source_dirty",
        "meta__source_dirty_fingerprint",
        "meta__camera_provenance",
        "meta__image_set_id",
        "meta__view_ids",
        "meta__camera_groups",
        "meta__analysis_tier",
    }
)


@dataclass(frozen=True)
class GeometryPreflightReport:
    passed: bool
    pair_units: int
    symmetric_pass_rate: float
    median_min_directional_overlap: float
    per_shoe_pass_rate: Mapping[str, float]
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeProjectionReport:
    passed: bool
    sentinel_job_ids: tuple[str, ...]
    projected_geometry_hours: float
    projected_feature_hours: float
    fixed_finalize_hours: float
    projected_total_hours: float
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class Adjacent16FinalGateReport:
    passed: bool
    expected_rows: Mapping[str, int]
    actual_rows: Mapping[str, int]
    controlled_complete_series_coverage: float
    all_four_shoe_primary_cell_rate: float
    minimum_common_support_retention: float
    stress_symmetric_pass_rate: float
    raw_feature_artifacts: tuple[str, ...]
    unexplained_jobs: tuple[str, ...]
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class _RegistryRow:
    job_id: str
    phase: str
    state: str
    payload: Mapping[str, object]
    payload_hash: str
    started_at: str | None
    finished_at: str | None
    output_manifest: Mapping[str, object] | None
    error: str | None


@dataclass(frozen=True)
class _ManifestViewEvidence:
    shoe_id: str
    camera_dataset_id: str
    quartet_id: str
    view_id: str
    orbit_position: int
    source_path: Path
    source_mask_path: Path
    source_sha256: str
    source_mask_sha256: str
    processed_image_correlation: float
    camera_azimuth_deg: float


def _atomic_bytes(content: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _atomic_json(value: Mapping[str, object], destination: Path) -> Path:
    return _atomic_bytes(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        destination,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, temporary)
        readback = pq.read_table(temporary)
        if readback.num_rows != table.num_rows or not readback.schema.equals(table.schema, check_metadata=True):
            raise ValueError("geometry preflight Parquet failed readback validation")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _tree_hash(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _logical_registry_sha256(rows: Sequence[_RegistryRow]) -> str:
    """Hash registry evidence independently of SQLite page layout."""

    logical = [
        {
            "job_id": row.job_id,
            "phase": row.phase,
            "state": row.state,
            "payload_hash": row.payload_hash,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "output_manifest": row.output_manifest,
        }
        for row in rows
    ]
    return hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_contract(root: Path) -> tuple[Adjacent16Contract, str]:
    config_path = root / "config" / "frozen_config.json"
    content = config_path.read_bytes()
    contract = load_adjacent16_contract(config_path)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be an object")
    config_sha256 = hashlib.sha256(content).hexdigest()
    if (
        manifest.get("experiment_id") != contract.experiment_id
        or manifest.get("config_sha256") != config_sha256
    ):
        raise ValueError("run manifest provenance differs from frozen config")
    return contract, config_sha256


def _registry_rows(root: Path) -> tuple[_RegistryRow, ...]:
    path = root / "registry" / "jobs.sqlite"
    if not path.is_file():
        raise ValueError("missing job registry")
    with sqlite3.connect(path) as connection:
        fields = connection.execute("PRAGMA table_info(jobs)").fetchall()
        names = {str(field[1]) for field in fields}
        required = {
            "job_id", "phase", "state", "payload_json", "payload_hash", "started_at",
            "finished_at", "output_manifest_json", "error",
        }
        if not required <= names:
            raise ValueError("job registry schema is incomplete")
        raw_rows = connection.execute(
            """SELECT job_id, phase, state, payload_json, payload_hash, started_at,
                      finished_at, output_manifest_json, error
               FROM jobs ORDER BY job_id"""
        ).fetchall()
    rows: list[_RegistryRow] = []
    for raw in raw_rows:
        try:
            payload = json.loads(raw[3])
            output = json.loads(raw[7]) if raw[7] is not None else None
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("registry JSON is malformed") from error
        if not isinstance(payload, Mapping) or (output is not None and not isinstance(output, Mapping)):
            raise ValueError("registry JSON type is malformed")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != raw[4]:
            raise ValueError("registry payload hash mismatch")
        rows.append(_RegistryRow(
            str(raw[0]), str(raw[1]), str(raw[2]), dict(payload), str(raw[4]),
            str(raw[5]) if raw[5] is not None else None,
            str(raw[6]) if raw[6] is not None else None,
            dict(output) if output is not None else None,
            str(raw[8]) if raw[8] is not None else None,
        ))
    return tuple(rows)


def _valid_output_manifest(row: _RegistryRow, root: Path) -> bool:
    manifest = row.output_manifest
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, list) or not files:
        return False
    seen: set[Path] = set()
    output_dir = row.payload.get("output_dir")
    job_type = row.payload.get("job_type")
    feature_track = row.payload.get("feature_track")
    expected: set[Path] = set()
    if job_type in {"stress_geometry", "geometry_calibration"}:
        if not isinstance(output_dir, str):
            return False
        base = Path(output_dir).resolve()
        expected = {base / "depth_geometry.npz", base / "depth_geometry.json"}
    elif job_type in {"feature", "stress", "geometry_calibration_feature"}:
        track = "geometry_calibration" if job_type == "geometry_calibration_feature" else feature_track
        if not isinstance(track, str):
            return False
        expected = {root / "metrics" / "parts" / track / f"{row.job_id}.parquet"}
    else:
        return False
    for entry in files:
        if not isinstance(entry, Mapping):
            return False
        raw_path, expected_hash = entry.get("path"), entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return False
        path = Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return False
        if path in seen or not path.is_file() or sha256_file(path) != expected_hash:
            return False
        if job_type in {"stress_geometry", "geometry_calibration"}:
            try:
                path.relative_to(base)
            except ValueError:
                return False
        elif path not in expected:
            visual = root / "visualizations" / row.job_id
            try:
                path.relative_to(visual)
            except ValueError:
                return False
        seen.add(path)
    return expected <= seen


def _elapsed_seconds(row: _RegistryRow) -> float:
    if row.started_at is None or row.finished_at is None:
        raise ValueError("job timestamps are missing")
    try:
        start = datetime.fromisoformat(row.started_at)
        finish = datetime.fromisoformat(row.finished_at)
    except ValueError as error:
        raise ValueError("job timestamps are malformed") from error
    if start.tzinfo is None or finish.tzinfo is None:
        raise ValueError("job timestamps must be timezone-aware")
    seconds = (finish - start).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("job duration must be finite and positive")
    return seconds


def _canonical_source_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError(f"{label} must be an absolute canonical path")
    if not path.is_file():
        raise ValueError(f"{label} does not exist")
    return path


def _source_manifest(
    root: Path,
    contract: Adjacent16Contract,
) -> Mapping[tuple[str, str], tuple[_ManifestViewEvidence, ...]]:
    path = root / "manifests" / "adjacent16_manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or frozenset(raw) != _ADJACENT_MANIFEST_KEYS:
        raise ValueError("adjacent source manifest has an unexpected top-level schema")
    if (
        raw.get("reconstruction_root") != str(contract.reconstruction_root)
        or raw.get("stress_root") != str(contract.stress_root)
    ):
        raise ValueError("adjacent source manifest roots differ from the frozen contract")
    raw_shoes = raw.get("shoes")
    expected_shoes = contract.raw.get("shoes")
    if not isinstance(raw_shoes, list) or not isinstance(expected_shoes, Mapping):
        raise ValueError("adjacent source manifest shoes are malformed")
    if len(raw_shoes) != len(expected_shoes):
        raise ValueError("adjacent source manifest shoe count differs")

    orbit_positions = {
        view_id: position for position, view_id in enumerate(contract.camera_order)
    }
    expected_views = tuple(
        (quartet.quartet_id, view_id)
        for quartet in contract.quartets
        for view_id in quartet.view_ids
    )
    by_unit: dict[tuple[str, str], tuple[_ManifestViewEvidence, ...]] = {}
    source_paths: set[Path] = set()
    source_mask_paths: set[Path] = set()
    source_hashes: set[str] = set()
    source_mask_hashes: set[str] = set()
    source_size = contract.raw.get("source_size")
    minimum_correlation = (
        contract.raw.get("preflight", {}).get("minimum_processed_image_correlation")
        if isinstance(contract.raw.get("preflight"), Mapping)
        else None
    )
    if not isinstance(source_size, int) or not isinstance(
        minimum_correlation, (int, float)
    ):
        raise ValueError("frozen source validation thresholds are malformed")

    for raw_shoe, (shoe_id, camera_dataset_id) in zip(
        raw_shoes, expected_shoes.items(), strict=True
    ):
        if (
            not isinstance(raw_shoe, Mapping)
            or frozenset(raw_shoe) != _ADJACENT_SHOE_KEYS
            or raw_shoe.get("shoe_id") != shoe_id
            or raw_shoe.get("camera_dataset_id") != camera_dataset_id
        ):
            raise ValueError("adjacent source shoe record differs from the contract")
        raw_views = raw_shoe.get("views")
        if not isinstance(raw_views, list) or len(raw_views) != len(expected_views):
            raise ValueError("adjacent source shoe views are malformed")
        grouped: dict[str, list[_ManifestViewEvidence]] = defaultdict(list)
        for raw_view, (quartet_id, view_id) in zip(
            raw_views, expected_views, strict=True
        ):
            if (
                not isinstance(raw_view, Mapping)
                or frozenset(raw_view) != _ADJACENT_VIEW_KEYS
                or raw_view.get("shoe_id") != shoe_id
                or raw_view.get("camera_dataset_id") != camera_dataset_id
                or raw_view.get("quartet_id") != quartet_id
                or raw_view.get("view_id") != view_id
            ):
                raise ValueError("adjacent source view record differs from the contract")
            orbit_position = raw_view.get("orbit_position")
            if (
                isinstance(orbit_position, bool)
                or not isinstance(orbit_position, int)
                or orbit_position != orbit_positions[view_id]
            ):
                raise ValueError("adjacent source orbit position differs from the contract")
            correlation = raw_view.get("processed_image_correlation")
            azimuth = raw_view.get("camera_azimuth_deg")
            if (
                isinstance(correlation, bool)
                or not isinstance(correlation, (int, float))
                or not math.isfinite(float(correlation))
                or float(correlation) < float(minimum_correlation)
                or isinstance(azimuth, bool)
                or not isinstance(azimuth, (int, float))
                or not math.isfinite(float(azimuth))
                or not 0.0 <= float(azimuth) < 360.0
            ):
                raise ValueError("adjacent source camera evidence is malformed")
            source_path = _canonical_source_path(
                raw_view.get("source_path"), label="source RGB"
            )
            source_mask_path = _canonical_source_path(
                raw_view.get("source_mask_path"), label="source mask"
            )
            source_sha256 = raw_view.get("source_sha256")
            source_mask_sha256 = raw_view.get("source_mask_sha256")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or sha256_file(source_path) != source_sha256
                or not isinstance(source_mask_sha256, str)
                or len(source_mask_sha256) != 64
                or sha256_file(source_mask_path) != source_mask_sha256
            ):
                raise ValueError("adjacent source content hash validation failed")
            with Image.open(source_path) as image:
                if image.mode != "RGB" or image.size != (source_size, source_size):
                    raise ValueError("adjacent source RGB dimensions or mode differ")
            with Image.open(source_mask_path) as image:
                if image.size != (source_size, source_size):
                    raise ValueError("adjacent source mask dimensions differ")
            evidence = _ManifestViewEvidence(
                shoe_id=shoe_id,
                camera_dataset_id=str(camera_dataset_id),
                quartet_id=quartet_id,
                view_id=view_id,
                orbit_position=orbit_position,
                source_path=source_path,
                source_mask_path=source_mask_path,
                source_sha256=source_sha256,
                source_mask_sha256=source_mask_sha256,
                processed_image_correlation=float(correlation),
                camera_azimuth_deg=float(azimuth),
            )
            grouped[quartet_id].append(evidence)
            source_paths.add(source_path)
            source_mask_paths.add(source_mask_path)
            source_hashes.add(source_sha256)
            source_mask_hashes.add(source_mask_sha256)
        for quartet in contract.quartets:
            records = tuple(grouped.get(quartet.quartet_id, ()))
            if tuple(record.view_id for record in records) != quartet.view_ids:
                raise ValueError("adjacent source quartet order differs")
            by_unit[shoe_id, quartet.quartet_id] = records

    expected_source_count = len(expected_shoes) * len(expected_views)
    if not all(
        len(values) == expected_source_count
        for values in (
            source_paths,
            source_mask_paths,
            source_hashes,
            source_mask_hashes,
        )
    ):
        raise ValueError("adjacent source paths and hashes must be globally unique")
    return by_unit


def _camera_arrays(
    root: Path,
    row: _RegistryRow,
    records: Sequence[_ManifestViewEvidence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    raw_camera = row.payload.get("camera_path")
    if not isinstance(raw_camera, str):
        raise ValueError("geometry job has no camera path")
    camera_path = Path(raw_camera).resolve()
    if len(records) != 4:
        raise ValueError("camera source binding requires four manifest records")
    expected_camera_path = (
        root
        / "cameras"
        / records[0].shoe_id
        / records[0].quartet_id
        / "cameras.npz"
    )
    if camera_path != expected_camera_path:
        raise ValueError("geometry camera path differs from the source manifest unit")
    camera_root = camera_path.parent
    if {
        path.name for path in camera_root.iterdir() if path.is_file()
    } != {"cameras.npz", "images.json", "metadata.json"}:
        raise ValueError("camera artifact file set is not exact")
    metadata = json.loads((camera_root / "metadata.json").read_text(encoding="utf-8"))
    images = json.loads((camera_root / "images.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or not isinstance(images, list) or len(images) != 4:
        raise ValueError("camera metadata is malformed")
    with np.load(camera_path, allow_pickle=False) as camera:
        required = {"extrinsics", "intrinsics", "mask", "view_ids", "camera_groups"}
        if set(camera.files) != required:
            raise ValueError("camera artifact set differs from frozen contract")
        extrinsics = np.asarray(camera["extrinsics"])
        intrinsics = np.asarray(camera["intrinsics"])
        camera_mask = np.asarray(camera["mask"], dtype=bool)
        view_ids = tuple(str(item) for item in np.asarray(camera["view_ids"]).tolist())
        groups = tuple(str(item) for item in np.asarray(camera["camera_groups"]).tolist())
    if (
        extrinsics.shape != (4, 4, 4) or intrinsics.shape != (4, 3, 3)
        or camera_mask.shape != (4, 504, 504)
        or len(view_ids) != 4 or len(set(view_ids)) != 4 or groups != ("upper",) * 4
        or not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all()
    ):
        raise ValueError("camera artifact is malformed")
    source_hashes = metadata.get("source_hashes")
    expected_views = tuple(record.view_id for record in records)
    if (
        metadata.get("camera_mode") != CAMERA_MODE_LABEL
        or metadata.get("camera_provenance") != "dense_camera_outputs"
        or metadata.get("camera_groups") != ["upper"] * 4
        or metadata.get("view_ids") != list(view_ids)
        or not isinstance(source_hashes, Mapping)
        or frozenset(source_hashes) != frozenset(expected_views)
        or view_ids != expected_views
        or images != [str(record.source_path) for record in records]
    ):
        raise ValueError("camera metadata provenance differs from frozen artifact")
    expected_camera_masks: list[np.ndarray] = []
    for manifest_record, raw_image in zip(records, images, strict=True):
        record = source_hashes.get(manifest_record.view_id)
        if (
            not isinstance(raw_image, str) or not isinstance(record, Mapping)
            or frozenset(record) != {"rgb", "mask"}
            or record.get("rgb") != manifest_record.source_sha256
            or record.get("mask") != manifest_record.source_mask_sha256
        ):
            raise ValueError("camera source hash validation failed")
        with Image.open(manifest_record.source_mask_path) as image:
            expected_camera_masks.append(
                np.asarray(
                    image.convert("L").resize(
                        (504, 504), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
    if not np.array_equal(camera_mask, np.stack(expected_camera_masks)):
        raise ValueError("camera masks differ from resized source masks")
    derived_azimuths = camera_azimuths(extrinsics)
    expected_azimuths = np.asarray(
        [record.camera_azimuth_deg for record in records], dtype=np.float64
    )
    azimuth_error = np.abs(
        (derived_azimuths - expected_azimuths + 180.0) % 360.0 - 180.0
    )
    if (
        not np.all(azimuth_error <= 1e-5)
        or len({tuple(np.asarray(pose).reshape(-1)) for pose in extrinsics}) != 4
    ):
        raise ValueError("camera extrinsics differ from manifest azimuth evidence")
    raw_output = row.payload.get("output_dir")
    if not isinstance(raw_output, str):
        raise ValueError("geometry job has no output path")
    output = Path(raw_output).resolve()
    try:
        output.relative_to(root / "geometry")
    except ValueError as error:
        raise ValueError("geometry output path escapes run root") from error
    with np.load(output / "depth_geometry.npz", allow_pickle=False) as geometry:
        required = {
            "depth", "confidence", "mask", "extrinsics", "intrinsics",
            "meta__artifact_type", "meta__camera_provenance", "meta__view_ids",
            "meta__camera_groups",
        }
        if not required <= set(geometry.files):
            raise ValueError("geometry artifact is incomplete")
        depth = np.asarray(geometry["depth"])
        confidence = np.asarray(geometry["confidence"])
        mask = np.asarray(geometry["mask"], dtype=bool)
        artifact_extrinsics = np.asarray(geometry["extrinsics"])
        artifact_intrinsics = np.asarray(geometry["intrinsics"])
        artifact_views = tuple(str(item) for item in np.asarray(geometry["meta__view_ids"]).tolist())
        artifact_groups = tuple(str(item) for item in np.asarray(geometry["meta__camera_groups"]).tolist())
        artifact_type = str(np.asarray(geometry["meta__artifact_type"]).item())
        provenance = str(np.asarray(geometry["meta__camera_provenance"]).item())
    if (
        depth.shape != (4, 504, 504) or confidence.shape != depth.shape or mask.shape != depth.shape
        or not np.isfinite(depth).all() or (depth <= 0).any()
        or not np.isfinite(confidence).all() or not np.array_equal(mask, camera_mask)
        or not np.array_equal(artifact_extrinsics, extrinsics)
        or not np.array_equal(artifact_intrinsics, intrinsics)
        or artifact_views != view_ids or artifact_groups != ("upper",) * 4
        or artifact_type != "fixed_camera_depth"
        or provenance != "fixed_precomputed_da3"
    ):
        raise ValueError("geometry/camera provenance validation failed")
    return depth, confidence, mask, intrinsics, extrinsics, view_ids


def _median_masked(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[0][mask[0]]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.detach().float().median().cpu().item())


def _confidence_mask(confidence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = confidence[mask]
    if selected.numel() == 0 or not torch.isfinite(selected).all():
        raise ValueError("geometry confidence evidence is empty or non-finite")
    retained, _ = confidence_percentile_mask(mask.unsqueeze(0), confidence.unsqueeze(0), percentile=10.0)
    retained = retained[0]
    if not retained.any():
        raise ValueError("confidence percentile removed all foreground evidence")
    return retained


def _score_preflight_pair(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    first_index: int,
    second_index: int,
) -> Mapping[str, float | bool]:
    """Score one confidence-filtered geometry pair under the frozen 10th percentile."""

    first_mask = _confidence_mask(confidence[first_index], mask[first_index]).unsqueeze(0)
    second_mask = _confidence_mask(confidence[second_index], mask[second_index]).unsqueeze(0)
    support = score_symmetric_geometry_support(
        depth[first_index : first_index + 1].float(), depth[second_index : second_index + 1].float(),
        intrinsics[first_index : first_index + 1].float(), intrinsics[second_index : second_index + 1].float(),
        extrinsics[first_index : first_index + 1].float(), extrinsics[second_index : second_index + 1].float(),
        first_mask=first_mask, second_mask=second_mask,
        config=KnownCameraConfig(min_overlap_fraction=0.08),
    )
    forward = float(support.first_to_second.overlap_fraction.item())
    backward = float(support.second_to_first.overlap_fraction.item())
    minimum = min(forward, backward)
    return {
        "first_to_second_overlap": forward,
        "second_to_first_overlap": backward,
        "minimum_directional_overlap": minimum,
        "overlap_asymmetry": abs(forward - backward),
        "common_valid_fraction": minimum,
        "cycle_error_px_median": _median_masked(support.first_to_second.cycle_error_px, support.first_to_second.valid_mask),
        "relative_depth_error_median": _median_masked(support.first_to_second.relative_depth_error, support.first_to_second.valid_mask),
        "symmetric_pass": bool(not support.abstained.item() and minimum >= 0.08),
    }


def _clean_pair_metadata(
    root: Path,
    contract: Adjacent16Contract,
    source_manifest: Mapping[
        tuple[str, str], tuple[_ManifestViewEvidence, ...]
    ],
) -> Mapping[tuple[str, str, str], tuple[int, float]]:
    records: dict[tuple[str, str, str], tuple[int, float]] = {}
    expected: dict[tuple[str, str, str], tuple[str, str, int, float]] = {}
    for shoe in contract.raw["shoes"]:
        for quartet in contract.quartets:
            manifest_views = source_manifest.get((shoe, quartet.quartet_id))
            if manifest_views is None:
                raise ValueError("source manifest is missing a clean pair unit")
            for first, second in itertools.combinations(manifest_views, 2):
                raw_steps = abs(first.orbit_position - second.orbit_position)
                steps = min(raw_steps, len(contract.camera_order) - raw_steps)
                raw_angle = abs(
                    first.camera_azimuth_deg - second.camera_azimuth_deg
                )
                angle = min(raw_angle, 360.0 - raw_angle)
                pair_id = (
                    f"{shoe}__{quartet.quartet_id}__"
                    f"{first.view_id}__{second.view_id}"
                )
                expected[shoe, quartet.quartet_id, pair_id] = (
                    first.view_id,
                    second.view_id,
                    steps,
                    angle,
                )
    for line in (root / "manifests" / "pair_conditions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or row.get("track") != "controlled" or row.get("corruption") != "clean":
            continue
        key = tuple(row.get(field) for field in ("shoe_id", "quartet_id", "pair_id"))
        steps, angle = row.get("baseline_steps"), row.get("angular_separation_deg")
        expected_row = expected.get(key)
        if (
            expected_row is None
            or key in records
            or row.get("first_view") != expected_row[0]
            or row.get("second_view") != expected_row[1]
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps != expected_row[2]
            or isinstance(angle, bool) or not isinstance(angle, (int, float))
            or not math.isfinite(float(angle))
            or not math.isclose(
                float(angle), expected_row[3], rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError("clean pair manifest is malformed")
        records[key] = (steps, float(angle))
    if set(records) != set(expected):
        raise ValueError("clean pair manifest is incomplete or contains extra rows")
    step_counts = {
        step: sum(1 for value in records.values() if value[0] == step)
        for step in (1, 2, 3)
    }
    if step_counts != {1: 48, 2: 32, 3: 16}:
        raise ValueError("clean pair manifest step distribution differs")
    return records


def validate_adjacent16_geometry_preflight(run_root: str | Path) -> GeometryPreflightReport:
    """Score and validate all 96 clean geometry pair units."""

    root = Path(run_root).expanduser().resolve()
    failures: set[str] = set()
    rows: list[dict[str, object]] = []
    per_shoe: dict[str, float] = {}
    input_hash: str | None = None
    output_hash: str | None = None
    try:
        contract, config_sha256 = _load_contract(root)
        registry = _registry_rows(root)
        source_manifest = _source_manifest(root, contract)
        pair_metadata = _clean_pair_metadata(root, contract, source_manifest)
        clean_jobs = [
            row for row in registry
            if row.phase == "geometry" and row.payload.get("job_type") == "geometry_calibration"
        ]
        expected_units = {
            (shoe_id, quartet.quartet_id)
            for shoe_id in contract.raw["shoes"] for quartet in contract.quartets
        }
        by_unit: dict[tuple[str, str], _RegistryRow] = {}
        for row in clean_jobs:
            key = (row.payload.get("shoe_id"), row.payload.get("quartet_id"))
            if key in by_unit or key not in expected_units:
                raise ValueError("clean geometry registry units are duplicate or unexpected")
            if row.payload.get("config_sha256") != config_sha256 or row.state != "completed":
                raise ValueError("clean geometry registry validation failed")
            if not _valid_output_manifest(row, root):
                raise ValueError("clean geometry output hash validation failed")
            by_unit[key] = row
        if set(by_unit) != expected_units:
            raise ValueError("clean geometry registry units are incomplete")
        for (shoe_id, quartet_id), row in sorted(by_unit.items()):
            manifest_records = source_manifest.get((shoe_id, quartet_id))
            if manifest_records is None:
                raise ValueError("geometry unit is absent from source manifest")
            depth, confidence, mask, intrinsics, extrinsics, views = _camera_arrays(
                root, row, manifest_records
            )
            expected_views = next(item.view_ids for item in contract.quartets if item.quartet_id == quartet_id)
            if views != expected_views:
                raise ValueError("camera view order differs from frozen quartet")
            for first_index, second_index in itertools.combinations(range(4), 2):
                scored = _score_preflight_pair(
                    torch.from_numpy(depth), torch.from_numpy(confidence), torch.from_numpy(mask),
                    torch.from_numpy(intrinsics), torch.from_numpy(extrinsics), first_index, second_index,
                )
                first_view, second_view = views[first_index], views[second_index]
                pair_id = f"{shoe_id}__{quartet_id}__{first_view}__{second_view}"
                steps, angle = pair_metadata[shoe_id, quartet_id, pair_id]
                values = tuple(float(scored[name]) for name in ("first_to_second_overlap", "second_to_first_overlap", "minimum_directional_overlap", "cycle_error_px_median", "relative_depth_error_median"))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("preflight overlap is non-finite")
                rows.append({
                    "shoe_id": shoe_id, "quartet_id": quartet_id,
                    "pair_id": pair_id,
                    "first_view": first_view, "second_view": second_view,
                    "baseline_steps": steps, "angular_separation_deg": angle,
                    **scored,
                })
        if len(rows) != 96 or set(rows[0]) != set(_PREFLIGHT_COLUMNS):
            raise ValueError("geometry preflight table does not contain exactly 96 canonical rows")
        table = pd.DataFrame(rows).loc[:, list(_PREFLIGHT_COLUMNS)]
        numeric_columns = (
            "baseline_steps", "angular_separation_deg", "first_to_second_overlap",
            "second_to_first_overlap", "minimum_directional_overlap", "overlap_asymmetry",
            "common_valid_fraction", "cycle_error_px_median", "relative_depth_error_median",
        )
        if (
            table.duplicated(["shoe_id", "quartet_id", "pair_id"]).any()
            or not np.isfinite(table.loc[:, numeric_columns].to_numpy(dtype=float)).all()
        ):
            raise ValueError("geometry preflight table contains duplicate or non-finite evidence")
        _atomic_parquet(table, root / "reports" / "geometry_preflight.parquet")
        artifact_hash = _tree_hash(root, [
            root / "config" / "frozen_config.json", root / "run_manifest.json",
            root / "manifests" / "adjacent16_manifest.json",
            root / "manifests" / "pair_conditions.jsonl",
            *(root / "cameras").glob("*/*/*"), *(root / "geometry").glob("**/depth_geometry.npz"),
        ])
        registry_hash = hashlib.sha256(json.dumps([
            {"job_id": item.job_id, "phase": item.phase, "state": item.state,
             "payload_hash": item.payload_hash, "output_manifest": item.output_manifest}
            for item in registry
        ], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        input_hash = hashlib.sha256(f"{artifact_hash}:{registry_hash}".encode("utf-8")).hexdigest()
        output_hash = sha256_file(root / "reports" / "geometry_preflight.parquet")
        symmetric_rate = float(table["symmetric_pass"].mean())
        median_overlap = float(table["minimum_directional_overlap"].median())
        per_shoe = {str(shoe): float(group["symmetric_pass"].mean()) for shoe, group in table.groupby("shoe_id", sort=True)}
        if symmetric_rate < 0.90:
            failures.add("symmetric_pass_rate")
        if median_overlap < 0.20:
            failures.add("median_directional_overlap")
        if any(rate < 0.80 for rate in per_shoe.values()):
            failures.add("per_shoe_pair_coverage")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, sqlite3.Error, pa.ArrowException):
        failures.add("input_validation")
        if not (root / "registry" / "jobs.sqlite").is_file():
            failures.add("registry_counts")
        symmetric_rate = float("nan")
        median_overlap = float("nan")
        per_shoe = {}
        rows = []
    report = GeometryPreflightReport(
        passed=not failures, pair_units=len(rows), symmetric_pass_rate=symmetric_rate,
        median_min_directional_overlap=median_overlap, per_shoe_pass_rate=per_shoe,
        failed_checks=tuple(sorted(failures)),
    )
    _atomic_json(_json_safe({**asdict(report), "config_sha256": input_hash and config_sha256 if "config_sha256" in locals() else None, "input_sha256": input_hash, "output_sha256": output_hash}), root / "reports" / "geometry_preflight.json")
    return report


def _state_index(root: Path) -> Mapping[str, Mapping[str, object]]:
    path = root / "manifests" / "image_states.jsonl"
    if not path.is_file():
        return {}
    records: dict[str, Mapping[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or not isinstance(value.get("state_id"), str) or value["state_id"] in records:
            raise ValueError("image state manifest is malformed")
        records[str(value["state_id"])] = dict(value)
    return records


def _sentinel_class(row: _RegistryRow, states: Mapping[str, Mapping[str, object]]) -> str:
    job_type = row.payload.get("job_type")
    if job_type == "stress":
        return "stress"
    if job_type == "geometry_calibration_feature":
        return "calibration"
    if job_type == "feature":
        state_id = row.payload.get("state_id")
        state = states.get(state_id) if isinstance(state_id, str) else None
        corruption = state.get("corruption") if state is not None else row.payload.get("corruption")
        return "clean" if corruption == "clean" else "changed"
    raise ValueError("sentinel job has no frozen feature class")


def _sentinel_jobs(root: Path, contract: Adjacent16Contract, rows: Sequence[_RegistryRow]) -> tuple[_RegistryRow, ...]:
    shoes = tuple(contract.raw["shoes"])
    if not shoes:
        raise ValueError("frozen contract has no shoes")
    shoe = shoes[0]
    states = _state_index(root)

    def field(row: _RegistryRow, name: str) -> object:
        state_id = row.payload.get("state_id")
        state = states.get(state_id) if isinstance(state_id, str) else None
        if state is not None and name in {"track", "corruption", "severity", "changed_view", "shoe_id", "quartet_id"}:
            return state.get(name)
        return row.payload.get(name)

    backbones = tuple(dict.fromkeys(variant.backbone for variant in contract.feature_variants))
    selected: list[_RegistryRow] = []
    for backbone in backbones:
        predicates = (
            ("clean", {"job_type": "feature", "track": "controlled", "corruption": "clean"}),
            ("changed", {"job_type": "feature", "track": "controlled", "corruption": "translation", "severity": 0.2, "changed_view": "1"}),
            ("stress", {"job_type": "stress", "track": "stress"}),
        )
        for expected_class, predicate in predicates:
            matches = [row for row in rows if row.phase == "feature" and row.payload.get("backbone") == backbone and row.payload.get("shoe_id") == shoe and row.payload.get("quartet_id") == "q0" and all(field(row, key) == value for key, value in predicate.items())]
            if len(matches) != 1 or _sentinel_class(matches[0], states) != expected_class:
                raise ValueError("frozen timed sentinel job is missing or ambiguous")
            selected.append(matches[0])
    calibration = [row for row in rows if row.phase == "feature" and row.payload.get("backbone") == "dinov3_vitl16_pixal3d" and row.payload.get("shoe_id") == shoe and row.payload.get("quartet_id") == "q0" and row.payload.get("job_type") == "geometry_calibration_feature"]
    if len(calibration) != 1 or _sentinel_class(calibration[0], states) != "calibration":
        raise ValueError("frozen calibration sentinel job is missing or ambiguous")
    selected.append(calibration[0])
    if len(selected) != 13 or len({row.job_id for row in selected}) != 13:
        raise ValueError("timed sentinel must contain exactly thirteen distinct jobs")
    return tuple(selected)


def write_timed_sentinel_manifest(run_root: str | Path) -> Path:
    """Write exact per-backbone job-ID files for the 13 sentinel jobs."""

    root = Path(run_root).expanduser().resolve()
    contract, config_sha256 = _load_contract(root)
    selected = _sentinel_jobs(root, contract, _registry_rows(root))
    destination = root / "manifests" / "timed_sentinel"
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        grouped[str(row.payload["backbone"])].append(row.job_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    hashes: dict[str, str] = {}
    for backbone, job_ids in sorted(grouped.items()):
        path = staging / f"{backbone}.json"
        _atomic_json({"backbone": backbone, "config_sha256": config_sha256, "job_ids": job_ids}, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload != {"backbone": backbone, "config_sha256": config_sha256, "job_ids": job_ids}:
            shutil.rmtree(staging)
            raise ValueError("staged timed sentinel manifest failed readback")
        hashes[path.name] = sha256_file(path)
    expected_names = {f"{backbone}.json" for backbone in grouped}
    if {path.name for path in staging.iterdir()} != expected_names:
        shutil.rmtree(staging)
        raise ValueError("staged timed sentinel manifest file set is incomplete")
    backup: Path | None = None
    try:
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.backup.{uuid.uuid4().hex}")
            destination.rename(backup)
        staging.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            backup.rename(destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    manifest_paths = tuple(sorted(destination.glob("*.json")))
    manifest_sha256 = _tree_hash(root, manifest_paths)
    registry_sha256 = _logical_registry_sha256(_registry_rows(root))
    image_states = root / "manifests" / "image_states.jsonl"
    input_sha256 = hashlib.sha256(json.dumps({
        "config_sha256": config_sha256,
        "registry_sha256": registry_sha256,
        "image_states_sha256": sha256_file(image_states) if image_states.is_file() else None,
        "sentinel_manifest_sha256": manifest_sha256,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report_payload = {
        "config_sha256": config_sha256,
        "registry_sha256": registry_sha256,
        "sentinel_job_ids": [row.job_id for row in selected],
        "files": hashes,
        "input_sha256": input_sha256,
        "output_sha256": manifest_sha256,
    }
    report_path = root / "reports" / "timed_sentinel_manifest.json"
    _atomic_json(report_payload, report_path)
    if json.loads(report_path.read_text(encoding="utf-8")) != report_payload:
        raise RuntimeError("timed sentinel manifest report readback differs from persisted decision")
    return destination


def validate_timed_runtime_projection(run_root: str | Path) -> RuntimeProjectionReport:
    """Project four-GPU wall time from completed sentinel durations."""

    root = Path(run_root).expanduser().resolve()
    failures: set[str] = set()
    selected: tuple[_RegistryRow, ...] = ()
    geometry_hours = feature_hours = total_hours = float("nan")
    fixed_hours = 2.0
    class_seconds: dict[tuple[str, str], float] = {}
    try:
        contract, config_sha256 = _load_contract(root)
        rows = _registry_rows(root)
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.phase] += 1
        if dict(counts) != _EXPECTED_REGISTRY_COUNTS or len(rows) != sum(_EXPECTED_REGISTRY_COUNTS.values()) or not _validate_registry_graph(root, contract, config_sha256, rows):
            failures.add("runtime_graph")
            raise ValueError("runtime registry differs from frozen graph")
        selected = _sentinel_jobs(root, contract, rows)
        if any(row.state != "completed" or not _valid_output_manifest(row, root) for row in selected):
            failures.add("sentinel_job_state")
            raise ValueError("sentinel jobs must all be completed")
        states = _state_index(root)
        for row in selected:
            class_seconds[str(row.payload["backbone"]), _sentinel_class(row, states)] = _elapsed_seconds(row)
        if len(class_seconds) != 13:
            raise ValueError("sentinel durations are unclassified or duplicated")
        geometry = [row for row in rows if row.phase == "geometry"]
        if len(geometry) != _EXPECTED_REGISTRY_COUNTS["geometry"] or any(row.state != "completed" or not _valid_output_manifest(row, root) for row in geometry):
            failures.add("geometry_job_state")
            raise ValueError("all frozen geometry jobs must complete before projection")
        for row in geometry:
            _elapsed_seconds(row)
        starts = [datetime.fromisoformat(str(row.started_at)) for row in geometry]
        finishes = [datetime.fromisoformat(str(row.finished_at)) for row in geometry]
        if any(item.tzinfo is None for item in (*starts, *finishes)):
            raise ValueError("geometry timestamps must be timezone-aware")
        geometry_seconds = (max(finishes) - min(starts)).total_seconds()
        if not math.isfinite(geometry_seconds) or geometry_seconds <= 0:
            raise ValueError("geometry wall duration must be finite and positive")
        geometry_hours = geometry_seconds / 3600.0
        assignments: dict[int, float] = {gpu: 0.0 for gpu in contract.physical_gpus}
        backbones = tuple(dict.fromkeys(variant.backbone for variant in contract.feature_variants))
        if len(backbones) != 4 or len(assignments) != 4:
            raise ValueError("frozen projection requires exactly four backbone GPUs")
        for gpu, backbone in zip(contract.physical_gpus, backbones, strict=True):
            for feature_class, count in _FEATURE_CLASS_COUNTS[backbone].items():
                if count == 0:
                    continue
                sampled = class_seconds.get((backbone, feature_class))
                if sampled is None or not math.isfinite(sampled) or sampled <= 0:
                    raise ValueError("sentinel class duration is missing or invalid")
                assignments[gpu] += count * sampled
        feature_hours = max(assignments.values()) / 3600.0
        fixed_hours = float(contract.raw["runtime"]["fixed_finalize_hours"])
        total_hours = geometry_hours + float(contract.raw["runtime"]["feature_wall_time_multiplier"]) * feature_hours + fixed_hours
        if total_hours > float(contract.raw["runtime"]["maximum_projected_hours"]):
            failures.add("projected_total_hours")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, sqlite3.Error):
        if not failures:
            failures.add("runtime_validation")
        selected = ()
    report = RuntimeProjectionReport(
        passed=not failures, sentinel_job_ids=tuple(row.job_id for row in selected),
        projected_geometry_hours=geometry_hours, projected_feature_hours=feature_hours,
        fixed_finalize_hours=fixed_hours, projected_total_hours=total_hours,
        failed_checks=tuple(sorted(failures)),
    )
    decision = _json_safe({
        **asdict(report),
        "sampled_class_seconds": {
            f"{backbone}/{kind}": seconds
            for (backbone, kind), seconds in sorted(class_seconds.items())
        },
    })
    provenance: dict[str, object] = {
        "config_sha256": None,
        "registry_sha256": None,
        "timed_sentinel_manifest_sha256": None,
        "input_sha256": None,
        "output_sha256": None,
    }
    try:
        config_path = root / "config" / "frozen_config.json"
        rows = _registry_rows(root)
        sentinel_report = root / "reports" / "timed_sentinel_manifest.json"
        if not sentinel_report.is_file():
            write_timed_sentinel_manifest(root)
        provenance["config_sha256"] = sha256_file(config_path)
        provenance["registry_sha256"] = _logical_registry_sha256(rows)
        provenance["timed_sentinel_manifest_sha256"] = sha256_file(sentinel_report)
        provenance["input_sha256"] = hashlib.sha256(json.dumps({
            "config_sha256": provenance["config_sha256"],
            "registry_sha256": provenance["registry_sha256"],
            "timed_sentinel_manifest_sha256": provenance["timed_sentinel_manifest_sha256"],
            "image_states_sha256": (
                sha256_file(root / "manifests" / "image_states.jsonl")
                if (root / "manifests" / "image_states.jsonl").is_file() else None
            ),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        provenance["output_sha256"] = hashlib.sha256(
            json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    except (OSError, ValueError, TypeError, json.JSONDecodeError, sqlite3.Error):
        failures.add("runtime_report_provenance")
        report = RuntimeProjectionReport(**{
            **asdict(report), "passed": False, "failed_checks": tuple(sorted(failures)),
        })
        decision = _json_safe({
            **asdict(report),
            "sampled_class_seconds": {
                f"{backbone}/{kind}": seconds
                for (backbone, kind), seconds in sorted(class_seconds.items())
            },
        })
        provenance["output_sha256"] = hashlib.sha256(
            json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    payload = {**decision, **provenance}
    destination = root / "reports" / "runtime_projection.json"
    _atomic_json(payload, destination)
    if json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("runtime projection report readback differs from persisted decision")
    return report


def _raw_feature_artifacts(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        relative = path.relative_to(root)
        if (
            relative.parts and relative.parts[0] == "features"
        ) or lower in {"features.npz", "features.npy", "embedding.npy"} or path.suffix.lower() in {
            ".pt", ".pth", ".npy", ".safetensors", ".bin",
        }:
            found.append(str(relative))
            continue
        if path.suffix.lower() == ".npz":
            camera_artifact = (
                len(relative.parts) == 4
                and relative.parts[0] == "cameras"
                and path.name == "cameras.npz"
            )
            geometry_artifact = (
                bool(relative.parts)
                and relative.parts[0] == "geometry"
                and path.name in {"depth_geometry.npz", "geometry.npz"}
            )
            if not camera_artifact and not geometry_artifact:
                found.append(str(relative))
                continue
            try:
                with np.load(path, allow_pickle=False) as artifact:
                    keys = frozenset(artifact.files)
                    if camera_artifact and keys != _CAMERA_NPZ_KEYS:
                        found.append(str(relative))
                    elif geometry_artifact and (
                        keys != _FIXED_GEOMETRY_NPZ_KEYS
                        or str(np.asarray(artifact["meta__artifact_type"]).item())
                        != "fixed_camera_depth"
                    ):
                        found.append(str(relative))
                    elif any(
                        key in artifact.files
                        and str(np.asarray(artifact[key]).item()) == "dino_features"
                        for key in ("meta__artifact_type", "artifact_type")
                    ):
                        found.append(str(path.relative_to(root)))
            except (OSError, ValueError):
                found.append(str(relative))
    return tuple(sorted(set(found)))


def _finite_primary(row: Mapping[str, object]) -> bool:
    if bool(row.get("abstained", False)):
        return False
    fields = (
        "score", "s00", "s10", "s01", "s11",
        "s11_first_to_second_overlap", "s11_second_to_first_overlap",
    )
    try:
        return all(math.isfinite(float(row[field])) for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def _read_track_frames(root: Path) -> tuple[dict[str, pd.DataFrame], set[str]]:
    frames: dict[str, pd.DataFrame] = {}
    failures: set[str] = set()
    for track, filename in _TRACK_FILES.items():
        path = root / "metrics" / filename
        if not path.is_file():
            failures.add("metric_tables")
            frames[track] = pd.DataFrame()
            continue
        try:
            frames[track] = pd.read_parquet(path)
        except (OSError, ValueError, pa.ArrowException):
            failures.add("metric_tables")
            frames[track] = pd.DataFrame()
    return frames, failures


def _expected_controlled_series(
    root: Path,
    contract: Adjacent16Contract,
) -> tuple[set[tuple[str, str, str, str, str, str]], set[tuple[str, str, str, str, str, str, float]]]:
    conditions_path = root / "manifests" / "pair_conditions.jsonl"
    variants_path = root / "manifests" / "feature_variants.json"
    conditions = [json.loads(line) for line in conditions_path.read_text(encoding="utf-8").splitlines() if line]
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    if not isinstance(variants, list) or not all(
        isinstance(item, Mapping)
        and all(isinstance(item.get(field), (str, int)) for field in ("feature_track", "backbone", "input_size", "feature_mode", "resolution"))
        for item in variants
    ):
        raise ValueError("feature variant manifest is malformed")
    expected_conditions: set[tuple[object, ...]] = set()
    changed_conditions: set[tuple[object, ...]] = set()
    for shoe in contract.raw["shoes"]:
        for quartet in contract.quartets:
            pairs = tuple(itertools.combinations(quartet.view_ids, 2))
            for first, second in pairs:
                pair_id = f"{shoe}__{quartet.quartet_id}__{first}__{second}"
                expected_conditions.add(("controlled", shoe, quartet.quartet_id, pair_id, first, second, None, "clean", 0.0))
            for corruption, strengths in contract.raw["corruptions"].items():
                for severity in strengths:
                    for changed in quartet.view_ids:
                        for first, second in pairs:
                            if changed not in {first, second}:
                                continue
                            pair_id = f"{shoe}__{quartet.quartet_id}__{first}__{second}"
                            record = ("controlled", shoe, quartet.quartet_id, pair_id, first, second, changed, corruption, float(severity))
                            expected_conditions.add(record)
                            changed_conditions.add(record)
    actual_conditions: set[tuple[object, ...]] = set()
    for condition in conditions:
        if not isinstance(condition, Mapping) or condition.get("track") != "controlled":
            continue
        severity = condition.get("severity")
        if isinstance(severity, bool) or not isinstance(severity, (int, float)):
            raise ValueError("controlled condition severity is malformed")
        actual_conditions.add(tuple(condition.get(field) for field in (
            "track", "shoe_id", "quartet_id", "pair_id", "first_view", "second_view", "changed_view", "corruption",
        )) + (float(severity),))
    controlled_count = sum(isinstance(condition, Mapping) and condition.get("track") == "controlled" for condition in conditions)
    if actual_conditions != expected_conditions or len(actual_conditions) != controlled_count:
        raise ValueError("controlled condition manifest differs from frozen protocol")
    series: set[tuple[str, str, str, str, str, str]] = set()
    rows: set[tuple[str, str, str, str, str, float, str]] = set()
    for _track, shoe, quartet, pair_id, _first, _second, changed, corruption, severity in changed_conditions:
        assert isinstance(shoe, str) and isinstance(quartet, str) and isinstance(pair_id, str)
        assert isinstance(changed, str) and isinstance(corruption, str) and isinstance(severity, float)
        for variant in variants:
            key = "__".join(str(variant[field]) for field in ("feature_track", "backbone", "input_size", "feature_mode", "resolution"))
            series.add((shoe, quartet, pair_id, changed, corruption, key))
            rows.add((shoe, quartet, pair_id, changed, corruption, severity, key))
    variant_keys = {
        "__".join(str(item[field]) for field in ("feature_track", "backbone", "input_size", "feature_mode", "resolution"))
        for item in variants
    }
    frozen_keys = {item.key for item in contract.feature_variants}
    strengths = sum(len(values) for values in contract.raw["corruptions"].values())
    expected_series = (
        len(contract.raw["shoes"]) * len(contract.quartets) * 4 * 3
        * len(contract.raw["corruptions"]) * len(contract.feature_variants)
    )
    expected_rows = (
        len(contract.raw["shoes"]) * len(contract.quartets) * 4 * 3
        * strengths * len(contract.feature_variants)
    )
    if variant_keys != frozen_keys or len(series) != expected_series or len(rows) != expected_rows:
        raise ValueError("controlled series denominator differs from the frozen protocol")
    return series, rows


def _actual_primary_rows(frame: pd.DataFrame) -> dict[tuple[object, ...], Mapping[str, object]]:
    required = {"shoe_id", "quartet_id", "pair_id", "changed_view", "corruption", "severity", "variant_key"}
    if not required <= set(frame):
        raise ValueError("controlled metric table lacks primary fields")
    indexed: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in frame.to_dict("records"):
        key = tuple(row.get(field) for field in ("shoe_id", "quartet_id", "pair_id", "changed_view", "corruption", "severity", "variant_key"))
        if key in indexed:
            raise ValueError("controlled metric table has duplicate primary rows")
        indexed[key] = row
    return indexed


def _frozen_design(root: Path, contract: Adjacent16Contract) -> Adjacent16Design:
    source = _source_manifest(root, contract)
    shoes: list[AdjacentShoeRecord] = []
    for shoe_id, camera_dataset_id in contract.raw["shoes"].items():
        views: list[AdjacentViewRecord] = []
        for quartet in contract.quartets:
            records = source.get((shoe_id, quartet.quartet_id))
            if records is None:
                raise ValueError("source manifest is missing a frozen design unit")
            for record in records:
                views.append(
                    AdjacentViewRecord(
                        shoe_id=record.shoe_id,
                        camera_dataset_id=record.camera_dataset_id,
                        quartet_id=record.quartet_id,
                        view_id=record.view_id,
                        orbit_position=record.orbit_position,
                        source_path=record.source_path,
                        source_mask_path=record.source_mask_path,
                        stress_path=None,
                        stress_mask_path=None,
                        deformation_json_path=None,
                        metrics_json_path=None,
                        task_json_path=None,
                        processed_image_correlation=record.processed_image_correlation,
                        camera_azimuth_deg=record.camera_azimuth_deg,
                        source_sha256=record.source_sha256,
                        source_mask_sha256=record.source_mask_sha256,
                        stress_sha256=None,
                        stress_mask_sha256=None,
                        quality=None,
                    )
                )
        shoes.append(
            AdjacentShoeRecord(shoe_id, str(camera_dataset_id), tuple(views))
        )
    manifest = Adjacent16Manifest(
        contract.reconstruction_root, contract.stress_root, tuple(shoes)
    )
    return build_adjacent16_design(manifest, contract)


def _read_exact_jsonl(path: Path) -> list[object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"manifest JSONL is empty or contains blank rows: {path}")
    return [json.loads(line) for line in lines]


def _validate_registry_graph(root: Path, contract: Adjacent16Contract, config_sha256: str, rows: Sequence[_RegistryRow]) -> bool:
    try:
        design = _frozen_design(root, contract)
        states = [
            json.loads(json.dumps(asdict(state)))
            for state in (*design.controlled_states, *design.stress_states)
        ]
        pairs = [
            json.loads(json.dumps(asdict(condition)))
            for condition in (*design.controlled_conditions, *design.stress_conditions)
        ]
        variants = [asdict(variant) for variant in design.feature_variants]
        if (
            _read_exact_jsonl(root / "manifests" / "image_states.jsonl") != states
            or _read_exact_jsonl(root / "manifests" / "pair_conditions.jsonl") != pairs
            or json.loads(
                (root / "manifests" / "feature_variants.json").read_text(
                    encoding="utf-8"
                )
            )
            != variants
        ):
            return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    variants_by_backbone: dict[str, list[str]] = defaultdict(list)
    for variant in contract.feature_variants:
        variants_by_backbone[variant.backbone].append(variant.key)
    expected: dict[str, tuple[str, Mapping[str, object]]] = {}

    def base(state: Mapping[str, object], analysis_track: str) -> dict[str, object]:
        return {"state_id": state["state_id"], "track": state["track"], "analysis_track": analysis_track, "shoe_id": state["shoe_id"], "quartet_id": state["quartet_id"], "state_dir": str((root / "states" / str(state["state_id"])).resolve()), "camera_path": str((root / "cameras" / str(state["shoe_id"]) / str(state["quartet_id"]) / "cameras.npz").resolve()), "config_sha256": config_sha256}

    for state in states:
        if state["track"] == "controlled":
            for backbone, keys in sorted(variants_by_backbone.items()):
                job_id = f"feature__controlled__{state['state_id']}__{backbone}__user_37_grid_track"
                expected[job_id] = ("feature", {**base(state, "controlled"), "job_type": "feature", "backbone": backbone, "feature_track": "user_37_grid_track", "variant_keys": keys, "geometry_contract": "fixed_base_only", "output_dir": str((root / "features" / "controlled" / str(state["state_id"]) / backbone).resolve())})
        else:
            geometry_id = f"geometry__stress__{state['state_id']}"
            expected[geometry_id] = ("geometry", {**base(state, "stress"), "job_type": "stress_geometry", "geometry_contract": "base_plus_adaptive", "output_dir": str((root / "geometry" / "stress" / str(state["state_id"])).resolve())})
            for backbone, keys in sorted(variants_by_backbone.items()):
                job_id = f"feature__stress__{state['state_id']}__{backbone}__user_37_grid_track"
                expected[job_id] = ("feature", {**base(state, "stress"), "job_type": "stress", "backbone": backbone, "feature_track": "user_37_grid_track", "variant_keys": keys, "geometry_contract": "base_plus_adaptive", "output_dir": str((root / "features" / "stress" / str(state["state_id"]) / backbone).resolve())})
    calibration_keys = [key for key in variants_by_backbone["dinov3_vitl16_pixal3d"] if key in contract.geometry_calibration_variant_keys]
    for state in states:
        if state["track"] != "controlled" or state["corruption"] != "clean":
            continue
        geometry_id = f"geometry__geometry_calibration__{state['state_id']}"
        expected[geometry_id] = ("geometry", {**base(state, "geometry_calibration"), "job_type": "geometry_calibration", "geometry_contract": "fixed_base_only", "calibration_artifacts": True, "output_dir": str((root / "geometry" / "calibration" / str(state["state_id"])).resolve())})
        feature_id = f"feature__geometry_calibration__{state['state_id']}__dinov3_vitl16_pixal3d__user_37_grid_track"
        expected[feature_id] = ("feature", {**base(state, "geometry_calibration"), "job_type": "geometry_calibration_feature", "backbone": "dinov3_vitl16_pixal3d", "feature_track": "user_37_grid_track", "variant_keys": calibration_keys, "geometry_contract": "fixed_base_only", "output_dir": str((root / "features" / "geometry_calibration" / str(state["state_id"])).resolve())})
    actual = {row.job_id: (row.phase, row.payload) for row in rows}
    return actual == expected


_METRIC_IDENTITY_FIELDS = (
    "job_id",
    "design_id",
    "image_state_id",
    "pair_id",
    "analysis_track",
    "track",
    "shoe_id",
    "quartet_id",
    "first_view",
    "second_view",
    "changed_view",
    "corruption",
    "severity",
    "baseline_steps",
    "angular_separation_deg",
    "variant_key",
    "feature_track",
    "backbone",
    "input_size",
    "feature_mode",
    "resolution",
)
_CALIBRATION_IDENTITY_FIELDS = (
    *_METRIC_IDENTITY_FIELDS[:15],
    "perturbation_kind",
    "magnitude",
    *_METRIC_IDENTITY_FIELDS[15:],
)


def _metric_identity(
    condition: object,
    variant: object,
    *,
    analysis_track: str,
    changed_view: str | None,
) -> tuple[object, ...]:
    job_id = (
        f"feature__{analysis_track}__{condition.image_state_id}__"
        f"{variant.backbone}__{variant.feature_track}"
    )
    return (
        job_id,
        condition.design_id,
        condition.image_state_id,
        condition.pair_id,
        analysis_track,
        condition.track,
        condition.shoe_id,
        condition.quartet_id,
        condition.first_view,
        condition.second_view,
        changed_view,
        condition.corruption,
        float(condition.severity),
        condition.baseline_steps,
        float(condition.angular_separation_deg),
        variant.key,
        variant.feature_track,
        variant.backbone,
        variant.input_size,
        variant.feature_mode,
        variant.resolution,
    )


def _exact_metric_keys(
    contract: Adjacent16Contract,
    design: Adjacent16Design,
) -> Mapping[str, set[tuple[object, ...]]]:
    controlled: set[tuple[object, ...]] = set()
    stress: set[tuple[object, ...]] = set()
    calibration: set[tuple[object, ...]] = set()
    calibration_variants = tuple(
        variant
        for variant in design.feature_variants
        if variant.key in contract.geometry_calibration_variant_keys
    )
    kind_config = {"rotation": "rotation_degrees", "relative_translation": "relative_translation", "depth_scale": "depth_scale", "relative_depth_noise": "relative_depth_noise_std"}
    for condition in design.controlled_conditions:
        for variant in design.feature_variants:
            controlled.add(
                _metric_identity(
                    condition,
                    variant,
                    analysis_track="controlled",
                    changed_view=condition.changed_view,
                )
            )
    for condition in design.stress_conditions:
        for variant in design.feature_variants:
            stress.add(
                _metric_identity(
                    condition,
                    variant,
                    analysis_track="stress",
                    changed_view=None,
                )
            )
    for condition in design.controlled_conditions:
        if condition.corruption != "clean":
            continue
        quartet = next(
            item
            for item in contract.quartets
            if item.quartet_id == condition.quartet_id
        )
        for changed_view in quartet.view_ids:
            if changed_view not in {condition.first_view, condition.second_view}:
                continue
            for kind, config_name in kind_config.items():
                for magnitude in contract.raw["geometry_calibration"][config_name]:
                    for variant in calibration_variants:
                        base = _metric_identity(
                            condition,
                            variant,
                            analysis_track="geometry_calibration",
                            changed_view=changed_view,
                        )
                        calibration.add(
                            (
                                *base[:15],
                                kind,
                                float(magnitude),
                                *base[15:],
                            )
                        )
    return {"controlled": controlled, "stress": stress, "geometry_calibration": calibration}


def _validate_metric_keys(
    root: Path,
    frames: Mapping[str, pd.DataFrame],
    contract: Adjacent16Contract,
) -> bool:
    expected = _exact_metric_keys(contract, _frozen_design(root, contract))
    for track, wanted in expected.items():
        frame = frames[track]
        fields = (
            _CALIBRATION_IDENTITY_FIELDS
            if track == "geometry_calibration"
            else _METRIC_IDENTITY_FIELDS
        )
        if not set(fields) <= set(frame):
            return False
        actual = {tuple(row.get(field) for field in fields) for row in frame.to_dict("records")}
        if len(actual) != len(frame) or actual != wanted:
            return False
    return True


def validate_adjacent16_final_gate(run_root: str | Path) -> Adjacent16FinalGateReport:
    """Validate exact adjacent-16 rows, evidence coverage, and storage policy."""

    root = Path(run_root).expanduser().resolve()
    failures: set[str] = set()
    expected_rows = {"controlled": 0, "geometry_calibration": 0, "stress": 0, "total": 0}
    actual_rows = dict(expected_rows)
    coverage = all_four = support_retention = stress_rate = float("nan")
    raw_features = _raw_feature_artifacts(root) if root.exists() else ()
    unexplained: tuple[str, ...] = ()
    try:
        contract, config_sha256 = _load_contract(root)
        expected_rows = dict(contract.expected_rows)
        registry = _registry_rows(root)
        counts: dict[str, int] = defaultdict(int)
        for row in registry:
            counts[row.phase] += 1
        if dict(counts) != _EXPECTED_REGISTRY_COUNTS or len(registry) != sum(_EXPECTED_REGISTRY_COUNTS.values()):
            failures.add("registry_counts")
        if not _validate_registry_graph(root, contract, config_sha256, registry):
            failures.add("registry_graph")
        bad_outputs = [row.job_id for row in registry if row.state == "completed" and not _valid_output_manifest(row, root)]
        if bad_outputs:
            failures.add("registry_output_hashes")
        unexplained = tuple(row.job_id for row in registry if row.state != "completed")
        if unexplained:
            failures.add("unexplained_jobs")
        frames, frame_failures = _read_track_frames(root)
        failures.update(frame_failures)
        actual_rows = {track: len(frames[track]) for track in _TRACK_FILES}
        actual_rows["total"] = sum(actual_rows.values())
        if actual_rows != expected_rows:
            failures.add("metric_row_counts")
        if not _validate_metric_keys(root, frames, contract):
            failures.add("metric_row_keys")
        for track, frame in frames.items():
            if frame.empty:
                continue
            provenance = {"config_sha256", "camera_label", "camera_provenance"}
            if not provenance <= set(frame) or not (
                (frame["config_sha256"] == config_sha256).all()
                and (frame["camera_label"] == CAMERA_MODE_LABEL).all()
                and (frame["camera_provenance"] == "fixed_precomputed_da3").all()
            ):
                failures.add("metric_provenance")
        series, expected_changed_rows = _expected_controlled_series(root, contract)
        controlled = _actual_primary_rows(frames["controlled"])
        baseline: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
        for key, row in controlled.items():
            shoe, quartet, pair, changed, corruption, _severity, variant = key
            if corruption == "clean" and (changed is None or pd.isna(changed)):
                baseline[shoe, quartet, pair, variant] = row
        complete: set[tuple[str, str, str, str, str, str]] = set()
        for unit in series:
            shoe, quartet, pair, _placement, _corruption, variant = unit
            required = [
                candidate
                for candidate in expected_changed_rows
                if (*candidate[:5], candidate[6]) == unit
            ]
            evidence = [controlled.get(item) for item in required]
            base = baseline.get((shoe, quartet, pair, variant))
            if base is not None and _finite_primary(base) and all(item is not None and _finite_primary(item) for item in evidence):
                complete.add(unit)
        coverage = len(complete) / len(series)
        if coverage < float(contract.raw["analyzability"]["minimum_controlled_complete_series_coverage"]):
            failures.add("controlled_complete_series")
        def physical_cell(unit: tuple[str, str, str, str, str, str]) -> tuple[str, str, str, str, str, str, str]:
            shoe, quartet, pair_id, placement, corruption, variant = unit
            parts = pair_id.split("__")
            if len(parts) < 2:
                raise ValueError("controlled pair ID is malformed")
            return quartet, parts[-2], parts[-1], placement, corruption, variant, shoe

        primary_cells = {physical_cell(unit)[:-1] for unit in series}
        completed_physical = {physical_cell(unit) for unit in complete}
        all_four = sum(
            all((*cell, shoe) in completed_physical for shoe in contract.raw["shoes"])
            for cell in primary_cells
        ) / len(primary_cells)
        if all_four < float(contract.raw["analyzability"]["minimum_all_four_shoe_primary_cell_rate"]):
            failures.add("all_four_shoe_primary_cells")
        def feature_key(backbone: str, mode: str, resolution: int) -> str:
            return f"user_37_grid_track__{backbone}__592__{mode}__{resolution}"

        valid_members: dict[str, set[tuple[object, ...]]] = defaultdict(set)
        for raw_key, row in controlled.items():
            if not _finite_primary(row):
                continue
            shoe, quartet, pair_id, changed, corruption, severity, variant = raw_key
            parts = str(pair_id).split("__")
            if len(parts) < 2:
                raise ValueError("controlled pair ID is malformed")
            valid_members[str(variant)].add((shoe, quartet, parts[-2], parts[-1], changed, corruption, severity))
        comparisons = [
            tuple(feature_key(backbone, "bilinear", 37) for backbone in (
                "dino_vitb16", "dinov2_vitb14", "dinov3_vitb16", "dinov3_vitl16_pixal3d",
            )),
            tuple(feature_key("dinov3_vitl16_pixal3d", "bilinear", resolution) for resolution in (37, 74, 148, 296, 592)),
            *[
                (
                    feature_key("dinov3_vitl16_pixal3d", "bilinear", resolution),
                    feature_key("dinov3_vitl16_pixal3d", "naf_normal_guide", resolution),
                )
                for resolution in (74, 148, 296, 592)
            ],
        ]
        retentions: list[float] = []
        for variants in comparisons:
            sets = [valid_members[variant] for variant in variants]
            union = set().union(*sets)
            if not union:
                retentions.append(0.0)
            else:
                retentions.append(len(set.intersection(*sets)) / len(union))
        support_retention = min(retentions) if retentions else float("nan")
        if support_retention < float(contract.raw["analyzability"]["minimum_common_support_retention"]):
            failures.add("common_support_retention")
        stress = frames["stress"]
        stress_fields = {"abstained", "s11_first_to_second_overlap", "s11_second_to_first_overlap"}
        if not stress_fields <= set(stress) or stress.empty:
            failures.add("stress_evidence")
        else:
            stress_rate = sum(
                _finite_primary(row)
                and min(
                    float(row["s11_first_to_second_overlap"]),
                    float(row["s11_second_to_first_overlap"]),
                ) >= 0.08
                for row in stress.to_dict("records")
            ) / len(stress)
            if stress_rate < float(contract.raw["analyzability"]["minimum_stress_symmetric_pass_rate"]):
                failures.add("stress_symmetric_pass_rate")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, sqlite3.Error, pa.ArrowException):
        failures.add("input_validation")
        if not (root / "registry" / "jobs.sqlite").is_file():
            failures.add("registry_counts")
    if raw_features:
        failures.add("raw_feature_artifact")
    report = Adjacent16FinalGateReport(
        passed=not failures, expected_rows=expected_rows, actual_rows=actual_rows,
        controlled_complete_series_coverage=coverage,
        all_four_shoe_primary_cell_rate=all_four,
        minimum_common_support_retention=support_retention,
        stress_symmetric_pass_rate=stress_rate, raw_feature_artifacts=raw_features,
        unexplained_jobs=unexplained, failed_checks=tuple(sorted(failures)),
    )
    provenance: dict[str, object] = {"config_sha256": None, "registry_sha256": None, "manifest_sha256": None, "metric_input_sha256": None, "output_sha256": None}
    try:
        config_path = root / "config" / "frozen_config.json"
        provenance["config_sha256"] = sha256_file(config_path)
        logical = [
            {"job_id": row.job_id, "phase": row.phase, "state": row.state,
             "payload_hash": row.payload_hash, "output_manifest": row.output_manifest}
            for row in _registry_rows(root)
        ]
        provenance["registry_sha256"] = hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifests = [root / "manifests" / name for name in ("image_states.jsonl", "pair_conditions.jsonl", "feature_variants.json")]
        provenance["manifest_sha256"] = _tree_hash(root, manifests)
        tables = [root / "metrics" / name for name in _TRACK_FILES.values()]
        provenance["metric_input_sha256"] = _tree_hash(root, tables)
        provenance["output_sha256"] = hashlib.sha256(json.dumps(_json_safe(asdict(report)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (OSError, ValueError, TypeError, json.JSONDecodeError, sqlite3.Error):
        failures.add("report_provenance")
        report = Adjacent16FinalGateReport(**{**asdict(report), "passed": False, "failed_checks": tuple(sorted(failures))})
    destination = root / "reports" / "adjacent16_final_gate.json"
    payload = _json_safe({**asdict(report), **provenance})
    _atomic_json(payload, destination)
    if json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("final gate report readback differs from persisted decision")
    return report
