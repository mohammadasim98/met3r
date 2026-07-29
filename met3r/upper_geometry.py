"""Long-lived fixed-camera DA3 geometry worker for upper-view experiment jobs."""

from __future__ import annotations

import errno
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .camera_metadata import CAMERA_MODE_LABEL
from .da3_depth import (
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DA3_REVISION,
    DepthArtifact,
    load_da3_model,
    run_fixed_camera_depth,
)
from .geometry_perturbations import write_geometry_calibration_artifacts
from .job_registry import JobRegistry, WorkerSummary, sha256_file
from .provenance import repository_state
from .upper_design import MaterializedState


_PROCESS_RES = 504
_STALE_SECONDS = 21_600
_SAFE_VIEW_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.EBUSY,
        errno.EINTR,
        errno.ETIMEDOUT,
    }
)


@dataclass(frozen=True)
class _CameraSet:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    view_ids: tuple[str, str, str, str]


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _path_within(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside {root}: {resolved}") from error
    return resolved


def _validated_view_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or _SAFE_VIEW_ID.fullmatch(value) is None
    ):
        raise ValueError("unsafe materialized state view ID")
    return value


def _validate_state_artifact_set(
    root: Path,
    expected_paths: tuple[Path, ...],
    label: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"materialized state artifact set is not exact: {label}")
    actual_paths: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or path.is_dir() or not path.is_file():
            raise ValueError(f"materialized state artifact set is not exact: {label}")
        actual_paths.add(path)
    if actual_paths != set(expected_paths):
        raise ValueError(f"materialized state artifact set is not exact: {label}")


def _load_config(run_root: Path) -> tuple[Mapping[str, object], str, str]:
    config_path = run_root / "config" / "frozen_config.json"
    config_content = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    config = json.loads(config_content)
    if not isinstance(config, Mapping):
        raise ValueError("frozen config must contain an object")
    run_manifest = _json_object(run_root / "run_manifest.json")
    if run_manifest.get("config_sha256") != config_sha256:
        raise ValueError("run manifest config hash mismatch")
    scope = run_manifest.get("scope")
    if scope not in {"pilot", "full"}:
        if not isinstance(run_manifest.get("experiment_id"), str):
            raise ValueError(
                "run manifest requires a 'pilot'/'full' scope or an experiment ID"
            )
        scope = "adjacent16"
    if config.get("stage_size") != _PROCESS_RES:
        raise ValueError("frozen stage_size must be exactly 504")
    models = config.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("frozen config requires a models object")
    if models.get("da3_model_id") != DEFAULT_DA3_MODEL_ID:
        raise ValueError("frozen DA3 model ID does not match the approved model")
    if models.get("da3_revision") != DEFAULT_DA3_REVISION:
        raise ValueError("frozen DA3 revision does not match the approved revision")
    return config, config_sha256, str(scope)


def _read_materialized_state(state_dir: Path) -> MaterializedState:
    metadata_path = state_dir / "state.json"
    metadata = _json_object(metadata_path)
    state = metadata.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("state_id"), str):
        raise ValueError(f"materialized state metadata is malformed: {metadata_path}")
    if metadata.get("stage_size") != _PROCESS_RES:
        raise ValueError("materialized state stage_size must be exactly 504")
    images_manifest_path = state_dir / "images.json"
    images_manifest = _json_object(images_manifest_path)
    records = images_manifest.get("views")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("materialized state images.json must contain exactly four views")
    manifest_records: list[Mapping[str, object]] = []
    view_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("materialized state images.json view is malformed")
        view_id = _validated_view_id(record.get("view_id"))
        manifest_records.append(record)
        view_ids.append(view_id)
    if len(set(view_ids)) != 4:
        raise ValueError("materialized state images.json has duplicate view IDs")
    exact_view_ids = tuple(view_ids)
    if len(exact_view_ids) != 4:
        raise AssertionError("four-view state contract was not enforced")
    image_root = state_dir / "images"
    mask_root = state_dir / "masks"
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    for record, view_id in zip(manifest_records, exact_view_ids):
        expected_image = f"images/{view_id}.png"
        expected_mask = f"masks/{view_id}.png"
        if record.get("image") != expected_image or record.get("mask") != expected_mask:
            raise ValueError("materialized state images.json path contract mismatch")
        image_paths.append(image_root / f"{view_id}.png")
        mask_paths.append(mask_root / f"{view_id}.png")
    exact_image_paths = tuple(image_paths)
    exact_mask_paths = tuple(mask_paths)
    _validate_state_artifact_set(image_root, exact_image_paths, "images")
    _validate_state_artifact_set(mask_root, exact_mask_paths, "masks")
    image_paths = [
        _path_within(path, image_root, "materialized state image")
        for path in exact_image_paths
    ]
    mask_paths = [
        _path_within(path, mask_root, "materialized state mask")
        for path in exact_mask_paths
    ]
    for metadata_view_ids in (state.get("view_ids"), metadata.get("view_ids")):
        if metadata_view_ids is not None:
            if (
                not isinstance(metadata_view_ids, list)
                or tuple(metadata_view_ids) != exact_view_ids
            ):
                raise ValueError("materialized state metadata view order mismatch")
    masks: list[np.ndarray] = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        with Image.open(image_path) as image:
            if image.convert("RGB").size != (_PROCESS_RES, _PROCESS_RES):
                raise ValueError(f"materialized RGB is not 504x504: {image_path}")
        with Image.open(mask_path) as image:
            mask_image = image.convert("L")
            if mask_image.size != (_PROCESS_RES, _PROCESS_RES):
                raise ValueError(f"materialized mask is not 504x504: {mask_path}")
            mask = np.asarray(mask_image) > 127
        masks.append(mask)
    return MaterializedState(
        state_id=str(state["state_id"]),
        state_dir=state_dir,
        image_paths=tuple(image_paths),
        mask_paths=tuple(mask_paths),
        masks=np.stack(masks),
        metadata_path=metadata_path,
    )


def _load_cameras(
    path: Path,
    *,
    expected_view_ids: tuple[str, str, str, str],
) -> _CameraSet:
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "extrinsics",
            "intrinsics",
            "view_ids",
            "camera_groups",
        }
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"camera artifact is missing arrays: {sorted(missing)}")
        extrinsics = np.asarray(artifact["extrinsics"])
        intrinsics = np.asarray(artifact["intrinsics"])
        view_ids = tuple(str(value) for value in artifact["view_ids"].tolist())
        groups = tuple(str(value) for value in artifact["camera_groups"].tolist())
    if extrinsics.shape not in {(4, 3, 4), (4, 4, 4)}:
        raise ValueError(
            f"upper extrinsics have an unexpected shape: {extrinsics.shape}"
        )
    if intrinsics.shape != (4, 3, 3):
        raise ValueError(
            f"upper intrinsics have an unexpected shape: {intrinsics.shape}"
        )
    if view_ids != expected_view_ids:
        raise ValueError(
            f"camera view order must match the materialized state: "
            f"{expected_view_ids}"
        )
    if len(set(view_ids)) != 4:
        raise ValueError("camera artifact has duplicate view IDs")
    if groups != ("upper",) * 4:
        raise ValueError("camera artifact must contain only four upper views")
    return _CameraSet(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        view_ids=view_ids,
    )


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(extrinsics)
    if value.shape == (4, 4, 4):
        return value
    result = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    result[:, :3, :4] = value
    return result


def _validate_fixed_camera_contract(
    artifact: DepthArtifact,
    cameras: _CameraSet,
) -> None:
    with np.load(artifact.arrays_path, allow_pickle=False) as arrays:
        required = {
            "depth",
            "confidence",
            "mask",
            "extrinsics",
            "intrinsics",
            "meta__view_ids",
            "meta__camera_groups",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"depth artifact is missing arrays: {sorted(missing)}")
        if "processed_images" in arrays.files:
            raise ValueError("compact depth artifact persisted processed_images")
        if np.asarray(arrays["depth"]).shape != (4, _PROCESS_RES, _PROCESS_RES):
            raise ValueError("depth artifact must contain four 504x504 upper views")
        if np.asarray(arrays["confidence"]).shape != (4, _PROCESS_RES, _PROCESS_RES):
            raise ValueError(
                "confidence artifact must contain four 504x504 upper views"
            )
        if np.asarray(arrays["mask"]).shape != (4, _PROCESS_RES, _PROCESS_RES):
            raise ValueError("mask artifact must contain four 504x504 upper views")
        if not np.allclose(
            np.asarray(arrays["extrinsics"]),
            _homogeneous(cameras.extrinsics),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("depth artifact extrinsics do not match fixed cameras")
        if not np.allclose(
            np.asarray(arrays["intrinsics"]),
            cameras.intrinsics,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("depth artifact intrinsics do not match fixed cameras")
        artifact_views = tuple(
            str(value) for value in arrays["meta__view_ids"].tolist()
        )
        if artifact_views != cameras.view_ids:
            raise ValueError("depth artifact view order mismatch")
        if tuple(
            str(value) for value in arrays["meta__camera_groups"].tolist()
        ) != ("upper",) * 4:
            raise ValueError("depth artifact contains a non-upper camera group")
        if str(arrays["meta__camera_provenance"].item()) != "fixed_precomputed_da3":
            raise ValueError("depth artifact camera provenance mismatch")
    metadata = _json_object(artifact.metadata_path)
    if metadata.get("camera_mode") != CAMERA_MODE_LABEL:
        raise ValueError("depth report camera label mismatch")
    if metadata.get("store_processed_images") is not False:
        raise ValueError("compact depth metadata must disable processed image storage")
    if metadata.get("process_res") != _PROCESS_RES:
        raise ValueError("depth metadata process_res must be exactly 504")
    if metadata.get("view_ids") != list(cameras.view_ids):
        raise ValueError("depth metadata view order mismatch")
    if metadata.get("camera_groups") != ["upper"] * 4:
        raise ValueError("depth metadata contains a non-upper camera group")


def _output_manifest(paths: tuple[Path, ...]) -> dict[str, object]:
    return {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(paths)
        ]
    }


def _numeric_sequence(
    calibration: Mapping[str, object],
    key: str,
) -> tuple[float, ...]:
    raw = calibration.get(key)
    if (
        not isinstance(raw, list)
        or not raw
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in raw
        )
    ):
        raise ValueError(
            f"geometry_calibration.{key} must be a non-empty number list"
        )
    values = tuple(float(value) for value in raw)
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError(
            f"geometry_calibration.{key} values must be positive and finite"
        )
    return values


def _write_full_calibration(
    artifact: DepthArtifact,
    output_dir: Path,
    published_output_dir: Path,
    config: Mapping[str, object],
    config_sha256: str,
    view_ids: tuple[str, str, str, str],
) -> tuple[Path, ...]:
    calibration = config.get("geometry_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("full run requires a geometry_calibration object")
    rotations = _numeric_sequence(calibration, "rotation_degrees")
    translations = _numeric_sequence(calibration, "relative_translation")
    depth_scales = _numeric_sequence(calibration, "depth_scale")
    depth_noise = _numeric_sequence(calibration, "relative_depth_noise_std")
    seed = config.get("seed")
    if seed != 42:
        raise ValueError("frozen calibration seed must be exactly 42")

    calibration_root = output_dir / "calibration"
    records: list[dict[str, object]] = []
    paths: list[Path] = []
    for view_index, view_id in enumerate(view_ids):
        view_manifest_path = write_geometry_calibration_artifacts(
            artifact.arrays_path,
            calibration_root / view_id,
            view_index=view_index,
            rotation_degrees=rotations,
            translation_fractions=translations,
            depth_scales=depth_scales,
            depth_noise_stds=depth_noise,
            seed=seed,
        )
        view_manifest = _json_object(view_manifest_path)
        raw_records = view_manifest.get("artifacts")
        if not isinstance(raw_records, list):
            raise ValueError(f"calibration manifest is malformed: {view_manifest_path}")
        published_view_records: list[dict[str, object]] = []
        paths.append(view_manifest_path)
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    f"calibration manifest record is malformed: {view_manifest_path}"
                )
            variant_path = Path(str(raw_record.get("path", ""))).resolve()
            if not variant_path.is_file():
                raise ValueError(f"calibration artifact is missing: {variant_path}")
            record = dict(raw_record)
            published_path = (
                published_output_dir / variant_path.relative_to(output_dir)
            ).resolve()
            record["path"] = str(published_path)
            record["view_id"] = view_id
            record["sha256"] = sha256_file(variant_path)
            records.append(record)
            published_view_record = dict(raw_record)
            published_view_record["path"] = str(published_path)
            published_view_records.append(published_view_record)
            paths.append(variant_path)
        view_manifest_path.write_text(
            json.dumps(
                {
                    **view_manifest,
                    "base_geometry": str(
                        (
                            published_output_dir
                            / artifact.arrays_path.relative_to(output_dir)
                        ).resolve()
                    ),
                    "artifacts": published_view_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    expected_count = len(view_ids) * (
        len(rotations) + len(translations) + len(depth_scales) + len(depth_noise)
    )
    if len(records) != expected_count:
        raise ValueError(
            f"calibration artifact count mismatch: {len(records)} != {expected_count}"
        )
    manifest_path = calibration_root / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "upper_geometry_calibration",
                "base_geometry": str(
                    (
                        published_output_dir
                        / artifact.arrays_path.relative_to(output_dir)
                    ).resolve()
                ),
                "config_sha256": config_sha256,
                "seed": seed,
                "views": list(view_ids),
                "artifacts": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths.append(manifest_path)
    return tuple(paths)


def _temporary_bundle(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _bundle_siblings(destination: Path) -> tuple[list[Path], list[Path]]:
    staging_paths = sorted(
        destination.parent.glob(f".{destination.name}.staging.*")
    )
    backup_paths = sorted(
        destination.parent.glob(f".{destination.name}.backup.*")
    )
    return staging_paths, backup_paths


def _recover_uncommitted_install(destination: Path) -> None:
    staging_paths, backup_paths = _bundle_siblings(destination)
    if len(backup_paths) > 1:
        raise ValueError(f"multiple interrupted backups found for {destination}")
    if backup_paths:
        if destination.exists():
            _remove_path(destination)
        backup_paths[0].rename(destination)
    for path in staging_paths:
        _remove_path(path)


def _clean_committed_install_debris(destination: Path) -> None:
    staging_paths, backup_paths = _bundle_siblings(destination)
    for path in (*staging_paths, *backup_paths):
        _remove_path(path)


def _install_bundle(staging: Path, destination: Path) -> Path | None:
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    if destination.exists():
        destination.rename(backup)
    else:
        backup = None
    try:
        staging.rename(destination)
    except Exception:
        if destination.exists():
            _remove_path(destination)
        if backup is not None and backup.exists():
            backup.rename(destination)
        raise
    return backup


def _rollback_installed_bundle(
    destination: Path,
    backup: Path | None,
) -> None:
    if destination.exists():
        _remove_path(destination)
    if backup is not None and backup.exists():
        backup.rename(destination)


def _published_paths(
    staging_paths: list[Path],
    staging_root: Path,
    destination: Path,
) -> tuple[Path, ...]:
    return tuple(
        destination / path.relative_to(staging_root)
        for path in staging_paths
    )


def _is_transient(error: OSError | RuntimeError) -> bool:
    if isinstance(error, OSError):
        return error.errno in _TRANSIENT_ERRNOS
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: device is busy" in message


def _validate_completed_geometry_jobs(
    registry: JobRegistry,
    shoe_id: str,
    run_root: Path,
) -> None:
    with sqlite3.connect(registry.path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT job_id, payload_json
            FROM jobs
            WHERE phase = 'geometry' AND state = 'completed'
            ORDER BY job_id
            """
        ).fetchall()
    for job_id, payload_json in rows:
        payload = json.loads(payload_json)
        if not isinstance(payload, Mapping):
            raise ValueError(f"stored geometry payload is malformed: {job_id}")
        if payload.get("shoe_id") == shoe_id:
            output_dir = _path_within(
                payload["output_dir"],
                run_root / "geometry",
                "output_dir",
            )
            state = registry.validated_state(str(job_id))
            if state == "completed":
                _clean_committed_install_debris(output_dir)
            else:
                _recover_uncommitted_install(output_dir)


def run_geometry_worker(
    run_root: str | Path,
    shoe_id: str,
    device: str,
    da3_repository: str | Path,
) -> WorkerSummary:
    """Claim one shoe's geometry jobs while reusing one pinned DA3 model."""

    root = Path(run_root).expanduser().resolve()
    config, config_sha256, scope = _load_config(root)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    _validate_completed_geometry_jobs(registry, shoe_id, root)
    registry.requeue_stale(
        phase="geometry",
        max_age_seconds=_STALE_SECONDS,
        now=datetime.now(timezone.utc),
    )
    model = load_da3_model(
        DEFAULT_DA3_MODEL_ID,
        revision=DEFAULT_DA3_REVISION,
        repository_root=da3_repository,
        device=device,
    )
    source_state = repository_state(da3_repository)
    completed = 0
    failed = 0
    while True:
        job = registry.claim_next(
            worker_id=f"geometry:{shoe_id}",
            phase="geometry",
            filters={"shoe_id": shoe_id},
        )
        if job is None:
            break
        try:
            job_type = job.payload.get("job_type")
            if job_type not in {
                "geometry",
                "stress_geometry",
                "geometry_calibration",
            }:
                raise ValueError(
                    "unexpected geometry job type: "
                    f"{job.payload.get('job_type')}"
                )
            if job.payload.get("config_sha256") != config_sha256:
                raise ValueError("geometry job config hash mismatch")
            state_dir = _path_within(
                job.payload["state_dir"],
                root / "states",
                "state_dir",
            )
            camera_path = _path_within(
                job.payload["camera_path"], root / "cameras", "camera_path"
            )
            output_dir = _path_within(
                job.payload["output_dir"], root / "geometry", "output_dir"
            )
            state = _read_materialized_state(state_dir)
            state_view_ids = tuple(path.stem for path in state.image_paths)
            if len(state_view_ids) != 4 or len(set(state_view_ids)) != 4:
                raise ValueError("materialized state does not have four unique views")
            if state.state_id != job.payload.get("state_id"):
                raise ValueError("materialized state ID does not match geometry job")
            state_metadata = _json_object(state.metadata_path)
            state_payload = state_metadata.get("state")
            if not isinstance(state_payload, Mapping):
                raise ValueError("materialized state payload is malformed")
            if (
                state_payload.get("shoe_id") != job.payload.get("shoe_id")
                or state_payload.get("shoe_id") != shoe_id
            ):
                raise ValueError("materialized state shoe does not match geometry job")
            if state_payload.get("track") != job.payload.get("track"):
                raise ValueError("materialized state track does not match geometry job")
            cameras = _load_cameras(
                camera_path,
                expected_view_ids=state_view_ids,
            )
            _recover_uncommitted_install(output_dir)
            staging_dir = _temporary_bundle(output_dir)
            try:
                artifact = run_fixed_camera_depth(
                    model,
                    state.image_paths,
                    cameras.extrinsics,
                    cameras.intrinsics,
                    output_dir=staging_dir,
                    model_id=DEFAULT_DA3_MODEL_ID,
                    model_revision=DEFAULT_DA3_REVISION,
                    camera_provenance="fixed_precomputed_da3",
                    process_res=_PROCESS_RES,
                    foreground_mask=state.masks,
                    view_ids=cameras.view_ids,
                    camera_groups=("upper",) * 4,
                    analysis_tier="primary",
                    source_state=source_state,
                    store_processed_images=False,
                )
                _validate_fixed_camera_contract(artifact, cameras)
                staging_paths = [artifact.arrays_path, artifact.metadata_path]
                if (
                    (
                        scope == "full"
                        and job_type == "geometry"
                        and state_payload.get("corruption") == "clean"
                    )
                    or (
                        job_type == "geometry_calibration"
                        and job.payload.get("calibration_artifacts") is True
                    )
                ):
                    staging_paths.extend(
                        _write_full_calibration(
                            artifact,
                            staging_dir,
                            output_dir,
                            config,
                            config_sha256,
                            cameras.view_ids,
                        )
                    )
                _output_manifest(tuple(staging_paths))
                published_paths = _published_paths(
                    staging_paths,
                    staging_dir,
                    output_dir,
                )
                backup_path = _install_bundle(staging_dir, output_dir)
            finally:
                if staging_dir.exists():
                    _remove_path(staging_dir)
            try:
                published_manifest = _output_manifest(published_paths)
                registry.complete(job.job_id, published_manifest)
            except Exception:
                _rollback_installed_bundle(output_dir, backup_path)
                raise
            if backup_path is not None and backup_path.exists():
                _remove_path(backup_path)
            completed += 1
        except (OSError, RuntimeError) as error:
            retryable = _is_transient(error)
            registry.fail(job.job_id, str(error), retryable=retryable)
            if not retryable or job.attempts >= 3:
                failed += 1
        except Exception:
            registry.fail(job.job_id, traceback.format_exc(), retryable=False)
            raise
    return WorkerSummary(completed=completed, abstained=0, failed=failed)
