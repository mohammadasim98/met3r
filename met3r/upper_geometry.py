"""Long-lived fixed-camera DA3 geometry worker for upper-view experiment jobs."""

from __future__ import annotations

import errno
import hashlib
import json
import traceback
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
from .upper_data import UPPER_VIEWS
from .upper_design import MaterializedState


_PROCESS_RES = 504
_STALE_SECONDS = 21_600
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
        raise ValueError("run manifest scope must be 'pilot' or 'full'")
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
    image_paths = tuple(
        state_dir / "images" / f"{view_id}.png" for view_id in UPPER_VIEWS
    )
    mask_paths = tuple(
        state_dir / "masks" / f"{view_id}.png" for view_id in UPPER_VIEWS
    )
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
        image_paths=image_paths,
        mask_paths=mask_paths,
        masks=np.stack(masks),
        metadata_path=metadata_path,
    )


def _load_cameras(path: Path) -> _CameraSet:
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
    if view_ids != UPPER_VIEWS:
        raise ValueError(f"camera view order must be exactly {UPPER_VIEWS}")
    if groups != ("upper",) * 4:
        raise ValueError("camera artifact must contain only four upper views")
    return _CameraSet(extrinsics=extrinsics, intrinsics=intrinsics)


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
        required = {"depth", "confidence", "mask", "extrinsics", "intrinsics"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"depth artifact is missing arrays: {sorted(missing)}")
        if "processed_images" in arrays.files:
            raise ValueError("compact depth artifact persisted processed_images")
        if np.asarray(arrays["depth"]).shape != (4, _PROCESS_RES, _PROCESS_RES):
            raise ValueError("depth artifact must contain four 504x504 upper views")
        if np.asarray(arrays["confidence"]).shape != (4, _PROCESS_RES, _PROCESS_RES):
            raise ValueError("confidence artifact must contain four 504x504 upper views")
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
        if artifact_views != UPPER_VIEWS:
            raise ValueError("depth artifact upper-view order mismatch")
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
    config: Mapping[str, object],
    config_sha256: str,
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
    for view_index, view_id in enumerate(UPPER_VIEWS):
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
            record["path"] = str(variant_path)
            record["view_id"] = view_id
            record["sha256"] = sha256_file(variant_path)
            records.append(record)
            paths.append(variant_path)
    expected_count = len(UPPER_VIEWS) * (
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
                "base_geometry": str(artifact.arrays_path.resolve()),
                "config_sha256": config_sha256,
                "seed": seed,
                "views": list(UPPER_VIEWS),
                "artifacts": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths.append(manifest_path)
    return tuple(paths)


def _is_transient(error: OSError | RuntimeError) -> bool:
    if isinstance(error, OSError):
        return error.errno in _TRANSIENT_ERRNOS
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: device is busy" in message


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
            if job.payload.get("job_type") not in {"geometry", "stress_geometry"}:
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
            cameras = _load_cameras(camera_path)
            artifact = run_fixed_camera_depth(
                model,
                state.image_paths,
                cameras.extrinsics,
                cameras.intrinsics,
                output_dir=output_dir,
                model_id=DEFAULT_DA3_MODEL_ID,
                model_revision=DEFAULT_DA3_REVISION,
                camera_provenance="fixed_precomputed_da3",
                process_res=_PROCESS_RES,
                foreground_mask=state.masks,
                view_ids=UPPER_VIEWS,
                camera_groups=("upper",) * 4,
                analysis_tier="primary",
                source_state=source_state,
                store_processed_images=False,
            )
            _validate_fixed_camera_contract(artifact, cameras)
            output_paths = [artifact.arrays_path, artifact.metadata_path]
            if (
                scope == "full"
                and job.payload.get("job_type") == "geometry"
                and state_payload.get("corruption") == "clean"
            ):
                output_paths.extend(
                    _write_full_calibration(
                        artifact,
                        output_dir,
                        config,
                        config_sha256,
                    )
                )
            registry.complete(
                job.job_id,
                _output_manifest(tuple(output_paths)),
            )
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
