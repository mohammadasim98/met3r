"""Resumable preparation for the frozen adjacent 16-view quartet run."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .adjacent16_contract import Adjacent16Contract, load_adjacent16_contract
from .adjacent16_data import Adjacent16Manifest, AdjacentViewRecord, build_adjacent16_manifest
from .adjacent16_design import Adjacent16Design, AdjacentImageState, build_adjacent16_design
from .adjacent16_materialize import materialize_adjacent_state
from .job_registry import JobDefinition, JobRegistry


@dataclass(frozen=True)
class PreparedAdjacent16Run:
    root: Path
    config_sha256: str
    registry: JobRegistry


_RUN_DIRS = (
    "config", "manifests", "registry", "cameras", "pca_fit_images", "states",
    "geometry", "features", "metrics/parts", "pca", "visualizations", "logs", "reports",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    path.write_bytes(content)


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(extrinsics)
    if value.shape == (48, 4, 4):
        return value.copy()
    if value.shape != (48, 3, 4):
        raise ValueError("adjacent camera extrinsics must have shape (48,3,4) or (48,4,4)")
    result = np.broadcast_to(np.eye(4), (48, 4, 4)).copy()
    result[:, :3, :4] = value
    return result


def _quartet_records(manifest: Adjacent16Manifest, shoe_id: str, quartet_id: str) -> tuple[AdjacentViewRecord, ...]:
    shoe = next((item for item in manifest.shoes if item.shoe_id == shoe_id), None)
    if shoe is None:
        raise ValueError(f"unknown shoe {shoe_id}")
    records = tuple(item for item in shoe.views if item.quartet_id == quartet_id)
    if len(records) != 4:
        raise ValueError("camera artifact requires exactly four quartet records")
    return records


def _camera_content(records: tuple[AdjacentViewRecord, ...]) -> tuple[dict[str, np.ndarray], list[str], dict[str, object]]:
    first = records[0]
    outputs = (
        first.source_path.parents[1] / "outputs"
    )
    with np.load(outputs / "intrinsics.npz") as artifact:
        intrinsics = np.asarray(artifact["intrinsics"])
    with np.load(outputs / "extrinsics.npz") as artifact:
        extrinsics = _homogeneous(np.asarray(artifact["extrinsics"]))
    indices = [int(record.view_id) for record in records]
    masks: list[np.ndarray] = []
    images: list[str] = []
    for record in records:
        with Image.open(record.source_mask_path) as image:
            masks.append(np.asarray(image.convert("L").resize((504, 504), Image.Resampling.NEAREST)) > 127)
        images.append(str(record.source_path))
    view_ids = [record.view_id for record in records]
    arrays = {
        "extrinsics": extrinsics[indices],
        "intrinsics": intrinsics[indices],
        "mask": np.stack(masks),
        "view_ids": np.asarray(view_ids),
        "camera_groups": np.asarray(["upper"] * 4),
    }
    metadata = {
        "camera_mode": "fixed/precomputed DA3 cameras (not calibrated GT)",
        "camera_provenance": "dense_camera_outputs",
        "camera_groups": ["upper"] * 4,
        "view_ids": view_ids,
        "source_hashes": {record.view_id: {"rgb": record.source_sha256, "mask": record.source_mask_sha256} for record in records},
    }
    return arrays, images, metadata


def _camera_matches(destination: Path, arrays: Mapping[str, np.ndarray], images: list[str], metadata: Mapping[str, object]) -> bool:
    try:
        with np.load(destination / "cameras.npz") as artifact:
            if set(artifact.files) != set(arrays) or any(not np.array_equal(np.asarray(artifact[key]), value) for key, value in arrays.items()):
                return False
        if json.loads((destination / "images.json").read_text(encoding="utf-8")) != images:
            return False
        if json.loads((destination / "metadata.json").read_text(encoding="utf-8")) != metadata:
            return False
        actual_files = {
            str(path.relative_to(destination))
            for path in destination.rglob("*")
            if path.is_file()
        }
        return actual_files == {"cameras.npz", "images.json", "metadata.json"}
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _replace(temporary: Path, destination: Path) -> None:
    backup: Path | None = None
    try:
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup.", dir=destination.parent))
            backup.rmdir()
            destination.rename(backup)
        temporary.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_camera_artifact(manifest: Adjacent16Manifest, root: Path, shoe_id: str, quartet_id: str) -> None:
    records = _quartet_records(manifest, shoe_id, quartet_id)
    arrays, images, metadata = _camera_content(records)
    destination = root / "cameras" / shoe_id / quartet_id
    if destination.is_dir() and _camera_matches(destination, arrays, images, metadata):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{quartet_id}.staging.", dir=destination.parent))
    try:
        np.savez_compressed(temporary / "cameras.npz", **arrays)
        (temporary / "images.json").write_bytes(_json_bytes(images))
        (temporary / "metadata.json").write_bytes(_json_bytes(metadata))
        if not _camera_matches(temporary, arrays, images, metadata):
            raise RuntimeError("staged quartet camera artifact failed canonical validation")
        _replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _audit_stress(manifest: Adjacent16Manifest) -> None:
    for shoe in manifest.shoes:
        for view in shoe.views:
            quality = view.quality
            if (
                quality is None
                or not quality.generation_valid
                or quality.task_status not in {"ok", "accepted"}
                or quality.background_preserved is None
                or quality.last_is_hidden is None
                or any(
                    value is None
                    for value in (
                        quality.same_shoe_identity,
                        quality.design_preservation,
                        quality.photorealism,
                        quality.deformation_plausibility,
                        quality.semantic_confidence,
                        quality.semantic_quality_score,
                        quality.center_shift_normalized,
                        quality.bottom_shift_normalized,
                        quality.width_ratio,
                    )
                )
                or quality.alignment_class == "unknown"
            ):
                raise ValueError(f"stress audit failed for {shoe.shoe_id}/{view.quartet_id}/{view.view_id}")


def _write_manifests(root: Path, manifest: Adjacent16Manifest, design: Adjacent16Design) -> None:
    def path_value(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [path_value(item) for item in value]
        if isinstance(value, dict):
            return {key: path_value(item) for key, item in value.items()}
        return value

    states = (*design.controlled_states, *design.stress_states)
    _immutable(root / "manifests/adjacent16_manifest.json", _json_bytes(path_value(asdict(manifest))))
    _immutable(root / "manifests/image_states.jsonl", b"".join((json.dumps(asdict(state), sort_keys=True, separators=(",", ":")) + "\n").encode() for state in states))
    _immutable(root / "manifests/pair_conditions.jsonl", b"".join((json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n").encode() for row in (*design.controlled_conditions, *design.stress_conditions)))
    _immutable(root / "manifests/feature_variants.json", _json_bytes([asdict(item) for item in design.feature_variants]))
    pca_sources = [
        {"shoe_id": shoe.shoe_id, "quartet_id": view.quartet_id, "view_id": view.view_id, "source_path": str(view.source_path)}
        for shoe in manifest.shoes for view in shoe.views
    ]
    _immutable(root / "pca_fit_images/clean_source_paths.json", _json_bytes(pca_sources))


def _payload(state: AdjacentImageState, root: Path, config_sha256: str, analysis_track: str) -> dict[str, object]:
    return {
        "state_id": state.state_id, "track": state.track, "analysis_track": analysis_track,
        "shoe_id": state.shoe_id, "quartet_id": state.quartet_id,
        "state_dir": str((root / "states" / state.state_id).resolve()),
        "camera_path": str((root / "cameras" / state.shoe_id / state.quartet_id / "cameras.npz").resolve()),
        "config_sha256": config_sha256,
    }


def register_adjacent16_jobs(registry: JobRegistry, run_root: Path, design: Adjacent16Design, config_sha256: str) -> None:
    """Register 32 geometry and 3,472 feature jobs exactly once."""

    by_backbone: dict[str, list[str]] = {}
    for variant in design.feature_variants:
        by_backbone.setdefault(variant.backbone, []).append(variant.key)
    if len(by_backbone) != 4:
        raise ValueError("frozen adjacent16 contract requires four feature backbones")
    clean = [state for state in design.controlled_states if state.corruption == "clean"]
    for state in design.controlled_states:
        base = _payload(state, run_root, config_sha256, "controlled")
        for backbone, variant_keys in sorted(by_backbone.items()):
            registry.register(JobDefinition(
                f"feature__controlled__{state.state_id}__{backbone}__user_37_grid_track", "feature",
                {**base, "job_type": "feature", "backbone": backbone, "feature_track": "user_37_grid_track", "variant_keys": variant_keys, "geometry_contract": "fixed_base_only", "output_dir": str((run_root / "features" / "controlled" / state.state_id / backbone).resolve())},
            ))
    for state in design.stress_states:
        base = _payload(state, run_root, config_sha256, "stress")
        registry.register(JobDefinition(
            f"geometry__stress__{state.state_id}", "geometry",
            {**base, "job_type": "stress_geometry", "geometry_contract": "base_plus_adaptive", "output_dir": str((run_root / "geometry" / "stress" / state.state_id).resolve())},
        ))
        for backbone, variant_keys in sorted(by_backbone.items()):
            registry.register(JobDefinition(
                f"feature__stress__{state.state_id}__{backbone}__user_37_grid_track", "feature",
                {**base, "job_type": "stress", "backbone": backbone, "feature_track": "user_37_grid_track", "variant_keys": variant_keys, "geometry_contract": "base_plus_adaptive", "output_dir": str((run_root / "features" / "stress" / state.state_id / backbone).resolve())},
            ))
    for state in clean:
        base = _payload(state, run_root, config_sha256, "geometry_calibration")
        registry.register(JobDefinition(
            f"geometry__geometry_calibration__{state.state_id}", "geometry",
            {**base, "job_type": "geometry_calibration", "geometry_contract": "fixed_base_only", "calibration_artifacts": True, "output_dir": str((run_root / "geometry" / "calibration" / state.state_id).resolve())},
        ))
        registry.register(JobDefinition(
            f"feature__geometry_calibration__{state.state_id}__dinov3_vitl16_pixal3d__user_37_grid_track", "feature",
            {**base, "job_type": "geometry_calibration_feature", "backbone": "dinov3_vitl16_pixal3d", "feature_track": "user_37_grid_track", "variant_keys": list(_calibration_keys(design)), "geometry_contract": "fixed_base_only", "output_dir": str((run_root / "features" / "geometry_calibration" / state.state_id).resolve())},
        ))
    if registry.counts_by_phase() != {"feature": 3472, "geometry": 32}:
        raise ValueError("adjacent16 registry job graph count differs from frozen protocol")


def _calibration_keys(design: Adjacent16Design) -> tuple[str, str]:
    matches = [item.key for item in design.feature_variants if item.key in {
        "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37",
        "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592",
    }]
    if len(matches) != 2:
        raise ValueError("frozen calibration feature variants are missing")
    return matches[0], matches[1]


def prepare_adjacent16_run(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    reconstruction_root: str | Path | None = None,
    stress_root: str | Path | None = None,
) -> PreparedAdjacent16Run:
    """Prepare audited quartet inputs and idempotent dynamic-camera jobs."""

    config_path = Path(config_path).resolve()
    content = config_path.read_bytes()
    contract = load_adjacent16_contract(config_path)
    if reconstruction_root is not None or stress_root is not None:
        contract = replace(
            contract,
            reconstruction_root=Path(reconstruction_root or contract.reconstruction_root).resolve(),
            stress_root=Path(stress_root or contract.stress_root).resolve(),
        )
    manifest = build_adjacent16_manifest(contract, require_stress=True)
    _audit_stress(manifest)
    design = build_adjacent16_design(manifest, contract)
    root = Path(output_root if output_root is not None else contract.output_root).expanduser().resolve()
    for relative in _RUN_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    config_sha256 = hashlib.sha256(content).hexdigest()
    _immutable(root / "config/frozen_config.json", content)
    _immutable(root / "run_manifest.json", _json_bytes({"experiment_id": contract.experiment_id, "config_sha256": config_sha256, "python_version": platform.python_version()}))
    _write_manifests(root, manifest, design)
    for shoe in manifest.shoes:
        for quartet_id in {view.quartet_id for view in shoe.views}:
            _write_camera_artifact(manifest, root, shoe.shoe_id, quartet_id)
    for state in (*design.controlled_states, *design.stress_states):
        materialize_adjacent_state(state, manifest, root, stage_size=504)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, config_sha256)
    return PreparedAdjacent16Run(root, config_sha256, registry)
