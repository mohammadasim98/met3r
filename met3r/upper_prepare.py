"""Idempotent preparation of pilot and full upper-view experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .bytesize_data import build_bytesize_manifest, write_upper_camera_artifacts
from .job_registry import JobDefinition, JobRegistry
from .upper_data import (
    CAMERA_DATASET_BY_SHOE,
    EXPECTED_SHOES,
    UpperManifest,
    asdict_paths,
    build_upper_manifest,
)
from .upper_design import FeatureVariant, ImageState, UpperDesign, build_upper_design
from .upper_materialize import materialize_image_state


@dataclass(frozen=True)
class PreparedRun:
    root: Path
    scope: str
    config_sha256: str
    registry: JobRegistry


_RUN_DIRS = (
    "config",
    "manifests",
    "registry",
    "cameras",
    "pca_fit_images",
    "states",
    "geometry",
    "metrics/parts",
    "pca",
    "visualizations",
    "logs",
    "reports",
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    path.write_bytes(content)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git_output(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return b"unavailable"


def _git_provenance(repository: Path) -> tuple[str, str]:
    commit = _git_output(repository, "rev-parse", "HEAD").decode().strip()
    status = _git_output(repository, "status", "--porcelain=v1", "-z")
    diff = _git_output(repository, "diff", "--binary", "HEAD", "--")
    fingerprint = hashlib.sha256(status + b"\0" + diff).hexdigest()
    return commit, fingerprint


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("met3r", "numpy", "Pillow", "pandas", "pyarrow", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


_FEATURE_BACKBONES = (
    "dino_vitb16",
    "dinov2_vitb14",
    "dinov3_vitb16",
    "dinov3_vitl16_pixal3d",
)


def _validated_physical_gpus(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or not all(
            isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0
            for gpu in value
        )
        or len(set(value)) != 4
    ):
        raise ValueError(
            "physical_gpus must contain exactly four unique non-negative integers"
        )
    return tuple(value)


def _physical_gpu_policy(
    config: Mapping[str, object],
    physical_gpus: Sequence[int] | None = None,
) -> dict[str, object]:
    source = "runtime_override" if physical_gpus is not None else "frozen_config"
    resolved = _validated_physical_gpus(
        config.get("physical_gpus") if physical_gpus is None else physical_gpus
    )
    return {
        "source": source,
        "physical_gpus": list(resolved),
        "pilot_gpu": 1,
        "geometry_assignments": {
            shoe_id: gpu for shoe_id, gpu in zip(EXPECTED_SHOES, resolved)
        },
        "feature_assignments": {
            backbone: gpu for backbone, gpu in zip(_FEATURE_BACKBONES, resolved)
        },
        "phase_order": ["geometry", "feature"],
        "allow_phase_overlap": False,
    }


def _write_run_manifest(
    run_root: Path,
    *,
    scope: str,
    config_sha256: str,
    physical_gpu_policy: Mapping[str, object],
) -> None:
    path = run_root / "run_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("scope") != scope
            or payload.get("config_sha256") != config_sha256
            or payload.get("physical_gpu_policy") != physical_gpu_policy
        ):
            raise ValueError(f"existing run manifest is incompatible: {path}")
        return
    repository = Path(__file__).resolve().parents[1]
    commit, dirty = _git_provenance(repository)
    _write_immutable(
        path,
        _json_bytes(
            {
                "scope": scope,
                "config_sha256": config_sha256,
                "git_commit": commit,
                "dirty_tree_fingerprint": dirty,
                "python_version": platform.python_version(),
                "package_versions": _package_versions(),
                "physical_gpu_policy": physical_gpu_policy,
            }
        ),
    )


def _write_design_manifests(
    run_root: Path,
    manifest: UpperManifest,
    design: UpperDesign,
    states: tuple[ImageState, ...],
) -> None:
    manifest_path = run_root / "manifests/upper_manifest.json"
    _write_immutable(
        manifest_path,
        json.dumps(asdict_paths(manifest), indent=2).encode("utf-8"),
    )
    _write_immutable(
        run_root / "manifests/image_states.jsonl",
        b"".join(
            (
                json.dumps(asdict(state), sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            for state in states
        ),
    )
    _write_immutable(
        run_root / "manifests/feature_variants.json",
        _json_bytes([asdict(variant) for variant in design.feature_variants]),
    )
    pair_path = run_root / "manifests/pair_conditions.parquet"
    rows = [
        asdict(row)
        for row in (*design.pair_conditions, *design.stress_pair_conditions)
    ]
    frame = pd.DataFrame(rows)
    if pair_path.is_file():
        existing = pd.read_parquet(pair_path)
        if not existing.equals(frame):
            raise ValueError(f"existing pair manifest differs: {pair_path}")
    else:
        frame.to_parquet(pair_path, index=False)


def _stage_pca_fit_images(
    manifest: UpperManifest,
    destination: Path,
    stage_size: int,
) -> None:
    for shoe in manifest.shoes:
        shoe_dir = destination / shoe.shoe_id
        shoe_dir.mkdir(parents=True, exist_ok=True)
        for view in shoe.views:
            output = shoe_dir / f"{view.view_id}.png"
            with Image.open(view.source_path) as image:
                rgb = np.asarray(
                    image.convert("RGB").resize(
                        (stage_size, stage_size), Image.Resampling.LANCZOS
                    )
                )
            with Image.open(view.source_mask_path) as image:
                mask = np.asarray(
                    image.convert("L").resize(
                        (stage_size, stage_size), Image.Resampling.NEAREST
                    )
                ) > 127
            expected = np.where(mask[..., None], rgb, 255).astype(np.uint8)
            if output.is_file():
                try:
                    with Image.open(output) as image:
                        actual = np.asarray(image.convert("RGB"))
                    if np.array_equal(actual, expected):
                        continue
                except OSError:
                    pass
            Image.fromarray(expected).save(output)


def _feature_pairs(
    variants: Sequence[FeatureVariant],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted({(variant.backbone, variant.feature_track) for variant in variants})
    )


def _state_payload(
    state: ImageState,
    run_root: Path,
    config_sha256: str,
    analysis_track: str,
) -> dict[str, object]:
    return {
        "state_id": state.state_id,
        "track": state.track,
        "analysis_track": analysis_track,
        "shoe_id": state.shoe_id,
        "state_dir": str((run_root / "states" / state.state_id).resolve()),
        "camera_path": str(
            (run_root / "cameras" / state.shoe_id / "cameras.npz").resolve()
        ),
        "config_sha256": config_sha256,
    }


def _register_state_jobs(
    registry: JobRegistry,
    run_root: Path,
    design: UpperDesign,
    config_sha256: str,
) -> None:
    feature_pairs = _feature_pairs(design.feature_variants)
    controlled_clean: list[ImageState] = []
    for state in design.image_states:
        base = _state_payload(state, run_root, config_sha256, "controlled")
        registry.register(
            JobDefinition(
                f"geometry__{state.state_id}",
                "geometry",
                {
                    **base,
                    "job_type": "geometry",
                    "output_dir": str(
                        (run_root / "geometry" / state.state_id).resolve()
                    ),
                },
            )
        )
        if state.corruption == "clean":
            controlled_clean.append(state)
        for backbone, feature_track in feature_pairs:
            registry.register(
                JobDefinition(
                    f"feature__controlled__{state.state_id}__{backbone}__{feature_track}",
                    "feature",
                    {
                        **base,
                        "job_type": "feature",
                        "backbone": backbone,
                        "feature_track": feature_track,
                    },
                )
            )

    if design.stress_states:
        for state in design.stress_states:
            base = _state_payload(state, run_root, config_sha256, "stress")
            registry.register(
                JobDefinition(
                    f"geometry__stress__{state.state_id}",
                    "geometry",
                    {
                        **base,
                        "job_type": "stress_geometry",
                        "output_dir": str(
                            (run_root / "geometry" / state.state_id).resolve()
                        ),
                    },
                )
            )
            for backbone, feature_track in feature_pairs:
                registry.register(
                    JobDefinition(
                        f"feature__stress__{state.state_id}__{backbone}__{feature_track}",
                        "feature",
                        {
                            **base,
                            "job_type": "stress",
                            "backbone": backbone,
                            "feature_track": feature_track,
                        },
                    )
                )
        for state in controlled_clean:
            base = _state_payload(
                state, run_root, config_sha256, "geometry_calibration"
            )
            for backbone, feature_track in feature_pairs:
                registry.register(
                    JobDefinition(
                        f"feature__geometry_calibration__{state.state_id}__"
                        f"{backbone}__{feature_track}",
                        "feature",
                        {
                            **base,
                            "job_type": "geometry_calibration",
                            "backbone": backbone,
                            "feature_track": feature_track,
                        },
                    )
                )


def prepare_run(
    config_path: str | Path,
    scope: Literal["pilot", "full"],
    *,
    output_root: str | Path | None = None,
    physical_gpus: Sequence[int] | None = None,
) -> PreparedRun:
    """Prepare immutable manifests, staged inputs, and idempotent job rows."""

    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be 'pilot' or 'full'")
    source = Path(config_path).expanduser().resolve()
    content = source.read_bytes()
    config_sha256 = _sha256_bytes(content)
    raw_config = json.loads(content)
    if not isinstance(raw_config, Mapping):
        raise ValueError("upper experiment config must be a JSON object")
    physical_gpu_policy = _physical_gpu_policy(raw_config, physical_gpus)
    raw_root = output_root if output_root is not None else raw_config.get("output_root")
    if not isinstance(raw_root, (str, Path)):
        raise ValueError("config requires output_root when no override is supplied")
    run_root = (Path(raw_root).expanduser() / scope).resolve()
    for relative in _RUN_DIRS:
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    _write_immutable(run_root / "config/frozen_config.json", content)
    _write_run_manifest(
        run_root,
        scope=scope,
        config_sha256=config_sha256,
        physical_gpu_policy=physical_gpu_policy,
    )

    reconstruction_root = raw_config.get("reconstruction_root")
    if not isinstance(reconstruction_root, str):
        raise ValueError("config requires reconstruction_root")
    manifest = build_upper_manifest(reconstruction_root)
    design = build_upper_design(manifest, raw_config, scope)
    states = (*design.image_states, *design.stress_states)
    _write_design_manifests(run_root, manifest, design, states)

    bytesize_manifest = build_bytesize_manifest(reconstruction_root)
    write_upper_camera_artifacts(
        bytesize_manifest,
        run_root / "cameras",
        shoe_aliases=CAMERA_DATASET_BY_SHOE,
    )
    stage_size = raw_config.get("stage_size")
    if not isinstance(stage_size, int) or stage_size != 504:
        raise ValueError("stage_size must be exactly 504")
    _stage_pca_fit_images(manifest, run_root / "pca_fit_images", stage_size)
    for state in states:
        materialize_image_state(state, manifest, run_root, stage_size=stage_size)

    registry = JobRegistry(run_root / "registry/jobs.sqlite")
    _register_state_jobs(registry, run_root, design, config_sha256)
    return PreparedRun(run_root, scope, config_sha256, registry)
