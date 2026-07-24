"""Pilot validation and registry-authoritative metric aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .camera_metadata import CAMERA_MODE_LABEL
from .da3_depth import DEFAULT_DA3_MODEL_ID, DEFAULT_DA3_REVISION
from .features import BACKBONE_SPECS, NAF_REVISION
from .job_registry import JobRegistry, sha256_file
from .pca_features import PCABasis
from .upper_data import EXPECTED_SHOES, UPPER_VIEWS
from .upper_design import FeatureVariant, build_upper_design


EXPECTED_PILOT_METRIC_ROWS = 1_320
EXPECTED_PILOT_GEOMETRY_JOBS = 32
EXPECTED_PILOT_FEATURE_JOBS = 64
EXPECTED_PILOT_SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_surplus_cq7743_300",
)
EXPECTED_PILOT_PAIRS = ("left__back", "left__right")
FULL_DESIGN_COUNTS = {
    "states": 344,
    "geometry_jobs": 344,
    "geometry_calibration_artifacts": 272,
    "controlled_metric_rows": 95_976,
    "geometry_calibration_metric_rows": 75_888,
    "stress_metric_rows": 2_232,
    "metric_rows": 174_096,
    "feature_jobs": 1_740,
    "backbones": 4,
}
_TRACK_FILENAMES = {
    "controlled": "controlled_fidelity.parquet",
    "geometry_calibration": "geometry_calibration.parquet",
    "stress": "real_deformation_stress.parquet",
}


@dataclass(frozen=True)
class PilotGateReport:
    passed: bool
    failed_checks: tuple[str, ...]
    expected_metric_rows: int
    actual_metric_rows: int
    clean_directional_coverage: float
    raw_feature_artifacts: tuple[str, ...]
    projected_total_bytes: int


@dataclass(frozen=True)
class _RegistryRow:
    job_id: str
    phase: str
    state: str
    payload: Mapping[str, object]
    payload_hash: str
    output_manifest: Mapping[str, object] | None
    error: str | None


@dataclass(frozen=True)
class _PilotDesign:
    states: Mapping[str, Mapping[str, object]]
    conditions: Mapping[str, Mapping[str, object]]
    variants: Mapping[str, FeatureVariant]
    expected_primary_keys: frozenset[tuple[str, str]]


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _live_repository_provenance() -> tuple[str, str]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "-z"],
        stderr=subprocess.DEVNULL,
    )
    diff = subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD", "--"],
        stderr=subprocess.DEVNULL,
    )
    return commit, hashlib.sha256(status + b"\0" + diff).hexdigest()


def _registry_rows(root: Path) -> tuple[_RegistryRow, ...]:
    registry_path = root / "registry" / "jobs.sqlite"
    with sqlite3.connect(registry_path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT job_id, phase, state, payload_json, payload_hash,
                   output_manifest_json, error
            FROM jobs
            ORDER BY job_id
            """
        ).fetchall()
    result: list[_RegistryRow] = []
    for job_id, phase, state, payload_json, payload_hash, output_json, error in rows:
        payload = json.loads(payload_json)
        output = json.loads(output_json) if output_json is not None else None
        if not isinstance(payload, Mapping) or (
            output is not None and not isinstance(output, Mapping)
        ):
            raise ValueError(f"registry row is malformed: {job_id}")
        result.append(
            _RegistryRow(
                str(job_id),
                str(phase),
                str(state),
                payload,
                str(payload_hash),
                output,
                None if error is None else str(error),
            )
        )
    return tuple(result)


def _normalized_optional(value: object) -> object:
    if value is None:
        return None
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _load_pilot_design(
    root: Path,
    config: Mapping[str, object],
) -> _PilotDesign:
    upper_manifest = _json_object(root / "manifests" / "upper_manifest.json")
    if upper_manifest.get("reconstruction_root") != config.get(
        "reconstruction_root"
    ):
        raise ValueError("prepared upper manifest reconstruction root differs")
    raw_shoes = upper_manifest.get("shoes")
    if not isinstance(raw_shoes, list):
        raise ValueError("prepared upper manifest requires shoes")
    shoes: list[SimpleNamespace] = []
    for expected_shoe, raw_shoe in zip(EXPECTED_SHOES, raw_shoes, strict=False):
        if (
            not isinstance(raw_shoe, Mapping)
            or raw_shoe.get("shoe_id") != expected_shoe
            or not isinstance(raw_shoe.get("views"), list)
        ):
            raise ValueError("prepared upper manifest shoe order differs")
        view_ids = tuple(
            raw.get("view_id") if isinstance(raw, Mapping) else None
            for raw in raw_shoe["views"]
        )
        if view_ids != UPPER_VIEWS:
            raise ValueError("prepared upper manifest view order differs")
        shoes.append(
            SimpleNamespace(
                shoe_id=expected_shoe,
                views=tuple(
                    SimpleNamespace(view_id=view_id)
                    for view_id in UPPER_VIEWS
                ),
            )
        )
    if len(raw_shoes) != len(EXPECTED_SHOES) or len(shoes) != len(EXPECTED_SHOES):
        raise ValueError("prepared upper manifest must contain exactly four shoes")
    canonical = build_upper_design(
        SimpleNamespace(shoes=tuple(shoes)),
        config,
        scope="pilot",
    )

    raw_variants = json.loads(
        (root / "manifests" / "feature_variants.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(raw_variants, list):
        raise ValueError("feature variant manifest must be a list")
    variants = tuple(
        FeatureVariant(
            feature_track=str(item["feature_track"]),
            backbone=str(item["backbone"]),
            input_size=int(item["input_size"]),
            feature_mode=str(item["feature_mode"]),
            resolution=int(item["resolution"]),
        )
        for item in raw_variants
        if isinstance(item, Mapping)
    )
    expected_variants = canonical.feature_variants
    if variants != expected_variants or len(variants) != 30:
        raise ValueError("feature variant manifest differs from the frozen pilot")
    variants_by_key = {item.key: item for item in variants}
    if len(variants_by_key) != len(variants):
        raise ValueError("feature variant keys must be unique")

    state_path = root / "manifests" / "image_states.jsonl"
    raw_states: list[Mapping[str, object]] = []
    for line in state_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("state_id"), str):
            raise ValueError("image state manifest row is malformed")
        raw_states.append(raw)
    expected_states = [asdict(state) for state in canonical.image_states]
    if raw_states != expected_states:
        raise ValueError("image state manifest differs from Task 3 enumeration")
    states = {
        str(state["state_id"]): state
        for state in raw_states
    }

    frame = pd.read_parquet(root / "manifests" / "pair_conditions.parquet")
    required = {
        "design_id",
        "track",
        "shoe_id",
        "pair_id",
        "first_view",
        "second_view",
        "changed_view",
        "image_state_id",
        "corruption",
        "severity",
    }
    if set(frame) != required or len(frame) != 44:
        raise ValueError("pilot pair-condition manifest is malformed")
    actual_conditions = [
        {
            key: _normalized_optional(value)
            for key, value in raw.items()
        }
        for raw in frame.to_dict("records")
    ]
    expected_conditions = [asdict(row) for row in canonical.pair_conditions]
    if actual_conditions != expected_conditions:
        raise ValueError("pair conditions differ from Task 3 enumeration")
    conditions = {
        str(row["design_id"]): row
        for row in actual_conditions
    }
    expected_keys = frozenset(
        (design_id, variant.key)
        for design_id in conditions
        for variant in variants
    )
    if len(expected_keys) != EXPECTED_PILOT_METRIC_ROWS:
        raise ValueError("pilot design does not expand to exactly 1,320 rows")
    return _PilotDesign(states, conditions, variants_by_key, expected_keys)


def _path_within(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    expected_root = root.resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the run tree: {resolved}") from error
    return resolved


def _manifest_files(row: _RegistryRow, root: Path) -> tuple[Path, ...]:
    if row.output_manifest is None:
        return ()
    files = row.output_manifest.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ValueError(f"registry output manifest is malformed: {row.job_id}")
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError(f"registry output file is malformed: {row.job_id}")
        paths.append(_path_within(str(item["path"]), root, "registered output"))
    return tuple(paths)


def _track_for_job(row: _RegistryRow) -> str | None:
    if row.phase != "feature":
        return None
    return {
        "feature": "controlled",
        "geometry_calibration": "geometry_calibration",
        "stress": "stress",
    }.get(str(row.payload.get("job_type")))


def _part_paths(
    root: Path,
    rows: Sequence[_RegistryRow],
    track: str,
    *,
    completed_only: bool = True,
) -> tuple[Path, ...]:
    parts: list[Path] = []
    for row in rows:
        if _track_for_job(row) != track:
            continue
        if completed_only and row.state != "completed":
            continue
        parquet = tuple(
            path
            for path in _manifest_files(row, root)
            if path.suffix == ".parquet"
            and path.is_relative_to((root / "metrics" / "parts").resolve())
        )
        if row.state == "completed" and len(parquet) != 1:
            raise ValueError(
                f"completed metric job must register one Parquet part: {row.job_id}"
            )
        parts.extend(parquet)
    if len(parts) != len(set(parts)):
        raise ValueError("multiple registry jobs reference the same metric part")
    return tuple(sorted(parts))


def _primary_key(track: str) -> tuple[str, ...]:
    if track in {"controlled", "stress"}:
        return ("design_id", "variant_key")
    return (
        "design_id",
        "variant_key",
        "changed_view",
        "perturbation_kind",
        "magnitude",
    )


def _sort_columns(track: str, columns: Sequence[str]) -> list[str]:
    preferred = list(_primary_key(track))
    return [column for column in preferred if column in columns]


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, temporary)
        readback = pq.read_table(temporary)
        if readback.num_rows != table.num_rows or not readback.schema.equals(
            table.schema, check_metadata=True
        ):
            raise ValueError("aggregated Parquet failed schema/row-count readback")
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


def _write_validated_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, destination)
    readback = pq.read_table(destination)
    if readback.num_rows != table.num_rows or not readback.schema.equals(
        table.schema, check_metadata=True
    ):
        raise ValueError("aggregated Parquet failed schema/row-count readback")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return destination


def _atomic_bytes(content: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(raw_temporary)
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


def _atomic_json(payload: Mapping[str, object], destination: Path) -> Path:
    return _atomic_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        destination,
    )


def _read_parts(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table()
    return table.to_pandas()


def _empty_metric_frame(track: str) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "design_id": pd.Series(dtype="string"),
        "variant_key": pd.Series(dtype="string"),
    }
    if track == "geometry_calibration":
        columns.update(
            {
                "changed_view": pd.Series(dtype="string"),
                "perturbation_kind": pd.Series(dtype="string"),
                "magnitude": pd.Series(dtype="float64"),
            }
        )
    return pd.DataFrame(columns)


def _row_matches_design(
    row: Mapping[str, object],
    condition: Mapping[str, object],
    variant: FeatureVariant,
) -> bool:
    condition_fields = (
        "track",
        "shoe_id",
        "pair_id",
        "first_view",
        "second_view",
        "changed_view",
        "image_state_id",
        "corruption",
        "severity",
    )
    for field in condition_fields:
        actual = _normalized_optional(row.get(field))
        expected = _normalized_optional(condition.get(field))
        if field == "severity":
            try:
                if float(actual) != float(expected):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return (
        row.get("feature_track") == variant.feature_track
        and row.get("backbone") == variant.backbone
        and row.get("input_size") == variant.input_size
        and row.get("feature_mode") == variant.feature_mode
        and row.get("resolution") == variant.resolution
    )


def _read_controlled_parts_with_ownership(
    root: Path,
    rows: Sequence[_RegistryRow],
    design: _PilotDesign,
) -> tuple[pd.DataFrame, bool]:
    frames: list[pd.DataFrame] = []
    ownership_valid = True
    controlled_jobs = [
        row for row in rows if _track_for_job(row) == "controlled"
    ]
    for registry_row in controlled_jobs:
        if registry_row.state != "completed":
            ownership_valid = False
            continue
        parts = [
            path
            for path in _manifest_files(registry_row, root)
            if path.suffix == ".parquet"
            and path.is_relative_to((root / "metrics" / "parts").resolve())
        ]
        feature_track = registry_row.payload.get("feature_track")
        expected_part = (
            root
            / "metrics"
            / "parts"
            / str(feature_track)
            / f"{registry_row.job_id}.parquet"
        ).resolve()
        if (
            len(parts) != 1
            or parts[0] != expected_part
        ):
            ownership_valid = False
            continue
        try:
            frame = pd.read_parquet(parts[0])
        except (OSError, ValueError):
            ownership_valid = False
            continue
        payload = registry_row.payload
        state_id = payload.get("state_id")
        feature_track = payload.get("feature_track")
        if (
            not isinstance(state_id, str)
            or state_id not in design.states
            or not isinstance(feature_track, str)
            or payload.get("track") != "controlled"
            or payload.get("analysis_track") != "controlled"
            or payload.get("backbone") != "dinov3_vitl16_pixal3d"
            or payload.get("shoe_id") != design.states[state_id].get("shoe_id")
        ):
            ownership_valid = False
            frames.append(frame)
            continue
        expected_keys = {
            (design_id, variant.key)
            for design_id, condition in design.conditions.items()
            if condition["image_state_id"] == state_id
            for variant in design.variants.values()
            if variant.feature_track == feature_track
        }
        actual_keys = set(
            zip(
                frame.get("design_id", pd.Series(dtype="object")),
                frame.get("variant_key", pd.Series(dtype="object")),
            )
        )
        if actual_keys != expected_keys or len(frame) != len(expected_keys):
            ownership_valid = False
        for raw in frame.to_dict("records"):
            design_id = raw.get("design_id")
            variant_key = raw.get("variant_key")
            condition = design.conditions.get(str(design_id))
            variant = design.variants.get(str(variant_key))
            if (
                raw.get("job_id") != registry_row.job_id
                or condition is None
                or variant is None
                or condition.get("image_state_id") != state_id
                or variant.feature_track != feature_track
                or not _row_matches_design(raw, condition, variant)
            ):
                ownership_valid = False
        frames.append(frame)
    if len(controlled_jobs) != EXPECTED_PILOT_FEATURE_JOBS:
        ownership_valid = False
    if not frames:
        return pd.DataFrame(), False
    return pd.concat(frames, ignore_index=True, sort=False), ownership_valid


def _validate_frame_primary_key(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    key = _primary_key(track)
    missing = set(key) - set(frame)
    if missing:
        raise ValueError(f"metric parts are missing primary-key columns: {sorted(missing)}")
    if frame.loc[:, list(key)].isna().any(axis=None):
        raise ValueError("metric primary key contains null values")
    duplicated = frame.duplicated(list(key), keep=False)
    if bool(duplicated.any()):
        values = frame.loc[duplicated, list(key)].head(2).to_dict("records")
        raise ValueError(f"duplicate metric primary key: {values}")
    return frame.sort_values(
        _sort_columns(track, frame.columns),
        kind="mergesort",
        ignore_index=True,
    )


def _aggregate_frame(root: Path, track: str) -> pd.DataFrame:
    if track not in _TRACK_FILENAMES:
        raise ValueError(f"unknown metric track: {track}")
    rows = _registry_rows(root)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    selected = [
        row for row in rows if row.state == "completed" and _track_for_job(row) == track
    ]
    for row in selected:
        if registry.validated_state(row.job_id) != "completed":
            raise ValueError(f"registered output hash validation failed: {row.job_id}")
    paths = _part_paths(root, rows, track)
    frame = _read_parts(paths)
    if frame.empty:
        frame = _empty_metric_frame(track)
    else:
        frame = _validate_frame_primary_key(frame, track)
    return frame


def aggregate_metric_parts(run_root: str | Path, track: str) -> Path:
    """Validate registered parts and atomically aggregate one analysis track."""

    root = Path(run_root).expanduser().resolve()
    if track not in _TRACK_FILENAMES:
        raise ValueError(f"unknown metric track: {track}")
    current = root / "metrics" / "current"
    if current.exists() or current.is_symlink():
        aggregate_all_metric_parts(root)
        return root / "metrics" / _TRACK_FILENAMES[track]
    frame = _aggregate_frame(root, track)
    destination = root / "metrics" / _TRACK_FILENAMES[track]
    return _atomic_parquet(frame, destination)


def _write_aggregate_preview(frame: pd.DataFrame, destination: Path) -> None:
    destination.write_text(frame.to_csv(index=False), encoding="utf-8")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def _aggregate_output_names() -> tuple[str, ...]:
    return (*_TRACK_FILENAMES.values(), "summary_preview.csv")


def resolve_aggregate_generation(run_root: str | Path) -> Path:
    """Resolve the aggregate pointer once for a consistent multi-file snapshot."""

    root = Path(run_root).expanduser().resolve()
    generations = (root / "metrics" / "aggregate_generations").resolve()
    current = root / "metrics" / "current"
    generation = current.resolve(strict=True)
    try:
        generation.relative_to(generations)
    except ValueError as error:
        raise ValueError("aggregate current pointer escapes generation root") from error
    if not generation.is_dir() or any(
        not (generation / name).is_file()
        for name in _aggregate_output_names()
    ):
        raise ValueError("aggregate generation is incomplete")
    return generation


def _publish_aggregate_pointer(metrics_root: Path, generation: Path) -> None:
    temporary = metrics_root / f".current.{uuid.uuid4().hex}"
    relative_target = Path("aggregate_generations") / generation.name
    temporary.symlink_to(relative_target, target_is_directory=True)
    try:
        os.replace(temporary, metrics_root / "current")
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    directory_fd = os.open(metrics_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    marker = generation / ".unpublished"
    if marker.exists():
        marker.unlink()


def _ensure_aggregate_aliases(metrics_root: Path) -> None:
    for name in _aggregate_output_names():
        alias = metrics_root / name
        temporary = metrics_root / f".{name}.alias.{uuid.uuid4().hex}"
        temporary.symlink_to(Path("current") / name)
        try:
            os.replace(temporary, alias)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()


def _cleanup_unpublished_generations(metrics_root: Path) -> None:
    generations = metrics_root / "aggregate_generations"
    generations.mkdir(parents=True, exist_ok=True)
    current: Path | None = None
    try:
        current = resolve_aggregate_generation(metrics_root.parent)
    except (OSError, ValueError):
        pass
    for path in generations.iterdir():
        if path.name.startswith(".staging-"):
            shutil.rmtree(path)
        elif (
            path.is_dir()
            and (path / ".unpublished").exists()
            and path.resolve() != current
        ):
            shutil.rmtree(path)
    if current is not None:
        marker = current / ".unpublished"
        if marker.exists():
            marker.unlink()


def aggregate_all_metric_parts(run_root: str | Path) -> tuple[Path, ...]:
    """Publish all aggregates as one recoverable filesystem generation."""

    root = Path(run_root).expanduser().resolve()
    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    _cleanup_unpublished_generations(metrics_root)
    frames = {
        track: _aggregate_frame(root, track)
        for track in _TRACK_FILENAMES
    }
    previews: list[pd.DataFrame] = []
    for track, frame in frames.items():
        if not frame.empty:
            previews.append(frame.assign(analysis_track=track).head(100))
    preview = (
        pd.concat(previews, ignore_index=True, sort=False)
        if previews
        else pd.DataFrame(columns=["analysis_track"])
    )
    generations = metrics_root / "aggregate_generations"
    staging = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=generations)
    )
    try:
        for track, frame in frames.items():
            _write_validated_parquet(
                frame,
                staging / _TRACK_FILENAMES[track],
            )
        _write_aggregate_preview(preview, staging / "summary_preview.csv")
        (staging / ".unpublished").write_text("unpublished\n", encoding="utf-8")
        digest = hashlib.sha256()
        for name in _aggregate_output_names():
            digest.update(name.encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(staging / name)))
        generation = generations / digest.hexdigest()
        if generation.exists():
            shutil.rmtree(staging)
        else:
            staging.rename(generation)
        _publish_aggregate_pointer(metrics_root, generation)
        _ensure_aggregate_aliases(metrics_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return tuple(
        metrics_root / _TRACK_FILENAMES[track]
        for track in _TRACK_FILENAMES
    )


def _raw_feature_artifacts(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower in {"features.npz", "features.npy"} or path.suffix.lower() in {
            ".pt",
            ".pth",
        }:
            found.append(str(path.relative_to(root)))
            continue
        if path.suffix.lower() != ".npz":
            continue
        try:
            with np.load(path, allow_pickle=False) as artifact:
                for key in ("meta__artifact_type", "artifact_type"):
                    if key in artifact.files and str(np.asarray(artifact[key]).item()) == (
                        "dino_features"
                    ):
                        found.append(str(path.relative_to(root)))
                        break
        except (OSError, ValueError):
            # Non-feature NPZ integrity is covered by its registry hash/schema owner.
            continue
    return tuple(sorted(set(found)))


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _pca_basis_bytes(path: Path) -> int:
    return sum(
        artifact.stat().st_size
        for backbone in path.iterdir()
        if backbone.is_dir()
        for artifact in (backbone / "basis.npz", backbone / "basis.json")
        if artifact.is_file()
    ) if path.is_dir() else 0


def _storage_projection(root: Path, actual_metric_rows: int) -> Mapping[str, object]:
    pilot_geometry_units = EXPECTED_PILOT_GEOMETRY_JOBS
    pilot_feature_units = EXPECTED_PILOT_FEATURE_JOBS
    pilot_backbones = 1
    pca_total_bytes = _directory_bytes(root / "pca")
    pca_basis_bytes = _pca_basis_bytes(root / "pca")
    pca_job_bytes = max(pca_total_bytes - pca_basis_bytes, 0)
    classified = {
        "states": _directory_bytes(root / "states"),
        "geometry": _directory_bytes(root / "geometry"),
        "metrics": _directory_bytes(root / "metrics" / "parts"),
        "pca_basis": pca_basis_bytes,
        "pca_job_outputs": pca_job_bytes,
        "visualizations": _directory_bytes(root / "visualizations"),
    }
    classified_total = sum(classified.values())
    reports_bytes = sum(
        path.stat().st_size
        for relative in (
            "config",
            "manifests",
            "registry",
            "cameras",
            "pca_fit_images",
            "logs",
        )
        for path in (root / relative).rglob("*")
        if path.is_file()
    )
    categories = {
        "run_metadata": {
            "measured_bytes": reports_bytes,
            "pilot_units": 1,
            "full_units": 1,
            "projected_bytes": reports_bytes,
        },
        "states": {
            "measured_bytes": classified["states"],
            "pilot_units": pilot_geometry_units,
            "full_units": FULL_DESIGN_COUNTS["states"],
        },
        "geometry": {
            "measured_bytes": classified["geometry"],
            "pilot_units": pilot_geometry_units,
            "full_units": (
                FULL_DESIGN_COUNTS["geometry_jobs"]
                + FULL_DESIGN_COUNTS["geometry_calibration_artifacts"]
            ),
        },
        "metrics": {
            "measured_bytes": classified["metrics"],
            "pilot_units": max(actual_metric_rows, 1),
            # Parts and the final aggregate are both persistent.
            "full_units": FULL_DESIGN_COUNTS["metric_rows"] * 2,
        },
        "pca_basis": {
            "measured_bytes": classified["pca_basis"],
            "pilot_units": pilot_backbones,
            "full_units": FULL_DESIGN_COUNTS["backbones"],
        },
        "pca_job_outputs": {
            "measured_bytes": classified["pca_job_outputs"],
            "pilot_units": pilot_feature_units,
            "full_units": FULL_DESIGN_COUNTS["feature_jobs"],
        },
        "visualizations": {
            "measured_bytes": classified["visualizations"],
            "pilot_units": pilot_feature_units,
            "full_units": FULL_DESIGN_COUNTS["feature_jobs"],
        },
    }
    for value in categories.values():
        if "projected_bytes" not in value:
            value["projected_bytes"] = math.ceil(
                int(value["measured_bytes"])
                * int(value["full_units"])
                / max(int(value["pilot_units"]), 1)
            )
    projected = sum(int(value["projected_bytes"]) for value in categories.values())
    return {
        "method": "artifact_category_full_design_projection",
        "full_design_counts": FULL_DESIGN_COUNTS,
        "pilot_classified_bytes": classified_total + reports_bytes,
        "categories": categories,
        "projected_total_bytes": projected,
    }


def _validate_registry(
    root: Path,
    rows: Sequence[_RegistryRow],
    config_sha256: str,
) -> set[str]:
    failures: set[str] = set()
    counts = {
        phase: sum(row.phase == phase for row in rows)
        for phase in ("geometry", "feature")
    }
    if counts != {
        "geometry": EXPECTED_PILOT_GEOMETRY_JOBS,
        "feature": EXPECTED_PILOT_FEATURE_JOBS,
    }:
        failures.add("planned_jobs")
    if any(row.state not in {"completed", "abstained"} for row in rows):
        failures.add("job_completion")
    if any(
        row.state == "abstained"
        and (row.error is None or not row.error.strip())
        for row in rows
    ):
        failures.add("job_completion")
    geometry = [row for row in rows if row.phase == "geometry"]
    feature = [row for row in rows if row.phase == "feature"]
    for shoe in EXPECTED_PILOT_SHOES:
        if sum(row.payload.get("shoe_id") == shoe for row in geometry) != 16:
            failures.add("planned_jobs")
        for feature_track in (
            "user_37_grid_track",
            "current_pixal3d_code_track",
        ):
            if (
                sum(
                    row.payload.get("shoe_id") == shoe
                    and row.payload.get("feature_track") == feature_track
                    for row in feature
                )
                != 16
            ):
                failures.add("planned_jobs")
    if any(
        row.payload.get("backbone") != "dinov3_vitl16_pixal3d"
        or row.payload.get("job_type") != "feature"
        for row in feature
    ):
        failures.add("planned_jobs")
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    for row in rows:
        expected_payload_hash = hashlib.sha256(
            _canonical_json(row.payload).encode("utf-8")
        ).hexdigest()
        if (
            row.payload_hash != expected_payload_hash
            or row.payload.get("config_sha256") != config_sha256
        ):
            failures.add("registry_hashes")
        if row.state == "completed" and registry.validated_state(row.job_id) != "completed":
            failures.add("registry_hashes")
    return failures


def _validate_geometry_provenance(
    root: Path,
    rows: Sequence[_RegistryRow],
    config: Mapping[str, object],
) -> bool:
    models = config.get("models")
    if not isinstance(models, Mapping):
        return False
    for row in rows:
        if row.phase != "geometry" or row.state != "completed":
            continue
        files = _manifest_files(row, root)
        metadata_paths = [path for path in files if path.name == "depth_geometry.json"]
        arrays_paths = [path for path in files if path.name == "depth_geometry.npz"]
        if len(metadata_paths) != 1 or len(arrays_paths) != 1:
            return False
        metadata = _json_object(metadata_paths[0])
        if (
            metadata.get("model_id") != DEFAULT_DA3_MODEL_ID
            or metadata.get("model_revision") != DEFAULT_DA3_REVISION
            or metadata.get("camera_provenance") != "fixed_precomputed_da3"
            or metadata.get("camera_mode") != CAMERA_MODE_LABEL
            or tuple(metadata.get("view_ids", ())) != UPPER_VIEWS
            or tuple(metadata.get("camera_groups", ())) != ("upper",) * 4
            or metadata.get("process_res") != 504
            or models.get("da3_model_id") != DEFAULT_DA3_MODEL_ID
            or models.get("da3_revision") != DEFAULT_DA3_REVISION
        ):
            return False
        with np.load(arrays_paths[0], allow_pickle=False) as artifact:
            if (
                str(artifact["meta__model_revision"].item())
                != DEFAULT_DA3_REVISION
                or str(artifact["meta__camera_provenance"].item())
                != "fixed_precomputed_da3"
                or tuple(str(value) for value in artifact["meta__view_ids"].tolist())
                != UPPER_VIEWS
                or tuple(
                    str(value) for value in artifact["meta__camera_groups"].tolist()
                )
                != ("upper",) * 4
            ):
                return False
    return True


def _validate_pca(
    root: Path,
    config: Mapping[str, object],
    config_sha256: str,
) -> bool:
    pilot = config.get("pilot")
    models = config.get("models")
    if not isinstance(pilot, Mapping) or not isinstance(models, Mapping):
        return False
    backbones = pilot.get("backbones")
    if not isinstance(backbones, list) or len(backbones) != 1:
        return False
    fit_root = root / "pca_fit_images"
    if not fit_root.is_dir():
        return False
    shoe_dirs = tuple(
        path.name for path in sorted(path for path in fit_root.iterdir() if path.is_dir())
    )
    if shoe_dirs != EXPECTED_SHOES:
        return False
    fit_paths: list[Path] = []
    for shoe_id in EXPECTED_SHOES:
        shoe_dir = fit_root / shoe_id
        files = tuple(sorted(path.name for path in shoe_dir.iterdir() if path.is_file()))
        if files != tuple(sorted(f"{view_id}.png" for view_id in UPPER_VIEWS)):
            return False
        fit_paths.extend(shoe_dir / f"{view_id}.png" for view_id in UPPER_VIEWS)
    if len(fit_paths) != 16 or not all(path.is_file() for path in fit_paths):
        return False

    manifests = sorted((root / "pca").glob("*/basis.json"))
    if len(manifests) != len(backbones):
        return False
    fingerprints: dict[str, set[str]] = {}
    for path in manifests:
        manifest = _json_object(path)
        backbone = str(manifest.get("backbone"))
        model = models.get(backbone)
        if (
            backbone not in backbones
            or not isinstance(model, Mapping)
            or manifest.get("backbone_revision") != model.get("revision")
            or manifest.get("config_sha256") != config_sha256
        ):
            return False
        basis = PCABasis.load(path.with_name("basis.npz"))
        fingerprint = str(manifest.get("fingerprint"))
        if basis.fingerprint != fingerprint:
            return False
        expected = {
            "artifact_type": "fixed_pca_basis",
            "backbone": backbone,
            "backbone_revision": model.get("revision"),
            "config_sha256": config_sha256,
            "seed": 42,
            "fit_image_count": 16,
            "fit_sample_count": 32,
            "input_sizes": [512, 592],
            "fit_images": [
                {
                    "path": str(image.resolve()),
                    "sha256": sha256_file(image),
                }
                for image in fit_paths
            ],
            "fingerprint": fingerprint,
        }
        if dict(manifest) != expected:
            return False
        fingerprints.setdefault(backbone, set()).add(fingerprint)
    return all(len(values) == 1 for values in fingerprints.values())


def _metric_provenance_valid(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    config_sha256: str,
) -> bool:
    required = {
        "config_sha256",
        "backbone",
        "backbone_revision",
        "camera_provenance",
        "camera_label",
        "first_view",
        "second_view",
        "pair_id",
        "shoe_id",
    }
    if not required <= set(frame):
        return False
    models = config.get("models")
    if not isinstance(models, Mapping):
        return False
    for backbone, rows in frame.groupby("backbone", dropna=False):
        model = models.get(str(backbone))
        specification = BACKBONE_SPECS.get(str(backbone))
        if (
            not isinstance(model, Mapping)
            or specification is None
            or model.get("revision") != specification.revision
            or set(rows["backbone_revision"]) != {specification.revision}
        ):
            return False
    expected_pairs = set(EXPECTED_PILOT_PAIRS)
    expected_shoes = set(EXPECTED_PILOT_SHOES)
    return (
        set(frame["config_sha256"]) == {config_sha256}
        and set(frame["camera_provenance"]) == {"fixed_precomputed_da3"}
        and set(frame["camera_label"]) == {CAMERA_MODE_LABEL}
        and set(frame["pair_id"]) <= expected_pairs
        and set(frame["shoe_id"]) == expected_shoes
        and set(frame["first_view"]) | set(frame["second_view"]) <= set(UPPER_VIEWS)
        and "bottom" not in set(frame["first_view"]) | set(frame["second_view"])
    )


def _finite_metrics(frame: pd.DataFrame) -> bool:
    required = {"score", "s00", "s10", "s01", "s11", "abstained"}
    if not required <= set(frame):
        return False
    metric_columns = [
        column
        for column in frame.columns
        if column in {"score", "s00", "s10", "s01", "s11", "interaction"}
        or column.endswith("_score")
        or column.endswith("_effect")
    ]
    active = frame.loc[~frame["abstained"].fillna(True).astype(bool), metric_columns]
    if active.empty:
        return True
    return bool(np.isfinite(active.to_numpy(dtype=np.float64)).all())


def _clean_directional_coverage(frame: pd.DataFrame) -> float:
    required = {
        "corruption",
        "s11_first_to_second_score",
        "s11_second_to_first_score",
        "s11_first_to_second_overlap",
        "s11_second_to_first_overlap",
    }
    if not required <= set(frame):
        return 0.0
    clean = frame.loc[frame["corruption"] == "clean"]
    if clean.empty:
        return 0.0
    covered = 0
    for prefix in ("s11_first_to_second", "s11_second_to_first"):
        score = pd.to_numeric(clean[f"{prefix}_score"], errors="coerce")
        overlap = pd.to_numeric(clean[f"{prefix}_overlap"], errors="coerce")
        covered += int((np.isfinite(score) & np.isfinite(overlap) & (overlap >= 0.10)).sum())
    return covered / (2 * len(clean))


def _sentinel_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column in {
            "score",
            "s00",
            "s10",
            "s01",
            "s11",
            "interaction",
            "overlap",
            "cycle_error_px_median",
            "relative_depth_error_median",
        }
        or column.endswith("_score")
        or column.endswith("_overlap")
        or column.endswith("_effect")
    ]


def _force_sentinel_rerun(
    root: Path,
    runner: Callable[[str | Path, str, str], object],
) -> Mapping[str, object]:
    rows = _registry_rows(root)
    candidates = [
        row
        for row in rows
        if row.state == "completed" and _track_for_job(row) == "controlled"
    ]
    if not candidates:
        raise ValueError("no completed controlled feature job is available as sentinel")
    sentinel = sorted(
        candidates,
        key=lambda row: (
            row.payload.get("state_id") != "controlled__"
            f"{EXPECTED_PILOT_SHOES[0]}__clean",
            row.job_id,
        ),
    )[0]
    parts = [
        path
        for path in _manifest_files(sentinel, root)
        if path.suffix == ".parquet"
        and path.is_relative_to((root / "metrics" / "parts").resolve())
    ]
    if len(parts) != 1:
        raise ValueError("sentinel job must register exactly one metric part")
    original_part = parts[0]
    original = pd.read_parquet(original_part)
    original_hash = sha256_file(original_part)
    with sqlite3.connect(root / "registry/jobs.sqlite", timeout=60) as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            """
            UPDATE jobs
            SET state = 'pending', worker_id = NULL, attempts = 0,
                started_at = NULL, finished_at = NULL,
                output_manifest_json = NULL, error = NULL
            WHERE job_id = ? AND state = 'completed'
            """,
            (sentinel.job_id,),
        )
        if updated.rowcount != 1:
            raise ValueError("sentinel job lost its completed-state transaction")
        connection.commit()
    backbone = sentinel.payload.get("backbone")
    if not isinstance(backbone, str):
        raise ValueError("sentinel job backbone is malformed")
    runner(root, backbone, "cuda:0")
    rerun_rows = {row.job_id: row for row in _registry_rows(root)}
    rerun = rerun_rows[sentinel.job_id]
    if rerun.state != "completed":
        raise ValueError(f"sentinel worker did not complete the job: {rerun.state}")
    registry = JobRegistry(root / "registry/jobs.sqlite")
    if registry.validated_state(sentinel.job_id) != "completed":
        raise ValueError("sentinel rerun output failed registry hash validation")
    rerun_parts = [
        path
        for path in _manifest_files(rerun, root)
        if path.suffix == ".parquet"
        and path.is_relative_to((root / "metrics" / "parts").resolve())
    ]
    if len(rerun_parts) != 1:
        raise ValueError("sentinel rerun must register exactly one metric part")
    rerun_part = rerun_parts[0]
    rerun_frame = pd.read_parquet(rerun_part)
    keys = list(_primary_key("controlled"))
    original_sorted = original.sort_values(keys, kind="mergesort").reset_index(drop=True)
    rerun_sorted = rerun_frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    design_ids_match = original_sorted[keys].equals(rerun_sorted[keys])
    metric_columns = _sentinel_metric_columns(original_sorted)
    metrics_match = (
        set(metric_columns) <= set(rerun_sorted)
        and len(original_sorted) == len(rerun_sorted)
    )
    if metrics_match:
        try:
            np.testing.assert_allclose(
                original_sorted[metric_columns].to_numpy(dtype=np.float64),
                rerun_sorted[metric_columns].to_numpy(dtype=np.float64),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )
        except AssertionError:
            metrics_match = False
    result = {
        "passed": bool(design_ids_match and metrics_match),
        "job_id": sentinel.job_id,
        "backbone": backbone,
        "design_ids_match": bool(design_ids_match),
        "metrics_match": bool(metrics_match),
        "rtol": 1e-5,
        "atol": 1e-6,
        "original_part_sha256": original_hash,
        "rerun_part_sha256": sha256_file(rerun_part),
    }
    _atomic_json(result, root / "reports" / "sentinel_rerun.json")
    return result


def _sentinel_valid(root: Path) -> bool:
    try:
        report = _json_object(root / "reports" / "sentinel_rerun.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    basic = (
        report.get("passed") is True
        and report.get("design_ids_match") is True
        and report.get("metrics_match") is True
        and report.get("rtol") == 1e-5
        and report.get("atol") == 1e-6
        and isinstance(report.get("job_id"), str)
        and isinstance(report.get("backbone"), str)
    )
    if not basic:
        return False
    try:
        row = next(
            item
            for item in _registry_rows(root)
            if item.job_id == report["job_id"]
        )
        parts = [
            path
            for path in _manifest_files(row, root)
            if path.suffix == ".parquet"
            and path.is_relative_to((root / "metrics" / "parts").resolve())
        ]
    except (OSError, ValueError, StopIteration):
        return False
    return (
        row.state == "completed"
        and row.payload.get("backbone") == report["backbone"]
        and len(parts) == 1
        and parts[0].is_file()
        and sha256_file(parts[0]) == report.get("rerun_part_sha256")
    )


def validate_pilot(
    run_root: str | Path,
    *,
    projected_bytes: int | None = None,
    rerun_sentinel: bool = False,
    sentinel_runner: Callable[[str | Path, str, str], object] | None = None,
) -> PilotGateReport:
    """Evaluate every operational pilot gate and atomically persist reports."""

    root = Path(run_root).expanduser().resolve()
    failures: set[str] = set()
    config_path = root / "config" / "frozen_config.json"
    config_content = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    config = json.loads(config_content)
    if not isinstance(config, Mapping):
        raise ValueError("frozen config must be an object")
    manifest = _json_object(root / "run_manifest.json")
    if manifest.get("scope") != "pilot" or manifest.get("config_sha256") != config_sha256:
        failures.add("config_provenance")
    if "git_commit" in manifest or "dirty_tree_fingerprint" in manifest:
        try:
            live_commit, live_dirty = _live_repository_provenance()
            if (
                manifest.get("git_commit") != live_commit
                or manifest.get("dirty_tree_fingerprint") != live_dirty
            ):
                failures.add("code_provenance")
        except (OSError, subprocess.CalledProcessError):
            failures.add("code_provenance")
    pilot = config.get("pilot")
    if (
        tuple(config.get("views", ())) != UPPER_VIEWS
        or config.get("camera_label") != CAMERA_MODE_LABEL
        or config.get("seed") != 42
        or config.get("stage_size") != 504
        or config.get("physical_gpus") != [4, 5, 6, 7]
        or not isinstance(pilot, Mapping)
        or tuple(pilot.get("shoes", ())) != EXPECTED_PILOT_SHOES
        or tuple(pilot.get("pairs", ())) != EXPECTED_PILOT_PAIRS
    ):
        failures.add("pilot_design")
    metric = config.get("metric")
    models = config.get("models")
    if (
        not isinstance(metric, Mapping)
        or metric.get("relative_depth_tolerance") != 0.03
        or metric.get("min_overlap_fraction") != 0.10
        or metric.get("confidence_percentile") != 10.0
        or not isinstance(models, Mapping)
        or models.get("naf_revision") != NAF_REVISION
    ):
        failures.add("config_provenance")
    design: _PilotDesign | None
    try:
        design = _load_pilot_design(root, config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        design = None
        failures.add("design_manifests")

    rows = _registry_rows(root)
    failures.update(_validate_registry(root, rows, config_sha256))
    if not _validate_geometry_provenance(root, rows, config):
        failures.add("model_provenance")

    if rerun_sentinel:
        if sentinel_runner is None:
            from .upper_metric_worker import run_feature_worker

            sentinel_runner = run_feature_worker
        try:
            sentinel = _force_sentinel_rerun(root, sentinel_runner)
            if not sentinel.get("passed"):
                failures.add("sentinel_determinism")
        except Exception as error:
            failures.add("sentinel_determinism")
            _atomic_json(
                {
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "rtol": 1e-5,
                    "atol": 1e-6,
                },
                root / "reports" / "sentinel_rerun.json",
            )
    if not _sentinel_valid(root):
        failures.add("sentinel_determinism")

    # Read paths from the pre-validation snapshot too, so a hash mismatch cannot
    # hide non-finite content by resetting its job to pending.
    try:
        if design is None:
            frame = _read_parts(_part_paths(root, rows, "controlled"))
            ownership_valid = False
        else:
            frame, ownership_valid = _read_controlled_parts_with_ownership(
                root,
                rows,
                design,
            )
    except (OSError, ValueError, pa.ArrowException):
        frame = pd.DataFrame()
        ownership_valid = False
        failures.add("metric_parts")
    if not ownership_valid:
        failures.add("design_ownership")
    actual_rows = len(frame)
    if actual_rows != EXPECTED_PILOT_METRIC_ROWS:
        failures.add("metric_row_count")
    if not frame.empty:
        try:
            checked = _validate_frame_primary_key(frame, "controlled")
            if (
                design is None
                or frozenset(
                    zip(checked["design_id"], checked["variant_key"])
                )
                != design.expected_primary_keys
            ):
                failures.add("metric_primary_keys")
        except ValueError:
            failures.add("metric_primary_keys")
    if not _metric_provenance_valid(frame, config, config_sha256):
        failures.add("model_provenance")
    if not _finite_metrics(frame):
        failures.add("finite_metrics")
    coverage = _clean_directional_coverage(frame)
    if coverage < 0.90:
        failures.add("clean_directional_coverage")
    if not _validate_pca(root, config, config_sha256):
        failures.add("pca_fingerprint")

    raw_features = _raw_feature_artifacts(root)
    if raw_features:
        failures.add("raw_feature_artifact")
    projection = dict(_storage_projection(root, actual_rows))
    if projected_bytes is not None:
        if projected_bytes < 0:
            raise ValueError("projected_bytes must be non-negative")
        projection["measured_projection_total_bytes"] = projection[
            "projected_total_bytes"
        ]
        projection["projected_total_bytes"] = int(projected_bytes)
        projection["override_for_validation"] = True
    projected_total = int(projection["projected_total_bytes"])
    limit_gb = config.get("persistent_storage_limit_gb")
    if not isinstance(limit_gb, (int, float)) or projected_total > float(limit_gb) * (
        1024**3
    ):
        failures.add("storage_limit")
    _atomic_json(projection, root / "reports" / "pilot_storage_projection.json")
    report = PilotGateReport(
        passed=not failures,
        failed_checks=tuple(sorted(failures)),
        expected_metric_rows=EXPECTED_PILOT_METRIC_ROWS,
        actual_metric_rows=actual_rows,
        clean_directional_coverage=coverage,
        raw_feature_artifacts=raw_features,
        projected_total_bytes=projected_total,
    )
    _atomic_json(asdict(report), root / "reports" / "pilot_gate.json")
    return report
