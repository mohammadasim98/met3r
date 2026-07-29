"""Deterministic post-hoc visualization for the adjacent16 numeric audit."""

from __future__ import annotations

import json
import math
import os
import gc
import fcntl
import hashlib
import shutil
import tempfile
import uuid
from contextlib import closing, contextmanager
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn

from .experiment import score_multiscale_factorial
from .features import (
    BACKBONE_SPECS,
    NAF_REVISION,
    MultiScaleFeatureExtractor,
    load_feature_backbone,
    load_naf_upsampler,
)
from .job_registry import sha256_file
from .pca_features import PCABasis, fit_pca_basis, project_pca
from .pilot_gate import resolve_aggregate_generation
from .streaming_features import iter_feature_variants
from .upper_design import FeatureVariant
from .upper_metric_worker import (
    _clean_state_for_shoe,
    _load_geometry,
    _pair_rows,
    _load_run_contract,
    _load_state_index,
    _state_images,
    _validate_adjacent_pair_contract,
    select_geometry,
)
from .visualization import save_direction_diagnostics


_LOW_VARIANT = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37"
)
_HIGH_VARIANT = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592"
)
_PAIR_IDENTITY = (
    "track",
    "shoe_id",
    "quartet_id",
    "pair_id",
    "first_view",
    "second_view",
    "baseline_steps",
    "angular_separation_deg",
    "changed_view",
    "image_state_id",
    "geometry_contract",
    "corruption",
    "severity",
)
_VALIDITY_COLUMNS = (
    "s00",
    "s10",
    "s11",
    "score",
    "s11_first_to_second_overlap",
    "s11_second_to_first_overlap",
)
_BACKBONE = "dinov3_vitl16_pixal3d"
_INPUT_SIZE = 592
_PCA_SEED = 42
_PCA_MAX_TOKENS = 50_000
_VARIANTS = (
    FeatureVariant(
        "user_37_grid_track",
        _BACKBONE,
        _INPUT_SIZE,
        "bilinear",
        37,
    ),
    FeatureVariant(
        "user_37_grid_track",
        _BACKBONE,
        _INPUT_SIZE,
        "naf_normal_guide",
        592,
    ),
)


def _atomic_json(destination: Path, payload: dict[str, object]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _paired_variants(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "design_id",
        "variant_key",
        "overlap",
        "abstained",
        *_PAIR_IDENTITY,
        *_VALIDITY_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} aggregate is missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["variant_key"].isin((_LOW_VARIANT, _HIGH_VARIANT))
    ].copy()
    low = selected.loc[selected["variant_key"] == _LOW_VARIANT].copy()
    high = selected.loc[selected["variant_key"] == _HIGH_VARIANT].copy()
    if low.empty or high.empty:
        raise ValueError(f"{label} aggregate requires exact low/high variants")
    if low["design_id"].duplicated().any() or high["design_id"].duplicated().any():
        raise ValueError(f"{label} aggregate has duplicate low/high design IDs")
    paired = low.merge(
        high,
        on="design_id",
        how="inner",
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    if len(paired) != len(low) or len(paired) != len(high):
        raise ValueError(f"{label} aggregate low/high support is not exact")
    for column in _PAIR_IDENTITY:
        low_column = paired[f"{column}_low"]
        high_column = paired[f"{column}_high"]
        equal = low_column.eq(high_column) | (
            low_column.isna() & high_column.isna()
        )
        if not bool(equal.all()):
            raise ValueError(
                f"{label} aggregate low/high identity differs for {column}"
            )
    for side in ("low", "high"):
        valid = ~paired[f"abstained_{side}"].fillna(True).astype(bool)
        for column in _VALIDITY_COLUMNS:
            valid &= np.isfinite(
                pd.to_numeric(paired[f"{column}_{side}"], errors="coerce")
            )
        paired[f"_valid_{side}"] = valid
        paired[f"_effect_{side}"] = (
            pd.to_numeric(paired[f"s10_{side}"], errors="coerce")
            - pd.to_numeric(paired[f"s00_{side}"], errors="coerce")
        )
    paired["_disagreement"] = (
        paired["_effect_high"] - paired["_effect_low"]
    ).abs()
    paired.loc[
        ~(paired["_valid_low"] & paired["_valid_high"]),
        "_disagreement",
    ] = np.nan
    paired["_joint_overlap"] = pd.concat(
        (
            pd.to_numeric(paired["overlap_low"], errors="coerce"),
            pd.to_numeric(paired["overlap_high"], errors="coerce"),
        ),
        axis=1,
    ).min(axis=1, skipna=False)
    return paired


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _case(
    row: pd.Series,
    *,
    role: str,
    stratum: int | str,
    selection_value: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "role": role,
        "stratum": stratum,
        "design_id": str(row["design_id"]),
    }
    for column in _PAIR_IDENTITY:
        value = row[f"{column}_low"]
        if pd.isna(value):
            result[column] = None
        elif column == "baseline_steps":
            result[column] = int(value)
        elif column in {"severity", "angular_separation_deg"}:
            result[column] = float(value)
        else:
            result[column] = str(value)
    result.update(
        {
            "low_valid": bool(row["_valid_low"]),
            "high_valid": bool(row["_valid_high"]),
            "joint_overlap": _finite_or_none(row["_joint_overlap"]),
            "low_effect": _finite_or_none(row["_effect_low"]),
            "high_effect": _finite_or_none(row["_effect_high"]),
            "finite_low_high_disagreement": _finite_or_none(
                row["_disagreement"]
            ),
            "selection_value": _finite_or_none(selection_value),
        }
    )
    return result


def _clean_cases(paired: pd.DataFrame) -> list[dict[str, object]]:
    clean = paired.loc[
        (paired["corruption_low"] == "clean")
        & np.isfinite(paired["_joint_overlap"])
    ]
    result: list[dict[str, object]] = []
    for steps in sorted(int(value) for value in clean["baseline_steps_low"].unique()):
        group = clean.loc[clean["baseline_steps_low"] == steps]
        if group.empty:
            continue
        minimum_order = group.sort_values(
            ["_joint_overlap", "design_id"],
            ascending=(True, True),
            kind="mergesort",
        )
        maximum_order = group.sort_values(
            ["_joint_overlap", "design_id"],
            ascending=(False, True),
            kind="mergesort",
        )
        median = float(np.median(group["_joint_overlap"].to_numpy(dtype=float)))
        middle = (
            group.assign(
                _median_distance=(group["_joint_overlap"] - median).abs()
            )
            .sort_values(
                ["_median_distance", "design_id"],
                kind="mergesort",
            )
            .iloc[0]
        )
        for role, row in (
            ("clean_overlap_minimum", minimum_order.iloc[0]),
            ("clean_overlap_median_nearest", middle),
            ("clean_overlap_maximum", maximum_order.iloc[0]),
        ):
            result.append(
                _case(
                    row,
                    role=role,
                    stratum=steps,
                    selection_value=row["_joint_overlap"],
                )
            )
    return result


def _maximum_corruption_cases(
    paired: pd.DataFrame,
) -> list[dict[str, object]]:
    corrupt = paired.loc[paired["corruption_low"] != "clean"].copy()
    corrupt["_severity_numeric"] = pd.to_numeric(
        corrupt["severity_low"],
        errors="coerce",
    )
    result: list[dict[str, object]] = []
    for corruption in sorted(str(value) for value in corrupt["corruption_low"].unique()):
        all_rows = corrupt.loc[corrupt["corruption_low"] == corruption]
        finite_severity = all_rows.loc[np.isfinite(all_rows["_severity_numeric"])]
        if finite_severity.empty:
            raise ValueError(f"corruption {corruption!r} has no finite severity")
        maximum = float(finite_severity["_severity_numeric"].max())
        group = finite_severity.loc[
            finite_severity["_severity_numeric"] == maximum
        ]
        jointly_valid = group.loc[
            group["_valid_low"] & group["_valid_high"]
        ].sort_values("design_id", kind="mergesort")
        if jointly_valid.empty:
            raise ValueError(
                f"maximum-severity corruption {corruption!r} has no jointly valid case"
            )
        representative = jointly_valid.iloc[0]
        result.append(
            _case(
                representative,
                role="maximum_corruption_jointly_valid",
                stratum=corruption,
                selection_value=maximum,
            )
        )
        low_valid_high_invalid = group.loc[
            group["_valid_low"] & ~group["_valid_high"]
        ].sort_values("design_id", kind="mergesort")
        if not low_valid_high_invalid.empty:
            selected = low_valid_high_invalid.iloc[0]
            result.append(
                _case(
                    selected,
                    role="maximum_corruption_low_valid_high_invalid",
                    stratum=corruption,
                    selection_value=maximum,
                )
            )
            continue
        disagreement = group.loc[
            np.isfinite(group["_disagreement"])
        ].sort_values(
            ["_disagreement", "design_id"],
            ascending=(False, True),
            kind="mergesort",
        )
        if disagreement.empty:
            raise ValueError(
                f"maximum-severity corruption {corruption!r} has no finite disagreement"
            )
        selected = disagreement.iloc[0]
        result.append(
            _case(
                selected,
                role="maximum_corruption_largest_finite_disagreement",
                stratum=corruption,
                selection_value=selected["_disagreement"],
            )
        )
    return result


def _stress_cases(paired: pd.DataFrame) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for steps in sorted(int(value) for value in paired["baseline_steps_low"].unique()):
        group = paired.loc[paired["baseline_steps_low"] == steps]
        disagreement = group.loc[
            np.isfinite(group["_disagreement"])
        ].sort_values(
            ["_disagreement", "design_id"],
            ascending=(False, True),
            kind="mergesort",
        )
        if disagreement.empty:
            raise ValueError(
                f"stress baseline step {steps} has no finite low/high disagreement"
            )
        selected = disagreement.iloc[0]
        result.append(
            _case(
                selected,
                role="stress_largest_finite_disagreement",
                stratum=steps,
                selection_value=selected["_disagreement"],
            )
        )
        low_valid_high_invalid = group.loc[
            group["_valid_low"] & ~group["_valid_high"]
        ].sort_values("design_id", kind="mergesort")
        if not low_valid_high_invalid.empty:
            selected = low_valid_high_invalid.iloc[0]
            result.append(
                _case(
                    selected,
                    role="stress_low_valid_high_invalid",
                    stratum=steps,
                    selection_value=None,
                )
            )
    return result


def build_adjacent16_audit_manifest(run_root: str | Path) -> Path:
    """Select deterministic numeric audit cases after aggregation."""

    root = Path(run_root).expanduser().resolve()
    destination = root / "reports/audit_visualizations"
    with _audit_publisher_lock(root):
        _recover_audit_destination(destination)
        if (
            (destination / "pca_render_manifest.json").exists()
            or (destination / "pca").exists()
            or (destination / "directional").exists()
        ):
            raise ValueError(
                "rendered audit snapshot is owned by render_adjacent16_audit"
            )
        payload = _build_audit_payload(root)
        return _atomic_json(
            destination / "audit_manifest.json",
            payload,
        )


def _build_audit_payload(
    root: Path,
    *,
    generation: Path | None = None,
) -> dict[str, Any]:
    if generation is None:
        generation = resolve_aggregate_generation(root)
    controlled = _paired_variants(
        pd.read_parquet(generation / "controlled_fidelity.parquet"),
        "controlled",
    )
    stress = _paired_variants(
        pd.read_parquet(generation / "real_deformation_stress.parquet"),
        "stress",
    )
    cases = [
        *_clean_cases(controlled),
        *_maximum_corruption_cases(controlled),
        *_stress_cases(stress),
    ]
    payload: dict[str, Any] = {
        "artifact_type": "adjacent16_numeric_audit_manifest",
        "selection_policy": "numeric_low_high_v1",
        "aggregate_generation": str(generation.relative_to(root)),
        "variant_keys": {
            "low": _LOW_VARIANT,
            "high": _HIGH_VARIANT,
        },
        "cases": cases,
    }
    return payload


def _assert_no_raw_artifacts(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        lower_stem = path.stem.lower()
        if suffix in {".pt", ".pth"}:
            raise ValueError(f"prohibited tensor artifact exists: {path}")
        if suffix in {".npy", ".npz"} and any(
            token in lower_stem
            for token in ("feature", "component", "basis")
        ):
            raise ValueError(
                f"prohibited feature, component, or basis array exists: {path}"
            )
        if suffix != ".npz":
            continue
        try:
            with np.load(path, allow_pickle=False) as artifact:
                if {
                    "components",
                    "pca_components",
                    "pca_basis",
                    "basis",
                } & set(artifact.files):
                    raise ValueError(
                        f"prohibited persisted PCA components exist: {path}"
                    )
                for field in ("artifact_type", "meta__artifact_type"):
                    if field not in artifact.files:
                        continue
                    marker = np.asarray(artifact[field]).item()
                    if isinstance(marker, bytes):
                        marker = marker.decode("utf-8", errors="replace")
                    if str(marker) == "dino_features":
                        raise ValueError(
                            f"prohibited DINO feature artifact exists: {path}"
                        )
        except ValueError as error:
            if "prohibited" in str(error):
                raise
            raise ValueError(f"cannot audit NPZ artifact for raw features: {path}") from error


def _canonical_pca_sources(
    root: Path,
) -> dict[tuple[str, str, str], tuple[Path, str]]:
    path = root / "manifests/adjacent16_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("canonical PCA source manifest is invalid") from error
    shoes = payload.get("shoes") if isinstance(payload, Mapping) else None
    if not isinstance(shoes, list):
        raise ValueError("canonical PCA source manifest requires shoes")
    result: dict[tuple[str, str, str], tuple[Path, str]] = {}
    for shoe in shoes:
        if not isinstance(shoe, Mapping) or not isinstance(
            shoe.get("shoe_id"), str
        ):
            raise ValueError("canonical PCA shoe record is malformed")
        views = shoe.get("views")
        if not isinstance(views, list):
            raise ValueError("canonical PCA shoe views are malformed")
        for view in views:
            if not isinstance(view, Mapping):
                raise ValueError("canonical PCA view record is malformed")
            identity = (
                str(shoe["shoe_id"]),
                str(view.get("quartet_id", "")),
                str(view.get("view_id", "")),
            )
            source = view.get("source_path")
            source_hash = view.get("source_sha256")
            if (
                any(not value for value in identity)
                or identity in result
                or not isinstance(source, str)
                or not isinstance(source_hash, str)
                or len(source_hash) != 64
            ):
                raise ValueError("canonical PCA identities or hashes are malformed")
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_file():
                raise ValueError(
                    f"canonical PCA source image is missing: {source_path}"
                )
            if sha256_file(source_path) != source_hash:
                raise ValueError(
                    f"canonical PCA source hash is stale: {source_path}"
                )
            result[identity] = (source_path, source_hash)
    if len(result) != 64:
        raise ValueError("canonical PCA source set must contain exactly 64 records")
    paths = [record[0] for record in result.values()]
    hashes = [record[1] for record in result.values()]
    if len(set(paths)) != 64 or len(set(hashes)) != 64:
        raise ValueError("canonical PCA source paths and hashes must be globally unique")
    return result


def _fit_source_records(root: Path) -> tuple[tuple[Path, ...], list[dict[str, str]]]:
    path = root / "pca_fit_images/clean_source_paths.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("clean PCA source manifest is invalid") from error
    if not isinstance(raw, list) or len(raw) != 64:
        raise ValueError("shared PCA fitting requires exactly 64 clean source images")
    owners = _canonical_pca_sources(root)
    canonical: list[tuple[tuple[str, str, str], Path, str]] = []
    identities: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("clean PCA source record is malformed")
        identity = tuple(
            str(item.get(field, ""))
            for field in ("shoe_id", "quartet_id", "view_id")
        )
        source = item.get("source_path")
        if (
            any(not value for value in identity)
            or identity in identities
            or not isinstance(source, str)
        ):
            raise ValueError("clean PCA source identities must be exact and unique")
        image_path = Path(source).expanduser().resolve()
        owner = owners.get(identity)
        if owner is None or image_path != owner[0]:
            raise ValueError(
                "clean PCA fit record differs from its canonical PCA source"
            )
        if sha256_file(image_path) != owner[1]:
            raise ValueError(
                f"clean PCA fit source hash differs from canonical PCA: {image_path}"
            )
        identities.add(identity)
        canonical.append((identity, image_path, owner[1]))
    if identities != set(owners):
        raise ValueError("clean PCA fit identities differ from canonical PCA sources")
    canonical.sort(key=lambda item: (*item[0], str(item[1])))
    paths: list[Path] = []
    records: list[dict[str, str]] = []
    for identity, image_path, source_hash in canonical:
        paths.append(image_path)
        records.append(
            {
                "shoe_id": identity[0],
                "quartet_id": identity[1],
                "view_id": identity[2],
                "path": str(image_path),
                "sha256": source_hash,
            }
        )
    return tuple(paths), records


def _aggregate_input_records(
    root: Path,
    contract: object,
    *,
    generation: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    expected_variants = {
        _LOW_VARIANT: ("bilinear", 37),
        _HIGH_VARIANT: ("naf_normal_guide", 592),
    }
    for name in (
        "controlled_fidelity.parquet",
        "real_deformation_stress.parquet",
    ):
        path = generation / name
        frame = pd.read_parquet(path)
        required = {
            "variant_key",
            "feature_track",
            "backbone",
            "backbone_revision",
            "config_sha256",
            "input_size",
            "feature_mode",
            "resolution",
        }
        if not required <= set(frame.columns):
            raise ValueError("aggregate provenance columns are incomplete")
        selected = frame.loc[frame["variant_key"].isin(expected_variants)]
        if selected.empty:
            raise ValueError("aggregate provenance has no audit variants")
        if (
            set(selected["config_sha256"]) != {contract.config_sha256}
            or set(selected["backbone_revision"])
            != {contract.backbone_revision}
            or set(selected["backbone"]) != {_BACKBONE}
            or set(selected["feature_track"]) != {"user_37_grid_track"}
            or set(pd.to_numeric(selected["input_size"], errors="coerce"))
            != {_INPUT_SIZE}
        ):
            raise ValueError("aggregate provenance differs from the frozen contract")
        for variant_key, (mode, resolution) in expected_variants.items():
            rows = selected.loc[selected["variant_key"] == variant_key]
            if (
                rows.empty
                or set(rows["feature_mode"]) != {mode}
                or set(pd.to_numeric(rows["resolution"], errors="coerce"))
                != {resolution}
            ):
                raise ValueError(
                    "aggregate provenance variant metadata differs from the "
                    "frozen contract"
                )
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
        )
    return records


def _fit_policy_fingerprint(
    fit_records: Sequence[Mapping[str, str]],
    *,
    token_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"adjacent16_shared_pca_v1\0")
    for value in (
        _PCA_SEED,
        _PCA_MAX_TOKENS,
        token_count,
        min(token_count, _PCA_MAX_TOKENS),
    ):
        digest.update(int(value).to_bytes(8, "big", signed=False))
    for record in fit_records:
        for field in ("shoe_id", "quartet_id", "view_id", "path", "sha256"):
            digest.update(field.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(record[field]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _load_source_tensor(
    paths: Sequence[Path],
    *,
    expected_hashes: Sequence[str],
    device: torch.device,
) -> Tensor:
    if len(paths) != len(expected_hashes):
        raise ValueError("PCA fit paths and hashes differ in length")
    tensors: list[Tensor] = []
    for path, expected_hash in zip(paths, expected_hashes, strict=True):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"PCA fit source hash changed before loading: {path}")
        with Image.open(path) as image:
            value = np.asarray(
                image.convert("RGB").resize(
                    (_INPUT_SIZE, _INPUT_SIZE),
                    Image.Resampling.LANCZOS,
                )
            ).copy()
        tensors.append(
            torch.from_numpy(value)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
    return torch.stack(tensors).to(device)


def _pca_batches(
    paths: Sequence[Path],
    extractor: MultiScaleFeatureExtractor,
    *,
    device: torch.device,
    counters: dict[str, int],
    expected_hashes: Sequence[str],
) -> Iterator[Tensor]:
    for start in range(0, len(paths), 4):
        images = _load_source_tensor(
            paths[start : start + 4],
            expected_hashes=expected_hashes[start : start + 4],
            device=device,
        )
        native = extractor.extract_native(images)
        _validate_native_feature(native)
        counters["tokens"] += (
            int(native.shape[0]) * int(native.shape[-2]) * int(native.shape[-1])
        )
        try:
            yield native
        finally:
            del native, images
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-+" else "_"
        for character in value
    )


def _validate_native_feature(native: Tensor) -> None:
    if native.ndim != 4 or tuple(native.shape[-2:]) != (37, 37):
        raise ValueError(
            "native DINOv3-L feature grid must be exactly 37 by 37"
        )


def _render_pca_images(
    feature: Tensor,
    basis: PCABasis,
    variant: FeatureVariant,
    *,
    design_id: str,
    view_ids: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for index, view_id in enumerate(view_ids):
        projected = project_pca(feature[index : index + 1], basis)[0]
        destination = output_dir / (
            f"{_safe_name(design_id)}__{_safe_name(variant.key)}__"
            f"{_safe_name(view_id)}.png"
        )
        Image.fromarray(projected, mode="RGB").save(destination)
        paths.append(destination)
        del projected
    return paths


def _case_matches_state(case: Mapping[str, object], state: object) -> None:
    for field in ("track", "shoe_id", "quartet_id", "corruption"):
        if case.get(field) != getattr(state, field):
            raise ValueError(f"audit case {field} differs from its image state")
    try:
        severity = float(case["severity"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("audit case severity is malformed") from error
    if severity != float(getattr(state, "severity")):
        raise ValueError("audit case severity differs from its image state")
    changed_view = case.get("changed_view")
    if changed_view != getattr(state, "changed_view"):
        raise ValueError("audit case changed_view differs from its image state")


def _identity_value(value: object) -> object:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _validate_cases_against_pair_manifest(
    root: Path,
    cases: Sequence[Mapping[str, object]],
) -> None:
    canonical: dict[str, Mapping[str, object]] = {}
    for row in _pair_rows(root):
        design_id = row.get("design_id")
        if not isinstance(design_id, str) or design_id in canonical:
            raise ValueError("canonical pair conditions have duplicate design IDs")
        canonical[design_id] = row
    for case in cases:
        design_id = str(case["design_id"])
        owner = canonical.get(design_id)
        if owner is None:
            raise ValueError(
                f"audit case {design_id!r} has no canonical pair condition"
            )
        for field in _PAIR_IDENTITY:
            selected = _identity_value(case.get(field))
            expected = _identity_value(
                (
                    "base_plus_adaptive"
                    if owner.get("track") == "stress"
                    else "fixed_base_only"
                )
                if field == "geometry_contract"
                else owner.get(field)
            )
            if field in {"severity", "angular_separation_deg"}:
                try:
                    equal = float(selected) == float(expected)
                except (TypeError, ValueError):
                    equal = False
            elif field == "baseline_steps":
                try:
                    equal = int(selected) == int(expected)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = selected == expected
            if not equal:
                raise ValueError(
                    f"audit case {design_id!r} differs from its canonical "
                    f"pair condition for {field}"
                )


def _validate_cases_against_states(
    root: Path,
    cases: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    states = _load_state_index(root)
    for case in cases:
        state_id = case.get("image_state_id")
        if not isinstance(state_id, str) or state_id not in states:
            raise ValueError("audit case references an unknown image state")
        _case_matches_state(case, states[state_id])
    return states


def _paired_variant_features(
    clean_native: Tensor,
    clean_images: Tensor,
    changed_native: Tensor,
    changed_images: Tensor,
    naf: nn.Module,
) -> Iterator[tuple[FeatureVariant, Tensor, Tensor]]:
    _validate_native_feature(clean_native)
    _validate_native_feature(changed_native)
    clean_iterator = iter_feature_variants(
        clean_native,
        clean_images,
        _VARIANTS,
        naf,
    )
    changed_iterator = iter_feature_variants(
        changed_native,
        changed_images,
        _VARIANTS,
        naf,
    )
    with closing(clean_iterator), closing(changed_iterator):
        for expected, clean_item, changed_item in zip(
            _VARIANTS,
            clean_iterator,
            changed_iterator,
            strict=True,
        ):
            clean_variant, clean_feature = clean_item
            changed_variant, changed_feature = changed_item
            if (
                clean_variant.key != expected.key
                or changed_variant.key != expected.key
            ):
                raise ValueError("audit replay variant order differs from the contract")
            wanted = (expected.resolution, expected.resolution)
            if (
                tuple(clean_feature.shape[-2:]) != wanted
                or tuple(changed_feature.shape[-2:]) != wanted
            ):
                raise ValueError(
                    f"variant resolution {expected.resolution} spatial outputs "
                    f"must be exactly {wanted}"
                )
            yield expected, clean_feature, changed_feature


def _replay_case(
    root: Path,
    case: Mapping[str, object],
    *,
    states: Mapping[str, object],
    extractor: MultiScaleFeatureExtractor,
    naf: nn.Module,
    basis: PCABasis,
    contract: object,
    device: torch.device,
    pca_dir: Path,
    directional_dir: Path,
) -> list[Path]:
    state_id = case.get("image_state_id")
    if not isinstance(state_id, str) or state_id not in states:
        raise ValueError("audit case references an unknown image state")
    state = states[state_id]
    _case_matches_state(case, state)
    clean_state = _clean_state_for_shoe(
        states,
        str(case["shoe_id"]),
        str(case["quartet_id"]),
    )
    _validate_adjacent_pair_contract(case, state, clean_state=clean_state)
    clean_images = _state_images(clean_state, _INPUT_SIZE, device)
    clean_native = extractor.extract_native(clean_images.tensor)
    changed_is_clean = state_id == clean_state.state_id
    if changed_is_clean:
        changed_images = clean_images
        changed_native = clean_native
    else:
        changed_images = _state_images(state, _INPUT_SIZE, device)
        changed_native = extractor.extract_native(changed_images.tensor)
    base_geometry = _load_geometry(
        root
        / "geometry"
        / "calibration"
        / clean_state.state_id
        / "depth_geometry.npz",
        device,
        expected_view_ids=clean_state.view_ids,
        validate_metadata=True,
    )
    adaptive_is_base = str(case["track"]) == "controlled"
    adaptive_geometry = (
        base_geometry
        if adaptive_is_base
        else _load_geometry(
            root
            / "geometry"
            / "stress"
            / state.state_id
            / "depth_geometry.npz",
            device,
            expected_view_ids=state.view_ids,
            validate_metadata=True,
        )
    )
    view_index = {view_id: index for index, view_id in enumerate(state.view_ids)}
    try:
        first = view_index[str(case["first_view"])]
        second = view_index[str(case["second_view"])]
    except KeyError as error:
        raise ValueError("audit pair is outside its image state") from error
    paths: list[Path] = []
    try:
        for variant, clean_feature, changed_feature in _paired_variant_features(
            clean_native,
            clean_images.tensor,
            changed_native,
            changed_images.tensor,
            naf,
        ):
            paths.extend(
                _render_pca_images(
                    changed_feature,
                    basis,
                    variant,
                    design_id=str(case["design_id"]),
                    view_ids=state.view_ids,
                    output_dir=pca_dir,
                )
            )
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
            stem = (
                f"{_safe_name(str(case['design_id']))}__"
                f"{_safe_name(variant.key)}"
            )
            first_path = directional_dir / f"{stem}__first_to_second.png"
            second_path = directional_dir / f"{stem}__second_to_first.png"
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
            paths.extend((first_path, second_path))
            del result, clean_feature, changed_feature
    finally:
        del clean_native, base_geometry, clean_images
        if not adaptive_is_base:
            del adaptive_geometry
        if not changed_is_clean:
            del changed_native, changed_images
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return paths


def _complete_audit_snapshot(path: Path) -> bool:
    return (
        (path / "audit_manifest.json").is_file()
        and (path / "pca_render_manifest.json").is_file()
        and (path / "pca").is_dir()
        and (path / "directional").is_dir()
    )


def _recover_audit_destination(destination: Path) -> None:
    stale_backups = sorted(
        destination.parent.glob(".audit_visualizations.backup.*")
    )
    if destination.exists():
        for stale in stale_backups:
            try:
                shutil.rmtree(stale)
            except OSError:
                pass
    elif len(stale_backups) == 1:
        if not _complete_audit_snapshot(stale_backups[0]):
            raise RuntimeError("retained audit backup is incomplete")
        os.replace(stale_backups[0], destination)
    elif len(stale_backups) > 1:
        raise RuntimeError(
            "ambiguous audit recovery: destination is absent and multiple "
            "backups are present"
        )


@contextmanager
def _audit_publisher_lock(root: Path):
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    lock_path = metrics / ".audit-visualizations-publisher.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_audit(staging: Path, destination: Path) -> None:
    _recover_audit_destination(destination)
    backup = destination.parent / (
        f".audit_visualizations.backup.{uuid.uuid4().hex}"
    )
    had_destination = destination.exists()
    try:
        if had_destination:
            os.replace(destination, backup)
        os.replace(staging, destination)
    except Exception:
        if destination.exists() and not had_destination:
            shutil.rmtree(destination)
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            pass


def _render_adjacent16_audit_locked(
    run_root: str | Path,
    *,
    device: str,
) -> tuple[Path, ...]:
    """Replay selected L37/L592 cases and persist PNG diagnostics only."""

    root = Path(run_root).expanduser().resolve()
    _assert_no_raw_artifacts(root)
    contract = _load_run_contract(root, _BACKBONE)
    configured = contract.config.get("visualization_variant_keys")
    if configured != [_LOW_VARIANT, _HIGH_VARIANT]:
        raise ValueError("adjacent audit variants differ from the frozen contract")
    specification = BACKBONE_SPECS[_BACKBONE]
    if contract.backbone_revision != specification.revision:
        raise ValueError("adjacent audit model revision is not pinned")
    torch_device = torch.device(device)
    generation = resolve_aggregate_generation(root)
    audit_payload = _build_audit_payload(root, generation=generation)
    raw_cases = audit_payload["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("adjacent audit cases are malformed")
    cases: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, Mapping) or not isinstance(
            case.get("design_id"), str
        ):
            raise ValueError("adjacent audit case is malformed")
        if str(case["design_id"]) not in seen:
            seen.add(str(case["design_id"]))
            cases.append(case)
    _validate_cases_against_pair_manifest(root, cases)
    states = _validate_cases_against_states(root, cases)
    aggregate_inputs = _aggregate_input_records(
        root,
        contract,
        generation=generation,
    )

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".audit_visualizations.staging.",
            dir=reports,
        )
    )
    pca_dir = staging / "pca"
    directional_dir = staging / "directional"
    pca_dir.mkdir()
    directional_dir.mkdir()
    try:
        _atomic_json(staging / "audit_manifest.json", audit_payload)
        fit_paths, fit_records = _fit_source_records(root)
        fit_hashes = tuple(record["sha256"] for record in fit_records)
        backbone = load_feature_backbone(_BACKBONE).to(torch_device)
        naf = load_naf_upsampler(device=torch_device).to(torch_device)
        extractor = MultiScaleFeatureExtractor(
            backbone,
            naf_upsampler=naf,
        ).to(torch_device)
        extractor.eval()
        pca_counters = {"tokens": 0}
        basis = fit_pca_basis(
            _pca_batches(
                fit_paths,
                extractor,
                device=torch_device,
                counters=pca_counters,
                expected_hashes=fit_hashes,
            ),
            max_tokens=_PCA_MAX_TOKENS,
            seed=_PCA_SEED,
        )
        if pca_counters["tokens"] <= _PCA_MAX_TOKENS:
            raise ValueError(
                "shared PCA source tokens do not exercise deterministic subsampling"
        )
        output_paths: list[Path] = []
        for case in cases:
            output_paths.extend(
                _replay_case(
                    root,
                    case,
                    states=states,
                    extractor=extractor,
                    naf=naf,
                    basis=basis,
                    contract=contract,
                    device=torch_device,
                    pca_dir=pca_dir,
                    directional_dir=directional_dir,
                )
            )
        output_paths = sorted(set(output_paths))
        render_payload = {
            "artifact_type": "adjacent16_in_memory_shared_pca",
            "aggregate_generation": audit_payload["aggregate_generation"],
            "config_sha256": contract.config_sha256,
            "audit_manifest_sha256": sha256_file(
                staging / "audit_manifest.json"
            ),
            "aggregate_inputs": aggregate_inputs,
            "fit_images": fit_records,
            "fit_policy_fingerprint": _fit_policy_fingerprint(
                fit_records,
                token_count=pca_counters["tokens"],
            ),
            "pca_fingerprint": basis.fingerprint,
            "global_bounds": {
                "lower": [float(value) for value in basis.lower],
                "upper": [float(value) for value in basis.upper],
            },
            "model_revision": contract.backbone_revision,
            "naf_revision": NAF_REVISION,
            "seed": _PCA_SEED,
            "max_tokens": _PCA_MAX_TOKENS,
            "fit_token_count": pca_counters["tokens"],
            "fit_sample_count": _PCA_MAX_TOKENS,
            "variant_keys": [_LOW_VARIANT, _HIGH_VARIANT],
            "outputs": [
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": sha256_file(path),
                }
                for path in output_paths
            ],
        }
        _atomic_json(
            staging / "pca_render_manifest.json",
            render_payload,
        )
        del basis, extractor, naf, backbone
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        _assert_no_raw_artifacts(root)
        destination = reports / "audit_visualizations"
        _publish_audit(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return tuple(
        path
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    )


def render_adjacent16_audit(
    run_root: str | Path,
    *,
    device: str,
) -> tuple[Path, ...]:
    """Replay selected L37/L592 cases and persist PNG diagnostics only."""

    root = Path(run_root).expanduser().resolve()
    with _audit_publisher_lock(root):
        return _render_adjacent16_audit_locked(root, device=device)
