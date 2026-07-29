"""Fail-closed loader for the frozen adjacent 16-view experiment protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .upper_design import FeatureVariant


_EXPERIMENT_ID = "20260727_adjacent16_core_v1"
_RECONSTRUCTION_ROOT = Path("/root/dev/ByteSize/Reconstruction")
_OUTPUT_ROOT = _RECONSTRUCTION_ROOT / "met3r_upper_experiments" / _EXPERIMENT_ID
_STRESS_ROOT = _OUTPUT_ROOT / "inputs" / "stress"
_CAMERA_ORDER = (
    "0", "11", "22", "33", "43", "44", "45", "46", "47", "1", "2", "3",
    "4", "5", "6", "7", "8", "9", "10", "12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "34", "35", "36", "37", "38", "39", "40", "41", "42",
)
_QUARTETS = (
    ("q0", ("1", "2", "3", "4")),
    ("q1", ("14", "15", "16", "17")),
    ("q2", ("27", "28", "29", "30")),
    ("q3", ("40", "41", "42", "0")),
)
_FROZEN_VARIANTS = (
    ("dino_vitb16", "bilinear", 37),
    ("dinov2_vitb14", "bilinear", 37),
    ("dinov3_vitb16", "bilinear", 37),
    ("dinov3_vitl16_pixal3d", "bilinear", 37),
    ("dinov3_vitl16_pixal3d", "bilinear", 74),
    ("dinov3_vitl16_pixal3d", "naf_normal_guide", 74),
    ("dinov3_vitl16_pixal3d", "bilinear", 148),
    ("dinov3_vitl16_pixal3d", "naf_normal_guide", 148),
    ("dinov3_vitl16_pixal3d", "bilinear", 296),
    ("dinov3_vitl16_pixal3d", "naf_normal_guide", 296),
    ("dinov3_vitl16_pixal3d", "bilinear", 592),
    ("dinov3_vitl16_pixal3d", "naf_normal_guide", 592),
)
_GEOMETRY_VARIANT_KEYS = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37",
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592",
)
_EXPECTED_CORRUPTIONS = {
    "translation": [0.03, 0.06, 0.12, 0.2],
    "thin_plate_warp": [0.25, 0.5, 0.75, 1.0],
    "jpeg": [0.25, 0.5, 0.75, 1.0],
    "identity_swap": [1.0],
}
_EXPECTED_MODELS = {
    "dino_vitb16": {"revision": "7c446df5b9f45747937fb0d72314eb9f7b66930a"},
    "dinov2_vitb14": {"revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415"},
    "dinov3_vitb16": {"revision": "5931719e67bbdb9737e363e781fb0c67687896bc"},
    "dinov3_vitl16_pixal3d": {"revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881"},
    "naf_revision": "37f2dfc180f2de53d98bd601109c0da0dd6b0f43",
    "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
    "da3_revision": "8615eefb62f2db4f8d6ebaa59160086981672829",
}
_EXPECTED_GEOMETRY = {
    "rotation_degrees": [0.25, 0.5, 1.0, 2.0, 5.0],
    "relative_translation": [0.0025, 0.005, 0.01, 0.02],
    "depth_scale": [1.01, 1.02, 1.05, 1.1],
    "relative_depth_noise_std": [0.005, 0.01, 0.02, 0.05],
}
_EXPECTED_PREFLIGHT = {
    "minimum_adjacent_orbit_degrees": 5.0,
    "maximum_adjacent_orbit_degrees": 10.0,
    "minimum_processed_image_correlation": 0.99,
    "minimum_symmetric_pass_rate": 0.9,
    "minimum_median_directional_overlap": 0.2,
    "minimum_per_shoe_pair_coverage": 0.8,
}
_EXPECTED_RUNTIME = {
    "physical_gpu_count": 4,
    "maximum_projected_hours": 24.0,
    "feature_wall_time_multiplier": 1.25,
    "fixed_finalize_hours": 2.0,
}
_EXPECTED_ANALYZABILITY = {
    "minimum_controlled_complete_series_coverage": 0.8,
    "minimum_all_four_shoe_primary_cell_rate": 0.8,
    "minimum_common_support_retention": 0.7,
    "minimum_stress_symmetric_pass_rate": 0.7,
    "maximum_unexplained_job_failures": 0,
}
_EXPECTED_SHOES = {
    "air_huarache_run_ultra_111449": "AIR HUARACHE RUN ULTRA(111449)",
    "air_max_90_nrg_dc6083_500": "AIR MAX 90 NRG(DC6083-500)",
    "air_max_90_surplus_cq7743_300": "AIR MAX 90 SURPLUS(CQ7743-300)",
    "air_tuned_max_cv6984_001": "AIR TUNED MAX(CV6984-001)",
}
_TOP_LEVEL_KEYS = frozenset(
    {
        "experiment_id", "reconstruction_root", "output_root", "stress_root", "source_root",
        "camera_label", "source_size", "stage_size", "seed", "camera_order", "quartets",
        "shoes", "metric", "physical_gpus", "corruptions", "models", "feature_variants",
        "geometry_calibration_variant_keys", "visualization_variant_keys", "geometry_calibration",
        "visualization", "preflight", "runtime", "analyzability", "expected_rows",
    }
)


@dataclass(frozen=True)
class QuartetSpec:
    quartet_id: str
    view_ids: tuple[str, str, str, str]


@dataclass(frozen=True)
class Adjacent16Contract:
    experiment_id: str
    output_root: Path
    reconstruction_root: Path
    stress_root: Path
    physical_gpus: tuple[int, int, int, int]
    camera_order: tuple[str, ...]
    quartets: tuple[QuartetSpec, ...]
    feature_variants: tuple[FeatureVariant, ...]
    geometry_calibration_variant_keys: tuple[str, str]
    visualization_variant_keys: tuple[str, str]
    expected_rows: Mapping[str, int]
    raw: Mapping[str, object]


def _same_json_value(actual: object, expected: object) -> bool:
    """Compare JSON-shaped values while preserving scalar type distinctions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_json_value(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_json_value(value, wanted) for value, wanted in zip(actual, expected))
        )
    return actual == expected


def _require_exact(actual: object, expected: object, field: str) -> None:
    if not _same_json_value(actual, expected):
        raise ValueError(f"{field} deviates from the frozen adjacent16 protocol")


def _expected_feature_records() -> list[dict[str, object]]:
    return [
        {
            "feature_track": "user_37_grid_track",
            "backbone": backbone,
            "input_size": 592,
            "feature_mode": mode,
            "resolution": resolution,
        }
        for backbone, mode, resolution in _FROZEN_VARIANTS
    ]


def _expected_rows() -> dict[str, int]:
    pair_units = len(_EXPECTED_SHOES) * len(_QUARTETS) * 6
    changed_strengths = sum(len(values) for values in _EXPECTED_CORRUPTIONS.values())
    controlled_per_variant = pair_units + pair_units * 2 * changed_strengths
    geometry_magnitudes = sum(len(values) for values in _EXPECTED_GEOMETRY.values())
    controlled = controlled_per_variant * len(_FROZEN_VARIANTS)
    geometry = pair_units * 2 * geometry_magnitudes * len(_GEOMETRY_VARIANT_KEYS)
    stress = pair_units * len(_FROZEN_VARIANTS)
    return {
        "controlled": controlled,
        "geometry_calibration": geometry,
        "stress": stress,
        "total": controlled + geometry + stress,
    }


def _parse_quartets(raw_quartets: object) -> tuple[QuartetSpec, ...]:
    expected = [
        {"quartet_id": quartet_id, "view_ids": list(view_ids)}
        for quartet_id, view_ids in _QUARTETS
    ]
    _require_exact(raw_quartets, expected, "quartets")
    assert isinstance(raw_quartets, list)
    return tuple(
        QuartetSpec(
            quartet_id=item["quartet_id"],
            view_ids=tuple(item["view_ids"]),  # type: ignore[arg-type]
        )
        for item in raw_quartets
    )


def _parse_variants(raw_variants: object) -> tuple[FeatureVariant, ...]:
    _require_exact(raw_variants, _expected_feature_records(), "feature_variants")
    assert isinstance(raw_variants, list)
    return tuple(FeatureVariant(**item) for item in raw_variants)


def load_adjacent16_contract(path: str | Path) -> Adjacent16Contract:
    """Load and fail closed on any deviation from the frozen protocol."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load adjacent16 contract {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("adjacent16 contract must be a JSON object")
    if frozenset(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("adjacent16 contract keys deviate from the frozen protocol")

    _require_exact(raw["experiment_id"], _EXPERIMENT_ID, "experiment_id")
    _require_exact(raw["reconstruction_root"], str(_RECONSTRUCTION_ROOT), "reconstruction_root")
    _require_exact(raw["output_root"], str(_OUTPUT_ROOT), "output_root")
    _require_exact(raw["stress_root"], str(_STRESS_ROOT), "stress_root")
    _require_exact(raw["source_root"], str(_RECONSTRUCTION_ROOT / "sample_result"), "source_root")
    _require_exact(
        raw["camera_label"],
        "fixed/precomputed DA3 cameras (not calibrated GT)",
        "camera_label",
    )
    _require_exact(raw["source_size"], 1080, "source_size")
    _require_exact(raw["stage_size"], 504, "stage_size")
    _require_exact(raw["seed"], 42, "seed")
    _require_exact(raw["camera_order"], list(_CAMERA_ORDER), "camera_order")
    quartets = _parse_quartets(raw["quartets"])
    _require_exact(raw["shoes"], _EXPECTED_SHOES, "shoes")
    _require_exact(
        raw["metric"],
        {
            "relative_depth_tolerance": 0.03,
            "min_overlap_fraction": 0.08,
            "confidence_percentile": 10.0,
        },
        "metric",
    )
    _require_exact(raw["physical_gpus"], [0, 1, 2, 3], "physical_gpus")
    _require_exact(raw["corruptions"], _EXPECTED_CORRUPTIONS, "corruptions")
    _require_exact(raw["models"], _EXPECTED_MODELS, "models")
    variants = _parse_variants(raw["feature_variants"])
    _require_exact(
        raw["geometry_calibration_variant_keys"],
        list(_GEOMETRY_VARIANT_KEYS),
        "geometry_calibration_variant_keys",
    )
    _require_exact(
        raw["visualization_variant_keys"],
        list(_GEOMETRY_VARIANT_KEYS),
        "visualization_variant_keys",
    )
    _require_exact(raw["visualization"], {"policy": "posthoc_audit_only"}, "visualization")
    _require_exact(raw["geometry_calibration"], _EXPECTED_GEOMETRY, "geometry_calibration")
    _require_exact(raw["preflight"], _EXPECTED_PREFLIGHT, "preflight")
    _require_exact(raw["runtime"], _EXPECTED_RUNTIME, "runtime")
    _require_exact(raw["analyzability"], _EXPECTED_ANALYZABILITY, "analyzability")
    expected_rows = _expected_rows()
    _require_exact(raw["expected_rows"], expected_rows, "expected_rows")

    return Adjacent16Contract(
        experiment_id=_EXPERIMENT_ID,
        output_root=_OUTPUT_ROOT,
        reconstruction_root=_RECONSTRUCTION_ROOT,
        stress_root=_STRESS_ROOT,
        physical_gpus=(0, 1, 2, 3),
        camera_order=_CAMERA_ORDER,
        quartets=quartets,
        feature_variants=variants,
        geometry_calibration_variant_keys=_GEOMETRY_VARIANT_KEYS,
        visualization_variant_keys=_GEOMETRY_VARIANT_KEYS,
        expected_rows=MappingProxyType(expected_rows),
        raw=MappingProxyType(raw),
    )
