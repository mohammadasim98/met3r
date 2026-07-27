from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from met3r.adjacent16_design import (
    Adjacent16Design,
    AdjacentImageState,
    AdjacentPairCondition,
)
from met3r.pilot_gate import (
    aggregate_all_metric_parts,
    resolve_aggregate_generation,
    validate_adjacent16_metric_ownership,
)
from met3r.upper_design import FeatureVariant
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file


_CALIBRATION_MAGNITUDES = {
    "rotation": (0.25, 0.5, 1.0, 2.0, 5.0),
    "relative_translation": (0.0025, 0.005, 0.01, 0.02),
    "depth_scale": (1.01, 1.02, 1.05, 1.1),
    "relative_depth_noise": (0.005, 0.01, 0.02, 0.05),
}


def _row(
    condition: AdjacentPairCondition,
    variant: FeatureVariant,
    job_id: str,
    analysis_track: str,
    *,
    changed_view: str | None = None,
    perturbation_kind: str | None = None,
    magnitude: float | None = None,
) -> dict[str, object]:
    return {
        **asdict(condition),
        "changed_view": changed_view if changed_view is not None else condition.changed_view,
        "job_id": job_id,
        "analysis_track": analysis_track,
        "variant_key": variant.key,
        "feature_track": variant.feature_track,
        "backbone": variant.backbone,
        "input_size": variant.input_size,
        "feature_mode": variant.feature_mode,
        "resolution": variant.resolution,
        "perturbation_kind": perturbation_kind,
        "magnitude": magnitude,
        "score": 0.8,
    }


def _complete(registry: JobRegistry, job_id: str, path: Path) -> None:
    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            "UPDATE jobs SET state='completed', output_manifest_json=? WHERE job_id=?",
            (
                json.dumps({"files": [{"path": str(path.resolve()), "sha256": sha256_file(path)}]}),
                job_id,
            ),
        )


@pytest.fixture
def owned_adjacent_run(tmp_path: Path) -> tuple[Path, Adjacent16Design]:
    clean = AdjacentImageState(
        "controlled__shoe__q0__clean", "controlled", "shoe", "q0",
        ("1", "2", "3", "4"), None, "clean", 0.0, 1, 0,
    )
    stress = AdjacentImageState(
        "stress__shoe__q0__deformed", "stress", "shoe", "q0",
        ("1", "2", "3", "4"), None, "deformed", 1.0, 1, 1,
    )
    controlled = AdjacentPairCondition(
        "C00000001", "controlled", "shoe", "q0", "shoe__q0__1__2",
        "1", "2", 1, 7.5, None, clean.state_id, "clean", 0.0,
    )
    stress_condition = AdjacentPairCondition(
        "S00000001", "stress", "shoe", "q0", "shoe__q0__1__2",
        "1", "2", 1, 7.5, None, stress.state_id, "deformed", 1.0,
    )
    variants = (
        FeatureVariant("user_37_grid_track", "dinov3_vitl16_pixal3d", 592, "bilinear", 37),
        FeatureVariant("user_37_grid_track", "dinov3_vitl16_pixal3d", 592, "naf_normal_guide", 592),
    )
    design = Adjacent16Design(
        (clean,), (stress,), (controlled,), (stress_condition,), variants,
        {"controlled": 2, "geometry_calibration": 68, "stress": 2},
    )
    registry = JobRegistry(tmp_path / "registry/jobs.sqlite")
    jobs = {
        "feature__controlled": ("feature", clean, "controlled", "controlled", variants),
        "feature__stress": ("stress", stress, "stress", "stress", variants),
        "feature__calibration": (
            "geometry_calibration_feature", clean, "geometry_calibration", "controlled", variants,
        ),
    }
    for job_id, (job_type, state, analysis_track, track, selected) in jobs.items():
        registry.register(JobDefinition(job_id, "feature", {
            "job_type": job_type, "state_id": state.state_id, "track": track,
            "analysis_track": analysis_track, "feature_track": "user_37_grid_track",
            "backbone": selected[0].backbone, "variant_keys": [variant.key for variant in selected],
            "shoe_id": state.shoe_id, "quartet_id": state.quartet_id,
        }))
        directory = "geometry_calibration" if job_type == "geometry_calibration_feature" else "user_37_grid_track"
        part = tmp_path / "metrics/parts" / directory / f"{job_id}.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        if job_type == "feature":
            rows = [_row(controlled, variant, job_id, analysis_track) for variant in selected]
        elif job_type == "stress":
            rows = [_row(stress_condition, variant, job_id, analysis_track) for variant in selected]
        else:
            rows = [
                _row(controlled, variant, job_id, analysis_track, changed_view=changed, perturbation_kind=kind, magnitude=magnitude)
                for changed in ("1", "2")
                for kind, magnitudes in _CALIBRATION_MAGNITUDES.items()
                for magnitude in magnitudes
                for variant in selected
            ]
        pd.DataFrame(rows).to_parquet(part, index=False)
        _complete(registry, job_id, part)
    return tmp_path, design


def test_adjacent16_ownership_counts_each_registered_real_metric_part(
    owned_adjacent_run: tuple[Path, Adjacent16Design],
) -> None:
    """Catch aggregation accepting a registry part whose rows belong to another job."""

    root, design = owned_adjacent_run

    assert validate_adjacent16_metric_ownership(root, design) == design.row_counts


def test_adjacent16_aggregation_preserves_quartet_baseline_and_generation_hash(
    owned_adjacent_run: tuple[Path, Adjacent16Design],
) -> None:
    """Catch aggregation dropping quartet/baseline columns or changing identical generation content."""

    root, design = owned_adjacent_run

    assert validate_adjacent16_metric_ownership(root, design) == {
        "controlled": 2,
        "geometry_calibration": 68,
        "stress": 2,
    }
    aggregate_all_metric_parts(root)
    first = resolve_aggregate_generation(root)
    controlled = pd.read_parquet(root / "metrics/controlled_fidelity.parquet")
    stress = pd.read_parquet(root / "metrics/real_deformation_stress.parquet")
    calibration = pd.read_parquet(root / "metrics/geometry_calibration.parquet")

    assert set(controlled[["design_id", "variant_key"]].itertuples(index=False, name=None)) == {
        ("C00000001", "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37"),
        ("C00000001", "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592"),
    }
    assert set(stress[["design_id", "variant_key"]].itertuples(index=False, name=None)) == {
        ("S00000001", "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37"),
        ("S00000001", "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592"),
    }
    assert {"quartet_id", "baseline_steps"} <= set(controlled)
    assert controlled[["quartet_id", "baseline_steps"]].drop_duplicates().to_dict("records") == [
        {"quartet_id": "q0", "baseline_steps": 1}
    ]
    assert len(calibration) == 68

    aggregate_all_metric_parts(root)
    assert resolve_aggregate_generation(root) == first


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "unregistered", "hash"))
def test_adjacent16_aggregation_rejects_non_owner_metric_parts(
    owned_adjacent_run: tuple[Path, Adjacent16Design],
    mutation: str,
) -> None:
    """Catch publication accepting duplicate, absent, unregistered, or hash-mutated parts."""

    root, _design = owned_adjacent_run
    part = root / "metrics/parts/user_37_grid_track/feature__controlled.parquet"
    frame = pd.read_parquet(part)
    if mutation == "duplicate":
        pd.concat((frame, frame.iloc[[0]]), ignore_index=True).to_parquet(part, index=False)
        _complete(JobRegistry(root / "registry/jobs.sqlite"), "feature__controlled", part)
    elif mutation == "missing":
        part.unlink()
    elif mutation == "unregistered":
        frame.iloc[[0]].to_parquet(part.with_name("unregistered.parquet"), index=False)
    else:
        frame.loc[0, "score"] = 0.9
        frame.to_parquet(part, index=False)

    with pytest.raises(ValueError):
        aggregate_all_metric_parts(root)


def test_adjacent16_ownership_rejects_calibration_variant_outside_job_subset(
    owned_adjacent_run: tuple[Path, Adjacent16Design],
) -> None:
    """Catch a calibration part emitting a variant outside its registered two-variant subset."""

    root, design = owned_adjacent_run
    part = root / "metrics/parts/geometry_calibration/feature__calibration.parquet"
    frame = pd.read_parquet(part)
    frame.loc[0, "variant_key"] = "unregistered_variant"
    frame.to_parquet(part, index=False)
    _complete(JobRegistry(root / "registry/jobs.sqlite"), "feature__calibration", part)

    with pytest.raises(ValueError, match="ownership"):
        validate_adjacent16_metric_ownership(root, design)
