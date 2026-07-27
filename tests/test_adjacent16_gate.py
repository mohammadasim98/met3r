from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest
import pandas as pd

from met3r.adjacent16_gate import (
    validate_adjacent16_final_gate,
    validate_timed_runtime_projection,
    write_timed_sentinel_manifest,
)
from met3r.job_registry import JobDefinition, JobRegistry
from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import Adjacent16Manifest, AdjacentShoeRecord, AdjacentViewRecord
from met3r.adjacent16_design import build_adjacent16_design
from met3r.adjacent16_prepare import register_adjacent16_jobs


def _write_frozen_manifest_fixture(root: Path) -> tuple[tuple[Path, ...], object, object]:
    contract = load_adjacent16_contract("configs/adjacent16_core.json")
    positions = {view: index for index, view in enumerate(contract.camera_order)}
    shoes = []
    for shoe in contract.raw["shoes"]:
        views = []
        for quartet in contract.quartets:
            for view in quartet.view_ids:
                views.append(AdjacentViewRecord(shoe, "fixture", quartet.quartet_id, view, positions[view], root / "rgb", root / "mask", None, None, None, None, None, 1.0, positions[view] * 7.5, "0" * 64, "0" * 64, None, None, None))
        shoes.append(AdjacentShoeRecord(shoe, "fixture", tuple(views)))
    design = build_adjacent16_design(Adjacent16Manifest(root, root, tuple(shoes)), contract)
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    states = (*design.controlled_states, *design.stress_states)
    state_path = manifests / "image_states.jsonl"
    pair_path = manifests / "pair_conditions.jsonl"
    variant_path = manifests / "feature_variants.json"
    state_path.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in states))
    pair_path.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in (*design.controlled_conditions, *design.stress_conditions)))
    variant_path.write_text(json.dumps([asdict(item) for item in design.feature_variants], sort_keys=True))
    return (state_path, pair_path, variant_path), contract, design


def test_independent_frozen_manifest_fixture_is_exact_and_byte_stable(tmp_path: Path) -> None:
    """Catch test fixtures that silently shrink the frozen manifest design."""

    first, _contract, _design = _write_frozen_manifest_fixture(tmp_path / "one")
    second, _contract2, _design2 = _write_frozen_manifest_fixture(tmp_path / "two")
    states = [json.loads(line) for line in first[0].read_text().splitlines()]
    pairs = [json.loads(line) for line in first[1].read_text().splitlines()]

    assert len(states) == 864
    assert sum(row["track"] == "controlled" for row in states) == 848
    assert sum(row["track"] == "stress" for row in states) == 16
    assert len({row["state_id"] for row in states}) == 864
    assert len(pairs) == 2688
    assert sum(row["track"] == "controlled" for row in pairs) == 2592
    assert sum(row["track"] == "stress" for row in pairs) == 96
    assert {row["severity"] for row in pairs if row["corruption"] == "identity_swap"} == {1.0}
    assert all(row["changed_view"] in {row["first_view"], row["second_view"]} for row in pairs if row["changed_view"] is not None)
    assert json.loads(first[2].read_text()) == json.loads(second[2].read_text())
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_independent_manifest_design_registers_the_exact_production_job_graph(tmp_path: Path) -> None:
    """Catch a fixture graph that diverges from production registration payloads."""

    _paths, contract, design = _write_frozen_manifest_fixture(tmp_path / "run")
    root = tmp_path / "run"
    marker = root / "reports" / "completed.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("fixture\n")
    registry = JobRegistry(root / "registry/jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, "a" * 64)
    with sqlite3.connect(registry.path) as connection:
        connection.execute("UPDATE jobs SET state='completed', attempts=1, started_at='2026-01-01T00:00:00+00:00', finished_at='2026-01-01T00:00:01+00:00', output_manifest_json=?", (json.dumps({"files": [{"path": str(marker.resolve()), "sha256": __import__("hashlib").sha256(marker.read_bytes()).hexdigest()}]}),))
        rows = connection.execute("SELECT job_id, phase, state, attempts, payload_json FROM jobs").fetchall()
    payloads = [json.loads(row[4]) for row in rows]
    assert len(rows) == 3504
    assert {row[1] for row in rows} == {"geometry", "feature"}
    assert sum(row[1] == "geometry" for row in rows) == 32
    assert sum(row[1] == "feature" for row in rows) == 3472
    assert all(row[2] == "completed" and row[3] == 1 for row in rows)
    assert len({row[0] for row in rows}) == 3504
    assert all(payload["config_sha256"] == "a" * 64 for payload in payloads)
    assert sum(payload["job_type"] == "geometry_calibration_feature" for payload in payloads) == 16
    assert sum(payload["job_type"] == "stress" for payload in payloads) == 64
    assert all(payload["geometry_contract"] in {"fixed_base_only", "base_plus_adaptive"} for payload in payloads)


def _metric_row(condition: dict[str, object], variant: object, config_hash: str, **extra: object) -> dict[str, object]:
    return {**condition, "variant_key": variant.key, "feature_track": variant.feature_track, "backbone": variant.backbone, "input_size": variant.input_size, "feature_mode": variant.feature_mode, "resolution": variant.resolution, "config_sha256": config_hash, "camera_label": "fixed/precomputed DA3 cameras (not calibrated GT)", "camera_provenance": "fixed_precomputed_da3", "abstained": False, "score": .8, "s00": .8, "s10": .8, "s01": .8, "s11": .8, "s11_first_to_second_overlap": .8, "s11_second_to_first_overlap": .8, **extra}


def _write_frozen_metric_tables(root: Path, contract: object, design: object, config_hash: str) -> dict[str, Path]:
    controlled = [_metric_row(asdict(condition), variant, config_hash) for condition in design.controlled_conditions for variant in design.feature_variants]
    stress = [_metric_row(asdict(condition), variant, config_hash) for condition in design.stress_conditions for variant in design.feature_variants]
    calibration_variants = [variant for variant in design.feature_variants if variant.key in contract.geometry_calibration_variant_keys]
    calibration = []
    kind_map = {"rotation": "rotation_degrees", "relative_translation": "relative_translation", "depth_scale": "depth_scale", "relative_depth_noise": "relative_depth_noise_std"}
    clean = [state for state in design.controlled_states if state.corruption == "clean"]
    for state in clean:
        pairs = [asdict(row) for row in design.controlled_conditions if row.image_state_id == state.state_id]
        for changed in state.view_ids:
            for condition in pairs:
                if changed not in {condition["first_view"], condition["second_view"]}:
                    continue
                for kind, config_name in kind_map.items():
                    for magnitude in contract.raw["geometry_calibration"][config_name]:
                        for variant in calibration_variants:
                            calibration.append(_metric_row(condition, variant, config_hash, changed_view=changed, perturbation_kind=kind, magnitude=float(magnitude)))
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    paths = {"controlled": metrics / "controlled_fidelity.parquet", "stress": metrics / "real_deformation_stress.parquet", "geometry_calibration": metrics / "geometry_calibration.parquet"}
    for name, rows in (("controlled", controlled), ("stress", stress), ("geometry_calibration", calibration)):
        pd.DataFrame(rows).to_parquet(paths[name], index=False)
    return paths


def test_independent_metric_fixture_has_all_frozen_rows_and_unique_keys(tmp_path: Path) -> None:
    """Catch metric fixtures that replace frozen hard cells with duplicate easy rows."""

    _paths, contract, design = _write_frozen_manifest_fixture(tmp_path / "run")
    tables = _write_frozen_metric_tables(tmp_path / "run", contract, design, "a" * 64)
    controlled, stress, calibration = (pd.read_parquet(tables[name]) for name in ("controlled", "stress", "geometry_calibration"))

    assert (len(controlled), len(stress), len(calibration)) == (31104, 1152, 6528)
    assert not controlled.duplicated(["shoe_id", "quartet_id", "first_view", "second_view", "changed_view", "corruption", "severity", "variant_key"]).any()
    assert not stress.duplicated(["shoe_id", "quartet_id", "first_view", "second_view", "variant_key"]).any()
    assert not calibration.duplicated(["shoe_id", "quartet_id", "first_view", "second_view", "changed_view", "perturbation_kind", "magnitude", "variant_key"]).any()
    assert set(controlled.loc[controlled.corruption == "identity_swap", "severity"]) == {1.0}
    assert set(controlled.variant_key) == {variant.key for variant in design.feature_variants}
    assert set(calibration.perturbation_kind) == {"rotation", "relative_translation", "depth_scale", "relative_depth_noise"}


def _frozen_config(root: Path) -> None:
    (root / "config").mkdir(parents=True)
    source = Path("configs/adjacent16_core.json")
    content = source.read_bytes()
    (root / "config" / "frozen_config.json").write_bytes(content)
    import hashlib

    (root / "run_manifest.json").write_text(
        json.dumps({
            "experiment_id": "20260727_adjacent16_core_v1",
            "config_sha256": hashlib.sha256(content).hexdigest(),
        }),
        encoding="utf-8",
    )


def _register_sentinel_jobs(root: Path) -> tuple[str, ...]:
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    jobs: list[str] = []
    backbones = (
        "dino_vitb16",
        "dinov2_vitb14",
        "dinov3_vitb16",
        "dinov3_vitl16_pixal3d",
    )
    for backbone in backbones:
        for job_type, corruption, severity, track in (
            ("feature", "clean", 0.0, "controlled"),
            ("feature", "translation", 0.2, "controlled"),
            ("stress", "stress", 1.0, "stress"),
        ):
            job_id = f"feature__{job_type}__{corruption}__{backbone}"
            registry.register(JobDefinition(job_id, "feature", {
                "job_type": job_type,
                "backbone": backbone,
                "shoe_id": "air_huarache_run_ultra_111449",
                "quartet_id": "q0",
                "track": track,
                "state_id": f"{track}__{corruption}__{severity}",
                "corruption": corruption,
                "severity": severity,
                "changed_view": "1" if corruption == "translation" else None,
            }))
            jobs.append(job_id)
    calibration = "feature__geometry_calibration_feature__dinov3_vitl16_pixal3d"
    registry.register(JobDefinition(calibration, "feature", {
        "job_type": "geometry_calibration_feature",
        "backbone": "dinov3_vitl16_pixal3d",
        "shoe_id": "air_huarache_run_ultra_111449",
        "quartet_id": "q0",
        "track": "controlled",
        "state_id": "controlled__clean__0.0",
        "corruption": "clean",
        "severity": 0.0,
    }))
    jobs.append(calibration)
    return tuple(jobs)


def test_timed_sentinel_manifest_selects_exactly_the_registered_thirteen_jobs(
    tmp_path: Path,
) -> None:
    """Catch a sampler that creates score-only jobs or omits a frozen class."""

    _frozen_config(tmp_path)
    expected = _register_sentinel_jobs(tmp_path)

    directory = write_timed_sentinel_manifest(tmp_path)

    selected = tuple(
        job_id
        for path in sorted(directory.glob("*.json"))
        for job_id in json.loads(path.read_text(encoding="utf-8"))["job_ids"]
    )
    assert set(selected) == set(expected)
    assert len(selected) == 13

    (directory / "stale.json").write_text("{}", encoding="utf-8")
    repeated = write_timed_sentinel_manifest(tmp_path)
    assert repeated == directory
    assert not (directory / "stale.json").exists()


def test_runtime_projection_rejects_pending_or_untimed_sentinel_jobs(
    tmp_path: Path,
) -> None:
    """Catch projection code that estimates from pending or zero-duration work."""

    _frozen_config(tmp_path)
    _register_sentinel_jobs(tmp_path)

    report = validate_timed_runtime_projection(tmp_path)

    assert report.passed is False
    assert report.sentinel_job_ids == ()
    assert "sentinel_job_state" in report.failed_checks


def test_runtime_projection_uses_the_frozen_four_gpu_max_load_formula(
    tmp_path: Path,
) -> None:
    """Catch summing backbone loads instead of taking the assigned-GPU maximum."""

    _frozen_config(tmp_path)
    sentinel = _register_sentinel_jobs(tmp_path)
    registry = JobRegistry(tmp_path / "registry" / "jobs.sqlite")
    for index in range(32):
        registry.register(JobDefinition(f"geometry__{index:02d}", "geometry", {
            "job_type": "geometry_calibration", "shoe_id": "shoe", "quartet_id": "q0",
        }))
    start = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(registry.path) as connection:
        for job_id in sentinel:
            kind = "calibration" if "geometry_calibration" in job_id else (
                "stress" if "stress" in job_id else "changed" if "translation" in job_id else "clean"
            )
            seconds = {"clean": 1, "changed": 2, "stress": 3, "calibration": 4}[kind]
            connection.execute(
                "UPDATE jobs SET state = 'completed', started_at = ?, finished_at = ? WHERE job_id = ?",
                (start, f"2026-01-01T00:00:{seconds:02d}+00:00", job_id),
            )
        connection.execute(
            "UPDATE jobs SET state = 'completed', started_at = ?, finished_at = ? WHERE phase = 'geometry'",
            (start, "2026-01-01T01:00:00+00:00"),
        )

    report = validate_timed_runtime_projection(tmp_path)

    assert report.passed is True
    assert report.projected_geometry_hours == 1.0
    assert report.projected_feature_hours == pytest.approx(1792 / 3600)
    assert report.projected_total_hours == pytest.approx(1.0 + 1.25 * (1792 / 3600) + 2.0)


def test_final_gate_reports_missing_inputs_as_failed_evidence(tmp_path: Path) -> None:
    """Catch a final gate that turns absent analysis artifacts into zero evidence."""

    _frozen_config(tmp_path)

    report = validate_adjacent16_final_gate(tmp_path)

    assert report.passed is False
    assert report.actual_rows == {"controlled": 0, "geometry_calibration": 0, "stress": 0, "total": 0}
    assert "registry_counts" in report.failed_checks
