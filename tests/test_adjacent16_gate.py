from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from met3r.adjacent16_gate import (
    validate_adjacent16_final_gate,
    validate_timed_runtime_projection,
    write_timed_sentinel_manifest,
)
from met3r.job_registry import JobDefinition, JobRegistry
from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import Adjacent16Manifest, AdjacentShoeRecord, AdjacentViewRecord
from met3r.adjacent16_design import build_adjacent16_design


def _write_frozen_manifest_fixture(root: Path) -> tuple[Path, ...]:
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
    return state_path, pair_path, variant_path


def test_independent_frozen_manifest_fixture_is_exact_and_byte_stable(tmp_path: Path) -> None:
    """Catch test fixtures that silently shrink the frozen manifest design."""

    first = _write_frozen_manifest_fixture(tmp_path / "one")
    second = _write_frozen_manifest_fixture(tmp_path / "two")
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
