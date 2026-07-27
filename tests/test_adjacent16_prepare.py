from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_prepare import prepare_adjacent16_run
from tests.helpers_adjacent16 import write_dense_source, write_valid_stress


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"jobs.sqlite", "jobs.sqlite-shm", "jobs.sqlite-wal"}
    }


@pytest.fixture()
def prepared(tmp_path: Path):
    contract = load_adjacent16_contract(CONFIG)
    source = tmp_path / "reconstruction"
    stress = tmp_path / "stress"
    write_dense_source(source, contract)
    for shoe_id in contract.raw["shoes"]:
        for quartet in contract.quartets:
            for view_id in quartet.view_ids:
                write_valid_stress(stress, shoe_id, quartet.quartet_id, view_id)
    return prepare_adjacent16_run(
        CONFIG,
        output_root=tmp_path / "run",
        reconstruction_root=source,
        stress_root=stress,
    )


def test_prepare_builds_exact_quartet_artifacts_and_job_graph(prepared) -> None:
    assert prepared.registry.counts_by_phase() == {"feature": 3_472, "geometry": 32}
    states = [json.loads(line) for line in (prepared.root / "manifests/image_states.jsonl").read_text().splitlines()]
    pairs = [json.loads(line) for line in (prepared.root / "manifests/pair_conditions.jsonl").read_text().splitlines()]
    assert sum(state["track"] == "controlled" for state in states) == 848
    assert sum(state["track"] == "stress" for state in states) == 16
    assert len(pairs) == 2_688
    assert len(json.loads((prepared.root / "pca_fit_images" / "clean_source_paths.json").read_text())) == 64
    for state in states:
        assert len(list((prepared.root / "states" / state["state_id"] / "images").glob("*.png"))) == 4
    for camera in (prepared.root / "cameras").glob("*/*/cameras.npz"):
        artifact = __import__("numpy").load(camera)
        assert artifact["extrinsics"].shape == (4, 4, 4)
        assert artifact["intrinsics"].shape == (4, 3, 3)
        assert artifact["mask"].shape == (4, 504, 504)
        assert artifact["camera_groups"].tolist() == ["upper"] * 4

    with sqlite3.connect(prepared.registry.path) as connection:
        jobs = [json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM jobs")]
    assert all("geometry_contract" in payload for payload in jobs if payload["job_type"] == "feature")
    assert not any(payload["job_type"] == "geometry" and "controlled__" in payload["state_id"] and "clean" not in payload["state_id"] for payload in jobs)
    assert sum(payload["job_type"] == "stress_geometry" for payload in jobs) == 16
    calibration = [payload for payload in jobs if payload["job_type"] == "geometry_calibration"]
    assert len(calibration) == 16
    assert all(payload["calibration_artifacts"] is True for payload in calibration)
    calibration_features = [payload for payload in jobs if payload["job_type"] == "geometry_calibration_feature"]
    assert len(calibration_features) == 16
    assert all(len(payload["variant_keys"]) == 2 for payload in calibration_features)


def test_prepare_is_byte_idempotent_and_preserves_running_registry_row(prepared) -> None:
    claimed = prepared.registry.claim_next("worker", "geometry", {})
    assert claimed is not None
    before = _tree_hashes(prepared.root)

    second = prepare_adjacent16_run(
        CONFIG,
        output_root=prepared.root,
        reconstruction_root=prepared.root.parent / "reconstruction",
        stress_root=prepared.root.parent / "stress",
    )

    assert _tree_hashes(second.root) == before
    assert second.registry.validated_state(claimed.job_id) == "running"


def test_failed_stress_audit_blocks_before_registration(tmp_path: Path) -> None:
    contract = load_adjacent16_contract(CONFIG)
    source = tmp_path / "reconstruction"
    stress = tmp_path / "stress"
    write_dense_source(source, contract)
    for shoe_id in contract.raw["shoes"]:
        for quartet in contract.quartets:
            for view_id in quartet.view_ids:
                write_valid_stress(stress, shoe_id, quartet.quartet_id, view_id)
    (stress / next(iter(contract.raw["shoes"])) / "q0" / "view_1" / "deformation" / "task.json").write_text('{"status":"failed"}')

    with pytest.raises(ValueError):
        prepare_adjacent16_run(CONFIG, output_root=tmp_path / "run", reconstruction_root=source, stress_root=stress)
    assert not (tmp_path / "run" / "registry" / "jobs.sqlite").exists()
