from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image

from met3r.job_registry import JobRegistry
from met3r.upper_data import build_upper_manifest
from met3r.upper_design import build_upper_design
from met3r.upper_prepare import _register_state_jobs, prepare_run


ROOT = Path("/root/dev/ByteSize/Reconstruction")
CONFIG = Path(__file__).resolve().parents[1] / "configs/upper_dino_fidelity.json"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"jobs.sqlite", "jobs.sqlite-shm", "jobs.sqlite-wal"}
    }


def test_prepare_registers_expected_pilot_jobs_and_manifests(tmp_path: Path) -> None:
    prepared = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)

    assert prepared.root == (tmp_path / "pilot").resolve()
    assert prepared.registry.counts_by_phase() == {"feature": 64, "geometry": 32}
    expected_dirs = (
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
    assert all((prepared.root / name).is_dir() for name in expected_dirs)

    states = (prepared.root / "manifests/image_states.jsonl").read_text().splitlines()
    pairs = pd.read_parquet(prepared.root / "manifests/pair_conditions.parquet")
    variants = json.loads(
        (prepared.root / "manifests/feature_variants.json").read_text()
    )
    assert len(states) == 32
    assert len(pairs) == 44
    assert len(variants) == 30


def test_prepare_stages_all_sixteen_clean_pca_images_in_pilot(tmp_path: Path) -> None:
    prepared = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)

    paths = sorted((prepared.root / "pca_fit_images").glob("*/*.png"))

    assert len(paths) == 16
    assert all(Image.open(path).size == (504, 504) for path in paths)


def test_prepare_is_idempotent_and_preserves_registered_state(tmp_path: Path) -> None:
    first = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)
    claimed = first.registry.claim_next("worker", "geometry", {})
    assert claimed is not None
    before = _tree_hashes(first.root)

    second = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)

    after = _tree_hashes(second.root)
    assert second.registry.counts_by_phase() == {"feature": 64, "geometry": 32}
    assert second.registry.validated_state(claimed.job_id) == "running"
    assert before == after


def test_prepare_records_provenance_and_exact_gpu_policy(tmp_path: Path) -> None:
    prepared = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)

    payload = json.loads((prepared.root / "run_manifest.json").read_text())

    assert payload["config_sha256"] == prepared.config_sha256
    assert payload["git_commit"]
    assert len(payload["dirty_tree_fingerprint"]) == 64
    assert payload["python_version"]
    assert payload["package_versions"]
    assert payload["physical_gpu_policy"]["physical_gpus"] == [4, 5, 6, 7]
    assert payload["physical_gpu_policy"]["pilot_gpu"] == 4
    assert payload["physical_gpu_policy"]["feature_assignments"] == {
        "dino_vitb16": 4,
        "dinov2_vitb14": 5,
        "dinov3_vitb16": 6,
        "dinov3_vitl16_pixal3d": 7,
    }


def test_pilot_feature_jobs_are_one_per_state_and_pixal3d_track(tmp_path: Path) -> None:
    prepared = prepare_run(CONFIG, scope="pilot", output_root=tmp_path)
    with sqlite3.connect(prepared.registry.path) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM jobs WHERE phase = 'feature'"
        ).fetchall()

    payloads = [json.loads(row[0]) for row in rows]
    assert {item["backbone"] for item in payloads} == {"dinov3_vitl16_pixal3d"}
    assert {item["feature_track"] for item in payloads} == {
        "user_37_grid_track",
        "current_pixal3d_code_track",
    }
    assert len({item["state_id"] for item in payloads}) == 32


def test_full_registration_separates_controlled_calibration_and_stress_jobs(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    manifest = build_upper_manifest(ROOT)
    design = build_upper_design(manifest, config, scope="full")
    registry = JobRegistry(tmp_path / "jobs.sqlite")

    _register_state_jobs(registry, tmp_path, design, "f" * 64)

    assert registry.counts_by_phase() == {"feature": 1740, "geometry": 344}
    with sqlite3.connect(registry.path) as connection:
        rows = connection.execute(
            """
            SELECT json_extract(payload_json, '$.job_type'), COUNT(*)
            FROM jobs
            GROUP BY json_extract(payload_json, '$.job_type')
            """
        ).fetchall()
    assert dict(rows) == {
        "feature": 1700,
        "geometry": 340,
        "geometry_calibration": 20,
        "stress": 20,
        "stress_geometry": 4,
    }
