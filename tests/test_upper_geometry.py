from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.upper_data import UPPER_VIEWS
from met3r.upper_geometry import run_geometry_worker


class _FakeDepthModel:
    def __init__(self, depth_value: float = 1.0) -> None:
        self.calls: list[dict[str, object]] = []
        self.depth_value = depth_value

    def inference(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        extrinsics = np.asarray(kwargs["extrinsics"])
        intrinsics = np.asarray(kwargs["intrinsics"])
        count = len(UPPER_VIEWS)
        return SimpleNamespace(
            depth=np.full(
                (count, 504, 504),
                self.depth_value,
                dtype=np.float32,
            ),
            conf=np.full((count, 504, 504), 0.75, dtype=np.float32),
            processed_images=np.zeros((count, 504, 504, 3), dtype=np.uint8),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )


def _write_state(run_root: Path, state_id: str, *, corruption: str) -> Path:
    state_dir = run_root / "states" / state_id
    (state_dir / "images").mkdir(parents=True)
    (state_dir / "masks").mkdir()
    for view_id in UPPER_VIEWS:
        Image.new("RGB", (504, 504), "white").save(
            state_dir / "images" / f"{view_id}.png"
        )
        Image.fromarray(np.ones((504, 504), dtype=np.uint8) * 255).save(
            state_dir / "masks" / f"{view_id}.png"
        )
    (state_dir / "images.json").write_text(
        json.dumps(
            {
                "views": [
                    {
                        "view_id": view_id,
                        "image": f"images/{view_id}.png",
                        "mask": f"masks/{view_id}.png",
                    }
                    for view_id in UPPER_VIEWS
                ]
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "state": {
                    "state_id": state_id,
                    "track": "controlled",
                    "shoe_id": "shoe-a",
                    "changed_view": None if corruption == "clean" else "left",
                    "corruption": corruption,
                    "severity": 0.0 if corruption == "clean" else 0.5,
                    "direction": 1,
                    "seed": 42,
                },
                "stage_size": 504,
            }
        ),
        encoding="utf-8",
    )
    return state_dir


def _fake_run(tmp_path: Path, *, scope: str = "pilot") -> tuple[Path, JobRegistry]:
    run_root = tmp_path / scope
    (run_root / "config").mkdir(parents=True)
    (run_root / "cameras" / "shoe-a").mkdir(parents=True)
    config = {
        "stage_size": 504,
        "models": {
            "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
            "da3_revision": "8615eefb62f2db4f8d6ebaa59160086981672829",
        },
        "geometry_calibration": {
            "rotation_degrees": [0.25, 0.5, 1.0, 2.0, 5.0],
            "relative_translation": [0.0025, 0.005, 0.01, 0.02],
            "depth_scale": [1.01, 1.02, 1.05, 1.1],
            "relative_depth_noise_std": [0.005, 0.01, 0.02, 0.05],
        },
        "seed": 42,
    }
    config_path = run_root / "config" / "frozen_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (run_root / "run_manifest.json").write_text(
        json.dumps({"scope": scope, "config_sha256": config_sha256}),
        encoding="utf-8",
    )
    extrinsics = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    extrinsics[:, 0, 3] = [0.0, 1.0, 2.0, 3.0]
    intrinsics = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    np.savez_compressed(
        run_root / "cameras" / "shoe-a" / "cameras.npz",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        mask=np.ones((4, 504, 504), dtype=bool),
        view_ids=np.asarray(UPPER_VIEWS),
        camera_groups=np.asarray(("upper",) * 4),
    )
    registry = JobRegistry(run_root / "registry" / "jobs.sqlite")
    return run_root, registry


def _register_geometry_job(
    registry: JobRegistry,
    run_root: Path,
    state_dir: Path,
    config_sha256: str,
) -> None:
    state_id = state_dir.name
    registry.register(
        JobDefinition(
            f"geometry__{state_id}",
            "geometry",
            {
                "job_type": "geometry",
                "state_id": state_id,
                "track": "controlled",
                "analysis_track": "controlled",
                "shoe_id": "shoe-a",
                "state_dir": str(state_dir),
                "camera_path": str(run_root / "cameras" / "shoe-a" / "cameras.npz"),
                "config_sha256": config_sha256,
                "output_dir": str(run_root / "geometry" / state_id),
            },
        )
    )


def _reset_job_pending(registry: JobRegistry, job_id: str) -> None:
    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET state = 'pending', worker_id = NULL, attempts = 0,
                started_at = NULL, finished_at = NULL,
                output_manifest_json = NULL, error = NULL
            WHERE job_id = ?
            """,
            (job_id,),
        )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_geometry_worker_loads_model_once_for_multiple_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    for state_id, corruption in (("clean", "clean"), ("changed", "blur")):
        _register_geometry_job(
            registry,
            run_root,
            _write_state(run_root, state_id, corruption=corruption),
            config_sha256,
        )
    loads: list[tuple[object, ...]] = []
    model = _FakeDepthModel()
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: loads.append((args, kwargs)) or model,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )

    summary = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert summary.completed == 2
    assert summary.failed == 0
    assert len(loads) == 1
    assert loads[0][0] == ("depth-anything/DA3NESTED-GIANT-LARGE",)
    assert loads[0][1] == {
        "revision": "8615eefb62f2db4f8d6ebaa59160086981672829",
        "repository_root": "/fake/da3",
        "device": "cpu",
    }
    assert len(model.calls) == 2
    assert all(call["process_res"] == 504 for call in model.calls)
    assert all(call["align_to_input_ext_scale"] is True for call in model.calls)
    for state_id in ("clean", "changed"):
        with np.load(run_root / "geometry" / state_id / "depth_geometry.npz") as data:
            assert "processed_images" not in data.files
            assert data["meta__view_ids"].tolist() == list(UPPER_VIEWS)
            assert data["meta__camera_groups"].tolist() == ["upper"] * 4
            assert data["meta__camera_provenance"].item() == "fixed_precomputed_da3"
        metadata = json.loads(
            (run_root / "geometry" / state_id / "depth_geometry.json").read_text()
        )
        assert (
            metadata["camera_mode"]
            == "fixed/precomputed DA3 cameras (not calibrated GT)"
        )
        assert metadata["store_processed_images"] is False
        assert not (run_root / "geometry" / state_id / "calibration").exists()


def test_full_clean_geometry_writes_exact_one_factor_calibration_and_hashes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path, scope="full")
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "clean", corruption="clean")
    _register_geometry_job(
        registry,
        run_root,
        state_dir,
        config_sha256,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: _FakeDepthModel(),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )

    summary = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert summary.completed == 1
    output_dir = run_root / "geometry" / "clean"
    manifest_path = output_dir / "calibration" / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["artifacts"]
    assert len(records) == 4 * (5 + 4 + 4 + 4)
    assert {record["view_id"] for record in records} == set(UPPER_VIEWS)
    expected_magnitudes = {
        "rotation": {0.25, 0.5, 1.0, 2.0, 5.0},
        "relative_translation": {0.0025, 0.005, 0.01, 0.02},
        "depth_scale": {1.01, 1.02, 1.05, 1.1},
        "relative_depth_noise": {0.005, 0.01, 0.02, 0.05},
    }
    for kind, magnitudes in expected_magnitudes.items():
        assert {
            record["magnitude"] for record in records if record["kind"] == kind
        } == magnitudes
    base_path = output_dir / "depth_geometry.npz"
    with np.load(base_path) as source:
        base = {key: np.asarray(source[key]) for key in source.files}
    for record in records:
        path = Path(record["path"])
        assert path.is_file()
        assert record["sha256"] == sha256_file(path)
        view_index = UPPER_VIEWS.index(record["view_id"])
        with np.load(path) as artifact:
            if record["kind"] in {"rotation", "relative_translation"}:
                np.testing.assert_array_equal(artifact["depth"], base["depth"])
                unchanged = [index for index in range(4) if index != view_index]
                np.testing.assert_array_equal(
                    artifact["extrinsics"][unchanged],
                    base["extrinsics"][unchanged],
                )
            else:
                np.testing.assert_array_equal(
                    artifact["extrinsics"], base["extrinsics"]
                )
                unchanged = [index for index in range(4) if index != view_index]
                np.testing.assert_array_equal(
                    artifact["depth"][unchanged],
                    base["depth"][unchanged],
                )
    with sqlite3.connect(registry.path) as connection:
        output_manifest = json.loads(
            connection.execute(
                "SELECT output_manifest_json FROM jobs WHERE job_id = ?",
                ("geometry__clean",),
            ).fetchone()[0]
        )
    registered = {item["path"]: item["sha256"] for item in output_manifest["files"]}
    assert registered[str(manifest_path.resolve())] == sha256_file(manifest_path)
    assert all(registered[record["path"]] == record["sha256"] for record in records)


def test_worker_rejects_state_shoe_contract_without_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "wrong-shoe", corruption="blur")
    metadata_path = state_dir / "state.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["state"]["shoe_id"] = "shoe-b"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    model = _FakeDepthModel()
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )

    with pytest.raises(ValueError, match="shoe"):
        run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert model.calls == []
    assert registry.validated_state("geometry__wrong-shoe") == "failed"
    with sqlite3.connect(registry.path) as connection:
        attempts = connection.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?",
            ("geometry__wrong-shoe",),
        ).fetchone()[0]
    assert attempts == 1


def test_cuda_oom_is_retried_only_to_registry_attempt_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "oom", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)

    class OOMModel:
        def __init__(self) -> None:
            self.calls = 0

        def inference(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            raise RuntimeError("CUDA out of memory while allocating tensor")

    model = OOMModel()
    loads: list[int] = []
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: loads.append(1) or model,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )

    summary = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert (summary.completed, summary.abstained, summary.failed) == (0, 0, 1)
    assert loads == [1]
    assert model.calls == 3
    assert registry.validated_state("geometry__oom") == "failed"
    with sqlite3.connect(registry.path) as connection:
        attempts = connection.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?",
            ("geometry__oom",),
        ).fetchone()[0]
    assert attempts == 3


def test_worker_revalidates_completed_outputs_and_recomputes_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "resume", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    models = [_FakeDepthModel(1.0), _FakeDepthModel(2.0)]
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )

    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    arrays_path = run_root / "geometry" / "resume" / "depth_geometry.npz"
    arrays_path.write_bytes(b"corrupted")
    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert second.completed == 1
    with np.load(arrays_path) as artifact:
        np.testing.assert_allclose(artifact["depth"], 2.0)
    assert registry.validated_state("geometry__resume") == "completed"
    with sqlite3.connect(registry.path) as connection:
        attempts = connection.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?",
            ("geometry__resume",),
        ).fetchone()[0]
    assert attempts == 1


def test_worker_base_failure_never_exposes_partial_final_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "base-failure", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: _FakeDepthModel(),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    original_write_text = Path.write_text

    def fail_depth_metadata(
        path: Path,
        data: str,
        *args,
        **kwargs,
    ) -> int:
        if path.name == "depth_geometry.json":
            raise ValueError("injected depth metadata failure")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_depth_metadata)

    with pytest.raises(ValueError, match="injected depth metadata failure"):
        run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    output_dir = run_root / "geometry" / "base-failure"
    assert not output_dir.exists()
    assert not list((run_root / "geometry").glob(".base-failure.*"))
    assert registry.validated_state("geometry__base-failure") == "failed"


def test_calibration_failure_preserves_prior_complete_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path, scope="full")
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "clean", corruption="clean")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    models = [_FakeDepthModel(1.0), _FakeDepthModel(2.0)]
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "clean"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__clean")
    from met3r import upper_geometry

    original_writer = upper_geometry.write_geometry_calibration_artifacts
    calls = 0

    def fail_second_view(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected calibration failure")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(
        "met3r.upper_geometry.write_geometry_calibration_artifacts",
        fail_second_view,
    )

    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert (second.completed, second.failed) == (0, 1)
    assert _tree_hashes(output_dir) == before
    with np.load(output_dir / "depth_geometry.npz") as artifact:
        np.testing.assert_allclose(artifact["depth"], 1.0)
    assert not list((run_root / "geometry").glob(".clean.*"))


def test_install_failure_rolls_back_prior_complete_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "install", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    models = [_FakeDepthModel(1.0), _FakeDepthModel(2.0)]
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "install"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__install")
    original_rename = Path.rename

    def fail_staging_install(path: Path, target: Path) -> Path:
        if (
            path.name.startswith(".install.staging.")
            and Path(target) == output_dir
        ):
            raise OSError("injected install failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_install)

    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert (second.completed, second.failed) == (0, 1)
    assert _tree_hashes(output_dir) == before
    with np.load(output_dir / "depth_geometry.npz") as artifact:
        np.testing.assert_allclose(artifact["depth"], 1.0)
    assert not list((run_root / "geometry").glob(".install.*"))


def test_worker_recovers_interrupted_backup_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "interrupted", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    first_model = _FakeDepthModel(1.0)
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: first_model,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "interrupted"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__interrupted")
    backup = output_dir.with_name(".interrupted.backup.killed")
    output_dir.rename(backup)
    stale_staging = output_dir.with_name(".interrupted.staging.killed")
    stale_staging.mkdir()
    (stale_staging / "partial").write_bytes(b"partial")

    class OOMModel:
        def inference(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: OOMModel(),
    )

    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert (second.completed, second.failed) == (0, 1)
    assert _tree_hashes(output_dir) == before
    assert not list((run_root / "geometry").glob(".interrupted.*"))


def test_published_hash_failure_restores_prior_complete_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "hash-failure", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    models = [_FakeDepthModel(1.0), _FakeDepthModel(2.0)]
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "hash-failure"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__hash-failure")
    from met3r import upper_geometry

    original_hash = upper_geometry.sha256_file

    def fail_published_hash(path: Path) -> str:
        if Path(path) == output_dir / "depth_geometry.npz":
            raise ValueError("injected published hash failure")
        return original_hash(path)

    monkeypatch.setattr("met3r.upper_geometry.sha256_file", fail_published_hash)

    with pytest.raises(ValueError, match="injected published hash failure"):
        run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert _tree_hashes(output_dir) == before
    assert registry.validated_state("geometry__hash-failure") == "failed"
    assert not list((run_root / "geometry").glob(".hash-failure.*"))


def test_registry_completion_failure_restores_prior_complete_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "complete-failure", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    models = [_FakeDepthModel(1.0), _FakeDepthModel(2.0)]
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "complete-failure"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__complete-failure")
    original_complete = JobRegistry.complete

    def fail_completion(
        self: JobRegistry,
        job_id: str,
        output_manifest,
    ) -> None:
        if job_id == "geometry__complete-failure":
            raise ValueError("injected registry completion failure")
        original_complete(self, job_id, output_manifest)

    monkeypatch.setattr(JobRegistry, "complete", fail_completion)

    with pytest.raises(ValueError, match="injected registry completion failure"):
        run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert _tree_hashes(output_dir) == before
    assert registry.validated_state("geometry__complete-failure") == "failed"
    assert not list((run_root / "geometry").glob(".complete-failure.*"))


def test_precompletion_crash_restores_backup_before_failed_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "precomplete-crash", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: _FakeDepthModel(1.0),
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "precomplete-crash"
    before = _tree_hashes(output_dir)
    _reset_job_pending(registry, "geometry__precomplete-crash")
    backup = output_dir.with_name(".precomplete-crash.backup.killed")
    shutil.copytree(output_dir, backup)
    arrays_path = output_dir / "depth_geometry.npz"
    with np.load(arrays_path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["depth"] = np.full((4, 504, 504), 2.0, dtype=np.float32)
    np.savez_compressed(arrays_path, **arrays)
    stale_staging = output_dir.with_name(".precomplete-crash.staging.killed")
    stale_staging.mkdir()
    (stale_staging / "partial").write_bytes(b"partial")

    class OOMModel:
        def inference(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: OOMModel(),
    )

    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert (second.completed, second.failed) == (0, 1)
    assert _tree_hashes(output_dir) == before
    assert not list((run_root / "geometry").glob(".precomplete-crash.*"))


def test_postcompletion_crash_keeps_valid_destination_and_cleans_debris(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root, registry = _fake_run(tmp_path)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    state_dir = _write_state(run_root, "postcomplete-crash", corruption="blur")
    _register_geometry_job(registry, run_root, state_dir, config_sha256)
    model = _FakeDepthModel(1.0)
    monkeypatch.setattr(
        "met3r.upper_geometry.load_da3_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )
    first = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")
    output_dir = run_root / "geometry" / "postcomplete-crash"
    before = _tree_hashes(output_dir)
    backup = output_dir.with_name(".postcomplete-crash.backup.killed")
    shutil.copytree(output_dir, backup)
    stale_staging = output_dir.with_name(".postcomplete-crash.staging.killed")
    stale_staging.mkdir()
    (stale_staging / "partial").write_bytes(b"partial")

    second = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert first.completed == 1
    assert second.completed == 0
    assert len(model.calls) == 1
    assert _tree_hashes(output_dir) == before
    assert registry.validated_state("geometry__postcomplete-crash") == "completed"
    assert not list((run_root / "geometry").glob(".postcomplete-crash.*"))
