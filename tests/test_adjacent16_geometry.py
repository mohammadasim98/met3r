from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from met3r.job_registry import JobDefinition, JobRegistry
from met3r.upper_geometry import run_geometry_worker


_Q3_VIEWS = ("40", "41", "42", "0")


class _FakeDepthModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def inference(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        extrinsics = np.asarray(kwargs["extrinsics"])
        intrinsics = np.asarray(kwargs["intrinsics"])
        return SimpleNamespace(
            depth=np.ones((4, 504, 504), dtype=np.float32),
            conf=np.full((4, 504, 504), 0.75, dtype=np.float32),
            processed_images=np.zeros((4, 504, 504, 3), dtype=np.uint8),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )


def _write_adjacent_run(tmp_path: Path) -> tuple[Path, JobRegistry, Path, str]:
    run_root = tmp_path / "adjacent"
    (run_root / "config").mkdir(parents=True)
    (run_root / "cameras" / "shoe-a" / "q3").mkdir(parents=True)
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
        json.dumps(
            {
                "experiment_id": "20260727_adjacent16_core_v1",
                "config_sha256": config_sha256,
            }
        ),
        encoding="utf-8",
    )

    state_dir = run_root / "states" / "controlled__shoe-a__q3__clean"
    (state_dir / "images").mkdir(parents=True)
    (state_dir / "masks").mkdir()
    views = []
    for index, view_id in enumerate(_Q3_VIEWS):
        image_path = state_dir / "images" / f"{view_id}.png"
        mask_path = state_dir / "masks" / f"{view_id}.png"
        Image.new("RGB", (504, 504), (index, index, index)).save(image_path)
        Image.fromarray(np.ones((504, 504), dtype=np.uint8) * 255).save(mask_path)
        views.append(
            {
                "view_id": view_id,
                "image": f"images/{view_id}.png",
                "mask": f"masks/{view_id}.png",
            }
        )
    (state_dir / "images.json").write_text(
        json.dumps({"views": views}), encoding="utf-8"
    )
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "state": {
                    "state_id": state_dir.name,
                    "track": "controlled",
                    "shoe_id": "shoe-a",
                    "quartet_id": "q3",
                    "view_ids": list(_Q3_VIEWS),
                    "changed_view": None,
                    "corruption": "clean",
                    "severity": 0.0,
                    "direction": 1,
                    "seed": 42,
                },
                "stage_size": 504,
            }
        ),
        encoding="utf-8",
    )

    extrinsics = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    extrinsics[:, 0, 3] = [40.0, 41.0, 42.0, 0.0]
    intrinsics = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    np.savez_compressed(
        run_root / "cameras" / "shoe-a" / "q3" / "cameras.npz",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        view_ids=np.asarray(_Q3_VIEWS),
        camera_groups=np.asarray(("upper",) * 4),
    )
    registry = JobRegistry(run_root / "registry" / "jobs.sqlite")
    registry.register(
        JobDefinition(
            f"geometry__{state_dir.name}",
            "geometry",
            {
                "job_type": "geometry_calibration",
                "calibration_artifacts": True,
                "state_id": state_dir.name,
                "track": "controlled",
                "shoe_id": "shoe-a",
                "quartet_id": "q3",
                "state_dir": str(state_dir),
                "camera_path": str(
                    run_root / "cameras" / "shoe-a" / "q3" / "cameras.npz"
                ),
                "config_sha256": config_sha256,
                "output_dir": str(run_root / "geometry" / state_dir.name),
            },
        )
    )
    return run_root, registry, state_dir, config_sha256


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: _FakeDepthModel) -> None:
    monkeypatch.setattr("met3r.upper_geometry.load_da3_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        "met3r.upper_geometry.repository_state",
        lambda path: {
            "path": str(path),
            "revision": "da3-source",
            "dirty": False,
            "dirty_fingerprint": None,
        },
    )


def test_geometry_uses_exact_q3_state_order_through_depth_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, state_dir, _ = _write_adjacent_run(tmp_path)
    model = _FakeDepthModel()
    _patch_model(monkeypatch, model)

    summary = run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert (summary.completed, summary.failed) == (1, 0)
    assert [Path(path).stem for path in model.calls[0]["image"]] == list(_Q3_VIEWS)
    output_dir = run_root / "geometry" / state_dir.name
    with np.load(output_dir / "depth_geometry.npz") as artifact:
        assert artifact["depth"].shape == (4, 504, 504)
        assert artifact["confidence"].shape == (4, 504, 504)
        assert artifact["mask"].shape == (4, 504, 504)
        assert artifact["meta__view_ids"].tolist() == list(_Q3_VIEWS)
    metadata = json.loads((output_dir / "depth_geometry.json").read_text())
    assert metadata["view_ids"] == list(_Q3_VIEWS)
    calibration = json.loads(
        (output_dir / "calibration" / "calibration_manifest.json").read_text()
    )
    assert calibration["views"] == list(_Q3_VIEWS)
    assert [record["view_id"] for record in calibration["artifacts"]] == [
        view_id for view_id in _Q3_VIEWS for _ in range(17)
    ]
    assert len(calibration["artifacts"]) == 68
    assert registry.validated_state(f"geometry__{state_dir.name}") == "completed"


def test_geometry_rejects_camera_order_mismatch_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, state_dir, _ = _write_adjacent_run(tmp_path)
    camera_path = run_root / "cameras" / "shoe-a" / "q3" / "cameras.npz"
    with np.load(camera_path) as original:
        np.savez_compressed(
            camera_path,
            **{
                name: np.asarray(original[name])
                for name in original.files
                if name != "view_ids"
            },
            view_ids=np.asarray(("0", "40", "41", "42")),
        )
    model = _FakeDepthModel()
    _patch_model(monkeypatch, model)

    with pytest.raises(ValueError, match="camera view order"):
        run_geometry_worker(run_root, "shoe-a", "cpu", "/fake/da3")

    assert model.calls == []
    assert registry.validated_state(f"geometry__{state_dir.name}") == "failed"
    with sqlite3.connect(registry.path) as connection:
        attempts = connection.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?",
            (f"geometry__{state_dir.name}",),
        ).fetchone()[0]
    assert attempts == 1
