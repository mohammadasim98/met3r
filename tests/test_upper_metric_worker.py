from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from met3r.camera_metadata import CAMERA_MODE_LABEL
from met3r.features import NAF_REVISION
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.upper_data import UPPER_VIEWS
from met3r.upper_metric_worker import (
    _assert_no_persistent_features,
    _write_parquet_atomic,
    run_feature_worker,
)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int, int, int]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(images.shape))
        output_size = images.shape[-1] // 4
        return F.adaptive_avg_pool2d(images, (output_size, output_size))


class FakeNAF(nn.Module):
    def forward(
        self,
        guide: torch.Tensor,
        native: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del guide
        return F.interpolate(
            native,
            output_size,
            mode="bilinear",
            align_corners=False,
        )


def _rgb(index: int) -> np.ndarray:
    yy, xx = np.indices((504, 504))
    return np.stack(
        (
            (xx + index * 17) % 256,
            (yy * 3 + index * 29) % 256,
            ((xx + yy) * 5 + index * 11) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _write_state(
    run_root: Path,
    state_id: str,
    *,
    shoe_id: str,
    track: str,
    corruption: str,
    changed_view: str | None,
) -> Path:
    state_dir = run_root / "states" / state_id
    (state_dir / "images").mkdir(parents=True)
    (state_dir / "masks").mkdir()
    for index, view_id in enumerate(UPPER_VIEWS):
        image = _rgb(index + (31 if corruption != "clean" else 0))
        Image.fromarray(image).save(state_dir / "images" / f"{view_id}.png")
        Image.fromarray(np.ones((504, 504), dtype=np.uint8) * 255).save(
            state_dir / "masks" / f"{view_id}.png"
        )
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "state": {
                    "state_id": state_id,
                    "track": track,
                    "shoe_id": shoe_id,
                    "changed_view": changed_view,
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


def _write_geometry(path: Path, *, depth_value: float = 1.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    intrinsics = np.broadcast_to(
        np.array(
            [[500.0, 0.0, 251.5], [0.0, 500.0, 251.5], [0.0, 0.0, 1.0]]
        ),
        (4, 3, 3),
    ).copy()
    extrinsics = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    arrays = path / "depth_geometry.npz"
    np.savez_compressed(
        arrays,
        depth=np.full((4, 504, 504), depth_value, dtype=np.float32),
        confidence=np.ones((4, 504, 504), dtype=np.float32),
        mask=np.ones((4, 504, 504), dtype=bool),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        meta__artifact_type=np.asarray("fixed_camera_depth"),
        meta__view_ids=np.asarray(UPPER_VIEWS),
        meta__camera_groups=np.asarray(("upper",) * 4),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
    )
    (path / "depth_geometry.json").write_text(
        json.dumps(
            {
                "camera_mode": CAMERA_MODE_LABEL,
                "camera_provenance": "fixed_precomputed_da3",
                "process_res": 504,
                "store_processed_images": False,
                "view_ids": list(UPPER_VIEWS),
                "camera_groups": ["upper"] * 4,
            }
        ),
        encoding="utf-8",
    )
    return arrays


def _quality(value: bool, relation: str) -> dict[str, object]:
    return {
        "status": "low_quality",
        "selected_attempt_id": "attempt-1",
        "semantic_gate": value,
        "preservation_gate": not value,
        "camera_preserved": value,
        "outsole_preserved": not value,
        "fit_relation": relation,
        "task_status": "complete",
    }


def _make_run(
    tmp_path: Path,
    *,
    scope: str = "pilot",
    job_type: str = "feature",
    two_tracks: bool = True,
) -> tuple[Path, JobRegistry, str]:
    run_root = tmp_path / scope
    for relative in (
        "config",
        "manifests",
        "registry",
        "states",
        "geometry",
        "metrics/parts",
        "pca_fit_images",
        "pca",
        "visualizations",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    tracks = {
        "track_8": {
            "fake_backbone": {"input_size": 8, "native_grid": 2},
            "common_evaluation_grids": [2, 4],
        }
    }
    variants = [
        {
            "feature_track": "track_8",
            "backbone": "fake_backbone",
            "input_size": 8,
            "feature_mode": "native",
            "resolution": 2,
        },
        {
            "feature_track": "track_8",
            "backbone": "fake_backbone",
            "input_size": 8,
            "feature_mode": "bilinear",
            "resolution": 4,
        },
    ]
    if two_tracks:
        tracks["track_12"] = {
            "fake_backbone": {"input_size": 12, "native_grid": 3},
            "common_evaluation_grids": [3, 6],
        }
        variants.extend(
            [
                {
                    "feature_track": "track_12",
                    "backbone": "fake_backbone",
                    "input_size": 12,
                    "feature_mode": "native",
                    "resolution": 3,
                },
                {
                    "feature_track": "track_12",
                    "backbone": "fake_backbone",
                    "input_size": 12,
                    "feature_mode": "bilinear",
                    "resolution": 6,
                },
            ]
        )
    config = {
        "camera_label": CAMERA_MODE_LABEL,
        "views": list(UPPER_VIEWS),
        "seed": 42,
        "stage_size": 504,
        "metric": {
            "relative_depth_tolerance": 0.03,
            "min_overlap_fraction": 0.08,
            "confidence_percentile": 10.0,
        },
        "models": {
            "fake_backbone": {"revision": "fake-revision"},
            "naf_revision": NAF_REVISION,
        },
        "feature_tracks": tracks,
        "geometry_calibration": {
            "rotation_degrees": [0.25, 0.5, 1.0, 2.0, 5.0],
            "relative_translation": [0.0025, 0.005, 0.01, 0.02],
            "depth_scale": [1.01, 1.02, 1.05, 1.1],
            "relative_depth_noise_std": [0.005, 0.01, 0.02, 0.05],
        },
        "corruptions": {"blur": [0.5, 1.0]},
    }
    config_path = run_root / "config" / "frozen_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (run_root / "run_manifest.json").write_text(
        json.dumps({"scope": scope, "config_sha256": config_sha256}),
        encoding="utf-8",
    )
    (run_root / "manifests" / "feature_variants.json").write_text(
        json.dumps(variants),
        encoding="utf-8",
    )

    for shoe_index in range(4):
        shoe_dir = run_root / "pca_fit_images" / f"shoe-{shoe_index}"
        shoe_dir.mkdir()
        for view_index, view_id in enumerate(UPPER_VIEWS):
            Image.fromarray(_rgb(shoe_index * 4 + view_index)).save(
                shoe_dir / f"{view_id}.png"
            )

    shoe_id = "shoe-a"
    clean_id = "controlled__shoe-a__clean"
    changed_id = (
        "stress__shoe-a__deformed"
        if job_type == "stress"
        else clean_id
        if job_type == "geometry_calibration"
        else "controlled__shoe-a__left__blur__0.500000__+1"
    )
    clean_state = _write_state(
        run_root,
        clean_id,
        shoe_id=shoe_id,
        track="controlled",
        corruption="clean",
        changed_view=None,
    )
    changed_state = clean_state
    if job_type != "geometry_calibration":
        changed_state = _write_state(
            run_root,
            changed_id,
            shoe_id=shoe_id,
            track="stress" if job_type == "stress" else "controlled",
            corruption="deformed" if job_type == "stress" else "blur",
            changed_view=None if job_type == "stress" else "left",
        )
    state_rows = [
        json.loads((clean_state / "state.json").read_text())["state"],
    ]
    if changed_state != clean_state:
        state_rows.append(json.loads((changed_state / "state.json").read_text())["state"])
    (run_root / "manifests" / "image_states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in state_rows),
        encoding="utf-8",
    )

    _write_geometry(run_root / "geometry" / clean_id)
    if changed_state != clean_state:
        _write_geometry(run_root / "geometry" / changed_id, depth_value=1.01)
    pair_rows = []
    pairs = (
        [(UPPER_VIEWS[i], UPPER_VIEWS[j]) for i in range(4) for j in range(i + 1, 4)]
        if job_type == "geometry_calibration"
        else [("left", "back")]
    )
    for index, (first, second) in enumerate(pairs):
        pair_rows.append(
            {
                "design_id": f"{'S' if job_type == 'stress' else 'D'}{index + 1:08d}",
                "track": "stress" if job_type == "stress" else "controlled",
                "shoe_id": shoe_id,
                "pair_id": f"{first}__{second}",
                "first_view": first,
                "second_view": second,
                "changed_view": None if job_type in {"stress", "geometry_calibration"} else "left",
                "image_state_id": changed_id,
                "corruption": (
                    "deformed"
                    if job_type == "stress"
                    else "clean"
                    if job_type == "geometry_calibration"
                    else "blur"
                ),
                "severity": 1.0 if job_type == "stress" else 0.0 if job_type == "geometry_calibration" else 0.5,
            }
        )
    pd.DataFrame(pair_rows).to_parquet(
        run_root / "manifests" / "pair_conditions.parquet",
        index=False,
    )
    shoes = [
        {
            "shoe_id": shoe_id,
            "views": [
                {
                    "view_id": view_id,
                    "quality": _quality(index % 2 == 0, ("improved", "equivalent", "worsened", "improved")[index]),
                }
                for index, view_id in enumerate(UPPER_VIEWS)
            ],
        }
    ]
    (run_root / "manifests" / "upper_manifest.json").write_text(
        json.dumps({"shoes": shoes}),
        encoding="utf-8",
    )

    if job_type == "geometry_calibration":
        calibration_root = run_root / "geometry" / clean_id / "calibration"
        calibration_root.mkdir()
        records = []
        values = {
            "rotation": [0.25, 0.5, 1.0, 2.0, 5.0],
            "relative_translation": [0.0025, 0.005, 0.01, 0.02],
            "depth_scale": [1.01, 1.02, 1.05, 1.1],
            "relative_depth_noise": [0.005, 0.01, 0.02, 0.05],
        }
        template = _write_geometry(calibration_root / "_template")
        for view_index, view_id in enumerate(UPPER_VIEWS):
            for kind, magnitudes in values.items():
                for magnitude in magnitudes:
                    destination = (
                        calibration_root
                        / view_id
                        / f"{kind}_{magnitude}"
                        / "geometry.npz"
                    )
                    destination.parent.mkdir(parents=True)
                    os.link(template, destination)
                    records.append(
                        {
                            "condition_id": f"{kind}_{magnitude}",
                            "kind": kind,
                            "magnitude": magnitude,
                            "path": str(destination.resolve()),
                            "view_index": view_index,
                            "view_id": view_id,
                            "sha256": sha256_file(destination),
                        }
                    )
        (calibration_root / "calibration_manifest.json").write_text(
            json.dumps(
                {
                    "artifact_type": "upper_geometry_calibration",
                    "base_geometry": str(
                        (run_root / "geometry" / clean_id / "depth_geometry.npz").resolve()
                    ),
                    "config_sha256": config_sha256,
                    "seed": 42,
                    "views": list(UPPER_VIEWS),
                    "artifacts": records,
                }
            ),
            encoding="utf-8",
        )

    registry = JobRegistry(run_root / "registry" / "jobs.sqlite")
    active_tracks = ["track_8", "track_12"] if two_tracks else ["track_8"]
    for feature_track in active_tracks:
        job_id = (
            f"feature__{job_type}__{changed_id}__fake_backbone__{feature_track}"
        )
        registry.register(
            JobDefinition(
                job_id,
                "feature",
                {
                    "job_type": job_type,
                    "state_id": changed_id,
                    "track": "stress" if job_type == "stress" else "controlled",
                    "analysis_track": (
                        "geometry_calibration"
                        if job_type == "geometry_calibration"
                        else job_type
                        if job_type == "stress"
                        else "controlled"
                    ),
                    "shoe_id": shoe_id,
                    "state_dir": str(changed_state.resolve()),
                    "camera_path": str(
                        (run_root / "cameras" / shoe_id / "cameras.npz").resolve()
                    ),
                    "config_sha256": config_sha256,
                    "backbone": "fake_backbone",
                    "feature_track": feature_track,
                },
            )
        )
    return run_root, registry, changed_id


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeBackbone, list[FakeNAF]]:
    backbone = FakeBackbone()
    naf_loads: list[FakeNAF] = []
    monkeypatch.setattr(
        "met3r.upper_metric_worker.load_feature_backbone",
        lambda name: backbone,
    )

    def load_naf(**kwargs: object) -> FakeNAF:
        del kwargs
        model = FakeNAF()
        naf_loads.append(model)
        return model

    monkeypatch.setattr(
        "met3r.upper_metric_worker.load_naf_upsampler",
        load_naf,
    )
    return backbone, naf_loads


def _read_parts(run_root: Path) -> pd.DataFrame:
    paths = sorted((run_root / "metrics" / "parts").rglob("*.parquet"))
    assert paths
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "registry" not in path.parts
        and path.name not in {"jobs.sqlite-shm", "jobs.sqlite-wal"}
    }


def _reset_feature_job_pending(registry: JobRegistry) -> None:
    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET state = 'pending', worker_id = NULL, attempts = 0,
                started_at = NULL, finished_at = NULL,
                output_manifest_json = NULL, error = NULL
            WHERE phase = 'feature'
            """
        )


def test_worker_writes_factorial_rows_without_feature_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _make_run(tmp_path)
    backbone, naf_loads = _patch_models(monkeypatch)

    summary = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert summary.completed == 2
    assert summary.failed == 0
    frame = _read_parts(run_root)
    assert len(frame) == 4
    assert {
        "s00",
        "s10",
        "s01",
        "s11",
        "feature_effect",
        "geometry_effect",
        "interaction",
        "s11_first_to_second_score",
        "s11_second_to_first_score",
        "s11_first_to_second_overlap",
        "s11_second_to_first_overlap",
        "cycle_error_px_median",
        "relative_depth_error_median",
        "config_sha256",
        "camera_provenance",
        "camera_label",
        "backbone_revision",
    } <= set(frame)
    assert set(frame["camera_label"]) == {CAMERA_MODE_LABEL}
    assert len(naf_loads) == 1
    assert backbone.calls[:8] == [
        (4, 3, 8, 8),
        (4, 3, 8, 8),
        (4, 3, 8, 8),
        (4, 3, 8, 8),
        (4, 3, 12, 12),
        (4, 3, 12, 12),
        (4, 3, 12, 12),
        (4, 3, 12, 12),
    ]
    basis_manifest = json.loads(
        (run_root / "pca" / "fake_backbone" / "basis.json").read_text()
    )
    assert basis_manifest["fit_image_count"] == 16
    assert basis_manifest["fit_sample_count"] == 32
    assert basis_manifest["input_sizes"] == [8, 12]
    assert not list(run_root.rglob("features.npz"))
    assert not list(run_root.rglob("*.pt"))
    assert len(list((run_root / "visualizations").rglob("*.png"))) == 8


def test_legacy_worker_rejects_manifest_without_explicit_native_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch adjacent implicit-native compatibility leaking into legacy manifests."""

    run_root, _, _ = _make_run(tmp_path, two_tracks=False)
    manifest_path = run_root / "manifests" / "feature_variants.json"
    variants = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants[0]["feature_mode"] = "bilinear"
    manifest_path.write_text(json.dumps(variants), encoding="utf-8")
    _patch_models(monkeypatch)

    with pytest.raises(
        ValueError,
        match="each feature track/input must have exactly one native variant",
    ):
        run_feature_worker(run_root, "fake_backbone", "cpu")


def test_worker_reuses_only_active_shoe_track_clean_native_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, _ = _make_run(tmp_path, two_tracks=False)
    second_id = "controlled__shoe-a__left__blur__1.000000__+1"
    second_state = _write_state(
        run_root,
        second_id,
        shoe_id="shoe-a",
        track="controlled",
        corruption="blur",
        changed_view="left",
    )
    state_payload = json.loads((second_state / "state.json").read_text())["state"]
    with (run_root / "manifests" / "image_states.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(json.dumps(state_payload) + "\n")
    _write_geometry(run_root / "geometry" / second_id, depth_value=1.02)
    pair_path = run_root / "manifests" / "pair_conditions.parquet"
    pairs = pd.read_parquet(pair_path)
    pairs = pd.concat(
        [
            pairs,
            pd.DataFrame(
                [
                    {
                        **pairs.iloc[0].to_dict(),
                        "design_id": "D00000002",
                        "image_state_id": second_id,
                        "severity": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    pairs.to_parquet(pair_path, index=False)
    config_sha256 = json.loads(
        (run_root / "run_manifest.json").read_text()
    )["config_sha256"]
    registry.register(
        JobDefinition(
            f"feature__feature__{second_id}__fake_backbone__track_8",
            "feature",
            {
                "job_type": "feature",
                "state_id": second_id,
                "track": "controlled",
                "analysis_track": "controlled",
                "shoe_id": "shoe-a",
                "state_dir": str(second_state.resolve()),
                "camera_path": str(
                    (run_root / "cameras" / "shoe-a" / "cameras.npz").resolve()
                ),
                "config_sha256": config_sha256,
                "backbone": "fake_backbone",
                "feature_track": "track_8",
            },
        )
    )
    backbone, _ = _patch_models(monkeypatch)

    summary = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert summary.completed == 2
    assert len(backbone.calls) == 7
    assert backbone.calls[:4] == [(4, 3, 8, 8)] * 4


def test_stress_rows_keep_both_view_quality_covariates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _make_run(
        tmp_path,
        scope="full",
        job_type="stress",
        two_tracks=False,
    )
    _patch_models(monkeypatch)

    summary = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert summary.completed == 1
    frame = _read_parts(run_root)
    assert {
        "first_semantic_gate",
        "second_semantic_gate",
        "first_preservation_gate",
        "second_preservation_gate",
        "first_camera_preserved",
        "second_camera_preserved",
        "first_outsole_preserved",
        "second_outsole_preserved",
        "first_fit_relation",
        "second_fit_relation",
    } <= set(frame)
    row = frame.iloc[0]
    assert bool(row["first_semantic_gate"]) is True
    assert bool(row["second_semantic_gate"]) is False
    assert row["first_fit_relation"] == "improved"
    assert row["second_fit_relation"] == "equivalent"


def test_geometry_calibration_scores_seventeen_perturbations_per_changed_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _make_run(
        tmp_path,
        scope="full",
        job_type="geometry_calibration",
        two_tracks=False,
    )
    _patch_models(monkeypatch)

    summary = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert summary.completed == 1
    path = next(
        (run_root / "metrics" / "parts" / "geometry_calibration").glob("*.parquet")
    )
    frame = pd.read_parquet(path)
    assert len(frame) == 2 * 4 * 17 * 3
    assert set(frame["changed_view"]) == set(UPPER_VIEWS)
    assert (
        frame.groupby(["variant_key", "changed_view"]).size().unique().tolist()
        == [51]
    )
    assert {
        "perturbation_kind",
        "magnitude",
        "base_score",
        "perturbed_score",
        "geometry_effect",
        "overlap",
        "abstained",
    } <= set(frame)


def test_atomic_parquet_readback_failure_preserves_existing_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "part.parquet"
    _write_parquet_atomic(pd.DataFrame([{"value": 1}]), destination)
    before = destination.read_bytes()
    monkeypatch.setattr(
        "met3r.upper_metric_worker.pq.read_table",
        lambda path: (_ for _ in ()).throw(OSError("readback failed")),
    )

    with pytest.raises(OSError, match="readback failed"):
        _write_parquet_atomic(pd.DataFrame([{"value": 2}]), destination)

    assert destination.read_bytes() == before
    assert not destination.with_suffix(".parquet.tmp").exists()


@pytest.mark.parametrize("mode", ["basename", "metadata"])
def test_persistent_feature_guard_rejects_raw_dino_artifacts(
    tmp_path: Path,
    mode: str,
) -> None:
    path = (
        tmp_path / "features.npz"
        if mode == "basename"
        else tmp_path / "renamed.npz"
    )
    np.savez_compressed(
        path,
        meta__artifact_type=np.asarray(
            "other" if mode == "basename" else "dino_features"
        ),
    )

    with pytest.raises(ValueError, match="persistent DINO feature"):
        _assert_no_persistent_features(tmp_path)


def test_persistent_feature_guard_tolerates_concurrent_directory_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disappearing = tmp_path / "concurrently-published-output"
    disappearing.mkdir()
    real_scandir = os.scandir
    removed = False

    def remove_before_scandir(path: os.PathLike[str] | str) -> os.ScandirIterator:
        nonlocal removed
        if Path(path) == disappearing:
            disappearing.rmdir()
            removed = True
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", remove_before_scandir)

    _assert_no_persistent_features(tmp_path)

    assert removed is True


def test_worker_revalidates_completed_part_and_recomputes_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, _ = _make_run(tmp_path, two_tracks=False)
    _patch_models(monkeypatch)
    first = run_feature_worker(run_root, "fake_backbone", "cpu")
    assert first.completed == 1
    part = next((run_root / "metrics" / "parts" / "track_8").glob("*.parquet"))
    part.write_bytes(b"corrupt")

    second = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert second.completed == 1
    assert registry.validated_state(
        "feature__feature__controlled__shoe-a__left__blur__0.500000__+1__"
        "fake_backbone__track_8"
    ) == "completed"
    assert pd.read_parquet(part).shape[0] == 2
    with sqlite3.connect(registry.path) as connection:
        attempts = connection.execute(
            "SELECT attempts FROM jobs WHERE state = 'completed'"
        ).fetchone()[0]
    assert attempts == 1


def test_registry_completion_failure_restores_prior_complete_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, _ = _make_run(tmp_path, two_tracks=False)
    _patch_models(monkeypatch)
    assert run_feature_worker(run_root, "fake_backbone", "cpu").completed == 1
    before = _tree_hashes(run_root)
    _reset_feature_job_pending(registry)
    original_complete = JobRegistry.complete

    def fail_feature_completion(
        self: JobRegistry,
        job_id: str,
        output_manifest: dict[str, object],
    ) -> None:
        if job_id.startswith("feature__"):
            raise RuntimeError("registry completion failed")
        original_complete(self, job_id, output_manifest)

    monkeypatch.setattr(JobRegistry, "complete", fail_feature_completion)

    with pytest.raises(RuntimeError, match="registry completion failed"):
        run_feature_worker(run_root, "fake_backbone", "cpu")

    assert _tree_hashes(run_root) == before
    assert registry.validated_state(
        "feature__feature__controlled__shoe-a__left__blur__0.500000__+1__"
        "fake_backbone__track_8"
    ) == "failed"
    assert not list(run_root.rglob("*.tmp"))
    assert not list(run_root.rglob(".*.backup.*"))
    assert not list(run_root.rglob(".*.empty.*"))


def test_first_install_registry_failure_exposes_no_partial_final_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, registry, _ = _make_run(tmp_path, two_tracks=False)
    _patch_models(monkeypatch)
    monkeypatch.setattr(
        JobRegistry,
        "complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("registry completion failed")
        ),
    )

    with pytest.raises(RuntimeError, match="registry completion failed"):
        run_feature_worker(run_root, "fake_backbone", "cpu")

    assert not list((run_root / "metrics" / "parts").rglob("*.parquet"))
    assert not list((run_root / "visualizations").glob("feature__*"))
    assert not list(
        (run_root / "pca" / "fake_backbone" / "track_8").glob("feature__*")
    )
    assert not list(run_root.rglob("*.tmp"))
    assert not list(run_root.rglob(".*.backup.*"))
    assert not list(run_root.rglob(".*.empty.*"))
    with sqlite3.connect(registry.path) as connection:
        state = connection.execute(
            "SELECT state FROM jobs WHERE phase = 'feature'"
        ).fetchone()[0]
    assert state == "failed"


def test_valid_completed_resume_keeps_outputs_and_cleans_exact_debris(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _make_run(tmp_path, two_tracks=False)
    _patch_models(monkeypatch)
    assert run_feature_worker(run_root, "fake_backbone", "cpu").completed == 1
    part = next((run_root / "metrics" / "parts" / "track_8").glob("*.parquet"))
    backup = part.with_name(f".{part.name}.backup.dead")
    backup.write_bytes(b"old")
    Path(f"{part}.tmp").write_bytes(b"partial")
    visual = next((run_root / "visualizations").glob("feature__*"))
    staging = visual.with_name(f".{visual.name}.staging.dead")
    staging.mkdir()
    (staging / "partial.png").write_bytes(b"partial")
    before = _tree_hashes(run_root)
    before.pop(str(backup.relative_to(run_root)))
    before.pop(str(Path(f"{part}.tmp").relative_to(run_root)))
    before.pop(str((staging / "partial.png").relative_to(run_root)))

    summary = run_feature_worker(run_root, "fake_backbone", "cpu")

    assert summary.completed == 0
    assert _tree_hashes(run_root) == before
    assert not backup.exists()
    assert not Path(f"{part}.tmp").exists()
    assert not staging.exists()


def test_exact_worker_run_does_not_claim_wrong_or_unlisted_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch timed sentinel fallback from named jobs to arbitrary pending work."""

    run_root, registry, _ = _make_run(tmp_path, two_tracks=True)
    _patch_models(monkeypatch)
    config_sha256 = json.loads((run_root / "run_manifest.json").read_text())["config_sha256"]
    registry.register(
        JobDefinition(
            "feature__wrong-backbone", "feature",
            {
                "job_type": "feature", "state_id": "controlled__shoe-a__left__blur__0.500000__+1",
                "track": "controlled", "analysis_track": "controlled", "shoe_id": "shoe-a",
                "state_dir": str((run_root / "states" / "controlled__shoe-a__left__blur__0.500000__+1").resolve()),
                "camera_path": str((run_root / "cameras" / "shoe-a" / "cameras.npz").resolve()),
                "config_sha256": config_sha256, "backbone": "other_backbone", "feature_track": "track_8",
            },
        )
    )
    target = "feature__feature__controlled__shoe-a__left__blur__0.500000__+1__fake_backbone__track_8"
    monkeypatch.setattr(
        JobRegistry, "claim_next", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback claim_next"))
    )

    summary = run_feature_worker(run_root, "fake_backbone", "cpu", job_ids=[target, target, "missing", "feature__wrong-backbone"])

    assert summary.completed == 1
    with sqlite3.connect(registry.path) as connection:
        rows = dict(connection.execute("SELECT job_id, state || ':' || attempts FROM jobs"))
    assert rows[target] == "completed:1"
    assert rows["feature__wrong-backbone"] == "pending:0"
    other = "feature__feature__controlled__shoe-a__left__blur__0.500000__+1__fake_backbone__track_12"
    assert rows[other] == "pending:0"
