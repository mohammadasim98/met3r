from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from met3r.camera_metadata import CAMERA_MODE_LABEL
from met3r.features import NAF_REVISION
from met3r.features import BACKBONE_SPECS
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.upper_metric_worker import (
    _StateRecord,
    _clean_state_for_shoe,
    _load_state_index,
    _validate_adjacent_pair_contract,
    _calibration_records,
    _RunContract,
    _assert_no_persistent_features,
    _sensitivity_row,
    run_feature_worker,
)
from met3r.known_camera import KnownCameraConfig


_L_BACKBONE = "dinov3_vitl16_pixal3d"
_L37_BILINEAR = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37"
)
_L592_NAF = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592"
)
_CALIBRATION_VALUES = {
    "rotation": [0.25, 0.5, 1.0, 2.0, 5.0],
    "relative_translation": [0.0025, 0.005, 0.01, 0.02],
    "depth_scale": [1.01, 1.02, 1.05, 1.1],
    "relative_depth_noise": [0.005, 0.01, 0.02, 0.05],
}


def test_state_index_and_clean_lookup_keep_quartets_separate(tmp_path: Path) -> None:
    """Catch a clean cache lookup collapsing distinct ordered four-view quartets."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    records = [
        {
            "state_id": "clean-q0", "track": "controlled", "shoe_id": "shoe",
            "quartet_id": "q0", "view_ids": ["1", "2", "3", "4"],
            "changed_view": None, "corruption": "clean", "severity": 0.0,
        },
        {
            "state_id": "clean-q1", "track": "controlled", "shoe_id": "shoe",
            "quartet_id": "q1", "view_ids": ["14", "15", "16", "17"],
            "changed_view": None, "corruption": "clean", "severity": 0.0,
        },
    ]
    (manifests / "image_states.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    states = _load_state_index(tmp_path)

    assert states["clean-q1"].view_ids == ("14", "15", "16", "17")
    assert _clean_state_for_shoe(states, "shoe", "q1").state_id == "clean-q1"


def test_adjacent_pair_contract_requires_quartet_and_measured_baseline() -> None:
    """Catch null/default baseline metadata being emitted for adjacent rows."""

    state = _StateRecord(
        "q3-clean", "controlled", "shoe", "q3", ("40", "41", "42", "0"),
        None, "clean", 0.0, Path("/states/q3-clean"),
    )
    pair = {
        "quartet_id": "q3", "first_view": "42", "second_view": "0",
        "baseline_steps": 1, "angular_separation_deg": 7.5,
    }

    _validate_adjacent_pair_contract(pair, state)

    for field, invalid in (("quartet_id", "q0"), ("baseline_steps", None), ("angular_separation_deg", float("nan"))):
        invalid_pair = {**pair, field: invalid}
        try:
            _validate_adjacent_pair_contract(invalid_pair, state)
        except ValueError:
            continue
        raise AssertionError(f"invalid adjacent {field} was accepted")


def test_adjacent_contract_rejects_same_quartet_with_reordered_views() -> None:
    """Catch pairing clean geometry order with a reordered changed-state tensor."""

    clean = _StateRecord(
        "clean", "controlled", "shoe", "q3", ("40", "41", "42", "0"),
        None, "clean", 0.0, Path("/states/clean"),
    )
    changed = _StateRecord(
        "changed", "controlled", "shoe", "q3", ("0", "40", "41", "42"),
        "0", "jpeg", 0.5, Path("/states/changed"),
    )

    try:
        _validate_adjacent_pair_contract({}, changed, clean_state=clean)
    except ValueError as error:
        assert "view order" in str(error)
    else:
        raise AssertionError("reordered state was accepted")


def _write_q3_geometry(path: Path, views: tuple[str, str, str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        depth=np.ones((4, 504, 504), dtype=np.float32),
        mask=np.ones((4, 504, 504), dtype=bool),
        intrinsics=np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
        extrinsics=np.broadcast_to(np.eye(4), (4, 4, 4)).copy(),
        meta__view_ids=np.asarray(views),
        meta__camera_groups=np.asarray(("upper",) * 4),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
    )


def _write_adjacent_calibration_fixture(
    root: Path,
    *,
    config: dict[str, object] | None = None,
    config_sha256: str = "config",
) -> tuple[_RunContract, _StateRecord]:
    """Build the exact 4 x 17 q3 calibration-manifest shape used by the worker."""
    views = ("40", "41", "42", "0")
    if config is None:
        config = {
            "geometry_calibration": {
                "rotation_degrees": _CALIBRATION_VALUES["rotation"],
                "relative_translation": _CALIBRATION_VALUES[
                    "relative_translation"
                ],
                "depth_scale": _CALIBRATION_VALUES["depth_scale"],
                "relative_depth_noise_std": _CALIBRATION_VALUES[
                    "relative_depth_noise"
                ],
            }
        }
    directory = root / "geometry" / "calibration" / "clean"
    directory.mkdir(parents=True)
    base = directory / "depth_geometry.npz"
    _write_q3_geometry(base, views)
    (directory / "depth_geometry.json").write_text(
        json.dumps(
            {
                "camera_mode": CAMERA_MODE_LABEL,
                "camera_provenance": "fixed_precomputed_da3",
                "process_res": 504,
                "view_ids": list(views),
                "camera_groups": ["upper"] * 4,
            }
        ),
        encoding="utf-8",
    )
    template = directory / "_template.npz"
    _write_q3_geometry(template, views)
    records = []
    for index, view in enumerate(views):
        for kind, magnitudes in _CALIBRATION_VALUES.items():
            for magnitude in magnitudes:
                path = (
                    directory
                    / view
                    / f"{kind}_{magnitude}"
                    / "geometry.npz"
                )
                path.parent.mkdir(parents=True)
                os.link(template, path)
                records.append(
                    {
                        "condition_id": f"{kind}_{magnitude}",
                        "kind": kind,
                        "magnitude": magnitude,
                        "path": str(path.resolve()),
                        "view_index": index,
                        "view_id": view,
                        "sha256": sha256_file(path),
                    }
                )
    (directory / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "upper_geometry_calibration",
                "base_geometry": str(base.resolve()),
                "config_sha256": config_sha256,
                "seed": 42,
                "views": list(views),
                "artifacts": records,
            }
        ),
        encoding="utf-8",
    )
    contract = _RunContract(
        root,
        config,
        config_sha256,
        "full",
        KnownCameraConfig(.03, .08),
        10.,
        "test",
        "posthoc_audit_only",
    )
    return contract, _StateRecord("clean", "controlled", "shoe", "q3", views, None, "clean", 0., root / "states" / "clean")


def test_adjacent_calibration_fixture_has_all_frozen_q3_artifacts(tmp_path: Path) -> None:
    """Catch an incomplete/reordered frozen perturbation matrix before worker use."""
    contract, clean = _write_adjacent_calibration_fixture(tmp_path)
    records = _calibration_records(contract, clean)
    assert len(records) == 68
    assert [record["view_id"] for record in records] == [view for view in clean.view_ids for _ in range(17)]
    assert {view: sum(record["view_id"] == view for record in records) for view in clean.view_ids} == {view: 17 for view in clean.view_ids}


def _fake_calibration_result() -> SimpleNamespace:
    """Match GeometrySensitivityResult fields consumed by _sensitivity_row."""
    def direction(score: float) -> SimpleNamespace:
        return SimpleNamespace(score=torch.tensor([score]), overlap_fraction=torch.tensor([.75]), cycle_error_px=torch.tensor([[1., 2.]]), relative_depth_error=torch.tensor([[.1, .2]]), valid_mask=torch.tensor([[True, True]]))
    return SimpleNamespace(baseline=SimpleNamespace(score=torch.tensor([.8]), first_to_second=direction(.8), second_to_first=direction(.7), abstained=torch.tensor([False])), perturbed=SimpleNamespace(score=torch.tensor([.6]), first_to_second=direction(.6), second_to_first=direction(.5), abstained=torch.tensor([False])), geometry_effect=torch.tensor([-.2]))


def test_fake_calibration_result_drives_real_sensitivity_row_schema() -> None:
    """Catch fake calibration scoring drift from the worker's real row adapter."""
    row = _sensitivity_row(_fake_calibration_result())
    assert {"base_score", "perturbed_score", "geometry_effect", "overlap", "abstained", "cycle_error_px_median", "relative_depth_error_median"} <= set(row)
    assert all(np.isfinite(float(row[name])) for name in ("base_score", "perturbed_score", "geometry_effect", "overlap", "cycle_error_px_median", "relative_depth_error_median"))


class _TinyBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))


class _TinyNAF(nn.Module):
    def forward(self, guide: torch.Tensor, native: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        del guide
        return torch.nn.functional.interpolate(native, output_size, mode="bilinear", align_corners=False)


class _CalibrationBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.nn.functional.adaptive_avg_pool2d(images, (37, 37))


class _CalibrationNAF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        guide: torch.Tensor,
        native: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del guide, output_size
        self.calls += 1
        return native


def _make_adjacent_calibration_run(
    tmp_path: Path,
    variant_keys: list[str],
) -> tuple[Path, JobRegistry, str]:
    root = tmp_path / "calibration-run"
    for name in ("config", "manifests", "registry", "states"):
        (root / name).mkdir(parents=True, exist_ok=True)

    frozen_path = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"
    config = json.loads(frozen_path.read_text(encoding="utf-8"))
    raw = json.dumps(config).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (root / "config" / "frozen_config.json").write_bytes(raw)
    (root / "run_manifest.json").write_text(
        json.dumps({"scope": "full", "config_sha256": digest}),
        encoding="utf-8",
    )
    (root / "manifests" / "feature_variants.json").write_text(
        json.dumps(config["feature_variants"]),
        encoding="utf-8",
    )

    views = ("40", "41", "42", "0")
    state_dir = root / "states" / "clean"
    (state_dir / "images").mkdir(parents=True)
    for index, view_id in enumerate(views):
        Image.new("RGB", (504, 504), (index * 30, 0, 0)).save(
            state_dir / "images" / f"{view_id}.png"
        )
    state = {
        "state_id": "clean",
        "track": "controlled",
        "shoe_id": "shoe",
        "quartet_id": "q3",
        "view_ids": list(views),
        "changed_view": None,
        "corruption": "clean",
        "severity": 0.0,
    }
    (state_dir / "state.json").write_text(
        json.dumps({"state": state, "stage_size": 504}),
        encoding="utf-8",
    )
    (root / "manifests" / "image_states.jsonl").write_text(
        json.dumps(state) + "\n",
        encoding="utf-8",
    )

    pair_specs = (
        ("40", "41", 1, 7.0),
        ("40", "42", 2, 15.0),
        ("40", "0", 3, 22.5),
        ("41", "42", 1, 8.0),
        ("41", "0", 2, 15.5),
        ("42", "0", 1, 7.5),
    )
    pairs = [
        {
            "design_id": f"G{index}",
            "track": "controlled",
            "shoe_id": "shoe",
            "quartet_id": "q3",
            "pair_id": f"shoe__q3__{first}__{second}",
            "first_view": first,
            "second_view": second,
            "baseline_steps": steps,
            "angular_separation_deg": angle,
            "changed_view": None,
            "image_state_id": "clean",
            "corruption": "clean",
            "severity": 0.0,
        }
        for index, (first, second, steps, angle) in enumerate(pair_specs, 1)
    ]
    (root / "manifests" / "pair_conditions.jsonl").write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs),
        encoding="utf-8",
    )
    quality = {
        "semantic_gate": True,
        "preservation_gate": True,
        "camera_preserved": True,
        "outsole_preserved": True,
        "fit_relation": "same",
    }
    (root / "manifests" / "upper_manifest.json").write_text(
        json.dumps(
            {
                "shoes": [
                    {
                        "shoe_id": "shoe",
                        "views": [
                            {"view_id": view_id, "quality": quality}
                            for view_id in views
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_adjacent_calibration_fixture(
        root,
        config=config,
        config_sha256=digest,
    )

    job_id = (
        "feature__geometry_calibration__clean__"
        "dinov3_vitl16_pixal3d__user_37_grid_track"
    )
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    registry.register(
        JobDefinition(
            job_id,
            "feature",
            {
                "state_id": "clean",
                "track": "controlled",
                "analysis_track": "geometry_calibration",
                "shoe_id": "shoe",
                "quartet_id": "q3",
                "state_dir": str(state_dir.resolve()),
                "camera_path": str(
                    (root / "cameras" / "shoe" / "q3" / "cameras.npz").resolve()
                ),
                "config_sha256": digest,
                "job_type": "geometry_calibration_feature",
                "backbone": _L_BACKBONE,
                "feature_track": "user_37_grid_track",
                "variant_keys": variant_keys,
                "geometry_contract": "fixed_base_only",
                "output_dir": str(
                    (root / "features" / "geometry_calibration" / "clean").resolve()
                ),
            },
        )
    )
    return root, registry, job_id


def _registry_row(registry: JobRegistry, job_id: str) -> sqlite3.Row:
    with sqlite3.connect(registry.path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT state, attempts, worker_id, output_manifest_json, error
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    assert row is not None
    return row


def _patch_calibration_work(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_CalibrationBackbone, _CalibrationNAF, list[None]]:
    backbone = _CalibrationBackbone()
    naf = _CalibrationNAF()
    score_calls: list[None] = []

    def fake_score(*args, **kwargs) -> SimpleNamespace:
        del args, kwargs
        score_calls.append(None)
        return _fake_calibration_result()

    monkeypatch.setattr(
        "met3r.upper_metric_worker.load_feature_backbone",
        lambda _: backbone,
    )
    monkeypatch.setattr(
        "met3r.upper_metric_worker.load_naf_upsampler",
        lambda **_: naf,
    )
    monkeypatch.setattr(
        "met3r.upper_metric_worker.score_geometry_sensitivity_pair",
        fake_score,
    )
    monkeypatch.setattr(
        "met3r.upper_metric_worker._fit_or_load_pca",
        lambda *args: (_ for _ in ()).throw(AssertionError("PCA")),
    )
    return backbone, naf, score_calls


def test_adjacent_q3_calibration_worker_runs_only_frozen_variants_in_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch payload-order execution, incomplete q3 calibration rows, or raw/PCA storage."""

    root, registry, job_id = _make_adjacent_calibration_run(
        tmp_path,
        [_L592_NAF, _L37_BILINEAR],
    )
    backbone, naf, score_calls = _patch_calibration_work(monkeypatch)

    summary = run_feature_worker(
        root,
        _L_BACKBONE,
        "cpu",
        job_ids=[job_id],
    )

    assert summary.completed == 1
    row = _registry_row(registry, job_id)
    assert (row["state"], row["attempts"], row["worker_id"]) == (
        "completed",
        1,
        f"feature:{_L_BACKBONE}",
    )
    part = root / "metrics" / "parts" / "geometry_calibration" / f"{job_id}.parquet"
    frame = pd.read_parquet(part)
    assert len(frame) == 408
    assert frame.groupby("variant_key", sort=False).size().to_dict() == {
        _L37_BILINEAR: 204,
        _L592_NAF: 204,
    }

    pair_order = {
        "40": (
            ("shoe__q3__40__41", 1, 7.0),
            ("shoe__q3__40__42", 2, 15.0),
            ("shoe__q3__40__0", 3, 22.5),
        ),
        "41": (
            ("shoe__q3__40__41", 1, 7.0),
            ("shoe__q3__41__42", 1, 8.0),
            ("shoe__q3__41__0", 2, 15.5),
        ),
        "42": (
            ("shoe__q3__40__42", 2, 15.0),
            ("shoe__q3__41__42", 1, 8.0),
            ("shoe__q3__42__0", 1, 7.5),
        ),
        "0": (
            ("shoe__q3__40__0", 3, 22.5),
            ("shoe__q3__41__0", 2, 15.5),
            ("shoe__q3__42__0", 1, 7.5),
        ),
    }
    perturbation_order = (
        ("rotation", 0.25),
        ("rotation", 0.5),
        ("rotation", 1.0),
        ("rotation", 2.0),
        ("rotation", 5.0),
        ("relative_translation", 0.0025),
        ("relative_translation", 0.005),
        ("relative_translation", 0.01),
        ("relative_translation", 0.02),
        ("depth_scale", 1.01),
        ("depth_scale", 1.02),
        ("depth_scale", 1.05),
        ("depth_scale", 1.1),
        ("relative_depth_noise", 0.005),
        ("relative_depth_noise", 0.01),
        ("relative_depth_noise", 0.02),
        ("relative_depth_noise", 0.05),
    )
    expected_order = [
        (variant_key, view_id, kind, magnitude, pair_id, steps, angle)
        for variant_key in (_L37_BILINEAR, _L592_NAF)
        for view_id in ("40", "41", "42", "0")
        for kind, magnitude in perturbation_order
        for pair_id, steps, angle in pair_order[view_id]
    ]
    assert list(
        frame[
            [
                "variant_key",
                "changed_view",
                "perturbation_kind",
                "magnitude",
                "pair_id",
                "baseline_steps",
                "angular_separation_deg",
            ]
        ].itertuples(index=False, name=None)
    ) == expected_order
    assert frame[
        [
            "job_id",
            "analysis_track",
            "feature_track",
            "backbone",
            "backbone_revision",
            "config_sha256",
            "camera_provenance",
            "camera_label",
            "track",
            "shoe_id",
            "quartet_id",
            "image_state_id",
            "corruption",
            "severity",
        ]
    ].drop_duplicates().to_dict(orient="records") == [
        {
            "job_id": job_id,
            "analysis_track": "geometry_calibration",
            "feature_track": "user_37_grid_track",
            "backbone": _L_BACKBONE,
            "backbone_revision": BACKBONE_SPECS[_L_BACKBONE].revision,
            "config_sha256": hashlib.sha256(
                (root / "config" / "frozen_config.json").read_bytes()
            ).hexdigest(),
            "camera_provenance": "fixed_precomputed_da3",
            "camera_label": CAMERA_MODE_LABEL,
            "track": "controlled",
            "shoe_id": "shoe",
            "quartet_id": "q3",
            "image_state_id": "clean",
            "corruption": "clean",
            "severity": 0.0,
        }
    ]
    assert frame[
        ["variant_key", "input_size", "feature_mode", "resolution"]
    ].drop_duplicates().to_dict(orient="records") == [
        {
            "variant_key": _L37_BILINEAR,
            "input_size": 592,
            "feature_mode": "bilinear",
            "resolution": 37,
        },
        {
            "variant_key": _L592_NAF,
            "input_size": 592,
            "feature_mode": "naf_normal_guide",
            "resolution": 592,
        },
    ]
    assert (backbone.calls, naf.calls, len(score_calls)) == (1, 2, 408)

    visual_manifest = root / "visualizations" / job_id / "manifest.json"
    assert [path.name for path in visual_manifest.parent.iterdir()] == [
        "manifest.json"
    ]
    assert json.loads(visual_manifest.read_text(encoding="utf-8")) == {"files": []}
    assert json.loads(row["output_manifest_json"]) == {
        "files": [
            {
                "path": str(part.resolve()),
                "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            },
            {
                "path": str(visual_manifest.resolve()),
                "sha256": hashlib.sha256(
                    visual_manifest.read_bytes()
                ).hexdigest(),
            },
        ]
    }
    assert not (root / "pca").exists()
    assert not (root / "features").exists()
    _assert_no_persistent_features(root)


def test_adjacent_q3_calibration_worker_rejects_available_unapproved_variant_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch calibration allowlist bypass, retry, or output staging before validation."""

    unapproved = (
        "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__74"
    )
    root, registry, job_id = _make_adjacent_calibration_run(
        tmp_path,
        [_L592_NAF, unapproved, _L37_BILINEAR],
    )
    backbone, naf, score_calls = _patch_calibration_work(monkeypatch)

    with pytest.raises(
        ValueError,
        match="adjacent calibration must use exactly the frozen variants",
    ):
        run_feature_worker(root, _L_BACKBONE, "cpu", job_ids=[job_id])

    first = _registry_row(registry, job_id)
    assert (first["state"], first["attempts"], first["worker_id"]) == (
        "failed",
        1,
        None,
    )
    assert first["output_manifest_json"] is None
    assert "adjacent calibration must use exactly the frozen variants" in first["error"]
    assert run_feature_worker(
        root,
        _L_BACKBONE,
        "cpu",
        job_ids=[job_id],
    ).completed == 0
    second = _registry_row(registry, job_id)
    assert (second["state"], second["attempts"]) == ("failed", 1)
    assert (backbone.calls, naf.calls, len(score_calls)) == (0, 0, 0)
    assert not (root / "metrics").exists()
    assert not (root / "visualizations").exists()
    assert not (root / "pca").exists()
    assert not (root / "features").exists()
    _assert_no_persistent_features(root)


@pytest.mark.parametrize("backbone", ["dino_vitb16", "dinov2_vitb14", "dinov3_vitb16"])
def test_adjacent_controlled_worker_uses_clean_geometry_and_posthoc_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backbone: str) -> None:
    """End-to-end q3 sentinel: fixed-base scoring must not touch changed geometry or PCA."""

    root = tmp_path / "run"
    for name in ("config", "manifests", "registry", "states", "geometry/calibration", "metrics/parts", "visualizations"):
        (root / name).mkdir(parents=True, exist_ok=True)
    views = ("40", "41", "42", "0")
    config = {"quartets": [{}], "stage_size": 504, "camera_label": CAMERA_MODE_LABEL, "seed": 42,
              "metric": {"relative_depth_tolerance": .03, "min_overlap_fraction": .08, "confidence_percentile": 10.},
              "models": {backbone: {"revision": BACKBONE_SPECS[backbone].revision}, "naf_revision": NAF_REVISION},
              "visualization": {"policy": "posthoc_audit_only"}}
    raw = json.dumps(config).encode(); digest = hashlib.sha256(raw).hexdigest()
    (root / "config/frozen_config.json").write_bytes(raw)
    (root / "run_manifest.json").write_text(json.dumps({"config_sha256": digest}))
    variants = [{"feature_track": "track", "backbone": backbone, "input_size": 8, "feature_mode": "bilinear", "resolution": 2}]
    (root / "manifests/feature_variants.json").write_text(json.dumps(variants))
    rows = []
    for sid, changed, corruption in (("clean", None, "clean"), ("changed", "0", "jpeg")):
        directory = root / "states" / sid; (directory / "images").mkdir(parents=True)
        for index, view in enumerate(views): Image.new("RGB", (504, 504), (index * 20, 0, 0)).save(directory / "images" / f"{view}.png")
        state = {"state_id": sid, "track": "controlled", "shoe_id": "shoe", "quartet_id": "q3", "view_ids": list(views), "changed_view": changed, "corruption": corruption, "severity": 0. if changed is None else .5}
        (directory / "state.json").write_text(json.dumps({"state": state, "stage_size": 504}))
        rows.append(state)
    (root / "manifests/image_states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pair = {"design_id": "C1", "track": "controlled", "shoe_id": "shoe", "quartet_id": "q3", "pair_id": "shoe__q3__42__0", "first_view": "42", "second_view": "0", "baseline_steps": 1, "angular_separation_deg": 7.5, "changed_view": "0", "image_state_id": "changed", "corruption": "jpeg", "severity": .5}
    (root / "manifests/pair_conditions.jsonl").write_text(json.dumps(pair) + "\n")
    (root / "manifests/upper_manifest.json").write_text(json.dumps({"shoes": [{"shoe_id": "shoe", "views": [{"view_id": v, "quality": {"semantic_gate": True, "preservation_gate": True, "camera_preserved": True, "outsole_preserved": True, "fit_relation": "same"}} for v in views]}]}))
    geometry = root / "geometry/calibration/clean"; geometry.mkdir()
    np.savez_compressed(geometry / "depth_geometry.npz", depth=np.ones((4,504,504),np.float32), mask=np.ones((4,504,504),bool), confidence=np.ones((4,504,504),np.float32), intrinsics=np.broadcast_to(np.eye(3),(4,3,3)).copy(), extrinsics=np.broadcast_to(np.eye(4),(4,4,4)).copy(), meta__view_ids=np.asarray(views), meta__camera_groups=np.asarray(("upper",)*4), meta__camera_provenance=np.asarray("fixed_precomputed_da3"))
    (geometry / "depth_geometry.json").write_text(json.dumps({"camera_mode": CAMERA_MODE_LABEL, "camera_provenance": "fixed_precomputed_da3", "process_res": 504, "view_ids": list(views), "camera_groups": ["upper"]*4}))
    registry = JobRegistry(root / "registry/jobs.sqlite")
    keys = [f"track__{backbone}__8__bilinear__2"]
    payload = {"job_type": "feature", "state_id": "changed", "track": "controlled", "analysis_track": "controlled", "shoe_id": "shoe", "quartet_id": "q3", "state_dir": str((root / "states/changed").resolve()), "config_sha256": digest, "backbone": backbone, "feature_track": "track", "variant_keys": list(reversed(keys)), "geometry_contract": "fixed_base_only"}
    registry.register(JobDefinition("wanted", "feature", payload)); registry.register(JobDefinition("other", "feature", payload))
    monkeypatch.setattr("met3r.upper_metric_worker.load_feature_backbone", lambda _: _TinyBackbone())
    monkeypatch.setattr("met3r.upper_metric_worker.load_naf_upsampler", lambda **_: _TinyNAF())
    monkeypatch.setattr("met3r.upper_metric_worker._fit_or_load_pca", lambda *args: (_ for _ in ()).throw(AssertionError("PCA")))
    assert run_feature_worker(root, backbone, "cpu", job_ids=["wanted"]).completed == 1
    frame = pd.read_parquet(root / "metrics/parts/track/wanted.parquet")
    assert len(frame) == 1 and frame["variant_key"].tolist() == keys
    assert frame.iloc[0][["s00", "s01"]].nunique() == 1 and frame.iloc[0][["s10", "s11"]].nunique() == 1
    assert frame.iloc[0]["geometry_effect"] == 0 and frame.iloc[0]["interaction"] == 0
    assert frame.iloc[0]["quartet_id"] == "q3" and frame.iloc[0]["baseline_steps"] == 1 and frame.iloc[0]["angular_separation_deg"] == 7.5
    assert registry.validated_state("other") == "pending"
    visual = root / "visualizations/wanted/manifest.json"; assert json.loads(visual.read_text()) == {"files": []}
    assert not list((root / "pca").rglob("*")) if (root / "pca").exists() else True


def test_adjacent_stress_worker_loads_only_ordered_base_and_adaptive_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end q3 stress keeps current order while loading both geometry cells."""
    root = tmp_path / "stress"
    for name in ("config", "manifests", "registry", "states", "geometry/calibration", "geometry/stress", "metrics/parts", "visualizations"):
        (root / name).mkdir(parents=True, exist_ok=True)
    views = ("40", "41", "42", "0")
    config = {"quartets": [{}], "stage_size": 504, "camera_label": CAMERA_MODE_LABEL, "seed": 42, "metric": {"relative_depth_tolerance": .03, "min_overlap_fraction": .08, "confidence_percentile": 10.}, "models": {"fake": {"revision": "test"}, "naf_revision": NAF_REVISION}, "visualization": {"policy": "posthoc_audit_only"}}
    raw = json.dumps(config).encode(); digest = hashlib.sha256(raw).hexdigest()
    (root / "config/frozen_config.json").write_bytes(raw); (root / "run_manifest.json").write_text(json.dumps({"config_sha256": digest}))
    (root / "manifests/feature_variants.json").write_text(json.dumps([{ "feature_track":"track", "backbone":"fake", "input_size":8, "feature_mode":"native", "resolution":2}]))
    for sid, track, corruption in (("clean", "controlled", "clean"), ("stress-state", "stress", "deformed")):
        directory = root / "states" / sid; (directory / "images").mkdir(parents=True)
        for index, view in enumerate(views): Image.new("RGB", (504,504), (index * 30,0,0)).save(directory / "images" / f"{view}.png")
        state = {"state_id":sid,"track":track,"shoe_id":"shoe","quartet_id":"q3","view_ids":list(views),"changed_view":None,"corruption":corruption,"severity":0. if sid=="clean" else 1.}
        (directory / "state.json").write_text(json.dumps({"state":state,"stage_size":504}))
        with (root / "manifests/image_states.jsonl").open("a") as handle: handle.write(json.dumps(state)+"\n")
    pair = {"design_id":"S1","track":"stress","shoe_id":"shoe","quartet_id":"q3","pair_id":"shoe__q3__42__0","first_view":"42","second_view":"0","baseline_steps":1,"angular_separation_deg":7.5,"changed_view":None,"image_state_id":"stress-state","corruption":"deformed","severity":1.}
    (root / "manifests/pair_conditions.jsonl").write_text(json.dumps(pair)+"\n")
    (root / "manifests/upper_manifest.json").write_text(json.dumps({"shoes":[{"shoe_id":"shoe","views":[{"view_id":v,"quality":{"semantic_gate":True,"preservation_gate":True,"camera_preserved":True,"outsole_preserved":True,"fit_relation":"same"}} for v in views]}]}))
    def write_geometry(directory: Path, depth: float) -> None:
        directory.mkdir(parents=True, exist_ok=True); np.savez_compressed(directory/"depth_geometry.npz", depth=np.full((4,504,504),depth,np.float32),mask=np.ones((4,504,504),bool),confidence=np.ones((4,504,504),np.float32),intrinsics=np.broadcast_to(np.eye(3),(4,3,3)).copy(),extrinsics=np.broadcast_to(np.eye(4),(4,4,4)).copy(),meta__view_ids=np.asarray(views),meta__camera_groups=np.asarray(("upper",)*4),meta__camera_provenance=np.asarray("fixed_precomputed_da3")); (directory/"depth_geometry.json").write_text(json.dumps({"camera_mode":CAMERA_MODE_LABEL,"camera_provenance":"fixed_precomputed_da3","process_res":504,"view_ids":list(views),"camera_groups":["upper"]*4}))
    write_geometry(root/"geometry/calibration/clean", 1.0); write_geometry(root/"geometry/stress/stress-state", 1.1)
    registry=JobRegistry(root/"registry/jobs.sqlite"); payload={"job_type":"stress","state_id":"stress-state","track":"stress","analysis_track":"stress","shoe_id":"shoe","quartet_id":"q3","state_dir":str((root/"states/stress-state").resolve()),"config_sha256":digest,"backbone":"fake","feature_track":"track","variant_keys":["track__fake__8__native__2"],"geometry_contract":"base_plus_adaptive"}; registry.register(JobDefinition("stress-job","feature",payload))
    monkeypatch.setattr("met3r.upper_metric_worker.load_feature_backbone",lambda _: _TinyBackbone()); monkeypatch.setattr("met3r.upper_metric_worker.load_naf_upsampler",lambda **_: _TinyNAF())
    from met3r.upper_metric_worker import _load_geometry as real_load
    loaded=[]
    def tracked(path, *args, **kwargs): loaded.append(Path(path)); return real_load(path,*args,**kwargs)
    monkeypatch.setattr("met3r.upper_metric_worker._load_geometry", tracked)
    assert run_feature_worker(root,"fake","cpu",job_ids=["stress-job"]).completed == 1
    assert loaded == [root/"geometry/calibration/clean/depth_geometry.npz", root/"geometry/stress/stress-state/depth_geometry.npz"]
    frame=pd.read_parquet(root/"metrics/parts/track/stress-job.parquet"); row=frame.iloc[0]
    assert (row["quartet_id"],row["baseline_steps"],row["angular_separation_deg"],row["geometry_contract"],row["camera_provenance"]) == ("q3",1,7.5,"base_plus_adaptive","fixed_precomputed_da3")
    assert json.loads((root/"visualizations/stress-job/manifest.json").read_text()) == {"files": []} and not (root/"pca").exists()
