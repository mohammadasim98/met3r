from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace
from pathlib import Path

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
    _sensitivity_row,
    run_feature_worker,
)
from met3r.known_camera import KnownCameraConfig


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


def _write_adjacent_calibration_fixture(root: Path) -> tuple[_RunContract, _StateRecord]:
    """Build the exact 4 x 17 q3 calibration-manifest shape used by the worker."""
    views = ("40", "41", "42", "0")
    values = {"rotation": [0.25, 0.5, 1.0, 2.0, 5.0], "relative_translation": [0.0025, 0.005, 0.01, 0.02], "depth_scale": [1.01, 1.02, 1.05, 1.1], "relative_depth_noise": [0.005, 0.01, 0.02, 0.05]}
    config = {"geometry_calibration": {"rotation_degrees": values["rotation"], "relative_translation": values["relative_translation"], "depth_scale": values["depth_scale"], "relative_depth_noise_std": values["relative_depth_noise"]}}
    directory = root / "geometry" / "calibration" / "clean"; directory.mkdir(parents=True)
    base = directory / "depth_geometry.npz"; base.write_bytes(b"base")
    records = []
    for index, view in enumerate(views):
        for kind, magnitudes in values.items():
            for magnitude in magnitudes:
                path = directory / view / f"{kind}_{magnitude}" / "geometry.npz"; path.parent.mkdir(parents=True); path.write_bytes(b"geometry")
                records.append({"condition_id": f"{kind}_{magnitude}", "kind": kind, "magnitude": magnitude, "path": str(path.resolve()), "view_index": index, "view_id": view, "sha256": sha256_file(path)})
    (directory / "calibration_manifest.json").write_text(json.dumps({"artifact_type": "upper_geometry_calibration", "base_geometry": str(base.resolve()), "config_sha256": "config", "seed": 42, "views": list(views), "artifacts": records}))
    contract = _RunContract(root, config, "config", "full", KnownCameraConfig(.03, .08), 10., "test", "posthoc_audit_only")
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
