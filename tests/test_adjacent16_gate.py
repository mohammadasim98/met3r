from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
from PIL import Image

from met3r.adjacent16_gate import (
    _registry_rows,
    _source_manifest,
    _validate_metric_keys,
    _validate_registry_graph,
    validate_adjacent16_final_gate,
    validate_timed_runtime_projection,
    write_timed_sentinel_manifest,
)
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import Adjacent16Manifest, AdjacentShoeRecord, AdjacentViewRecord
from met3r.adjacent16_design import build_adjacent16_design
from met3r.adjacent16_prepare import _write_manifests, register_adjacent16_jobs
from met3r.deformation_audit import DeformationQuality
from tests.helpers_adjacent16 import write_valid_stress


@pytest.fixture(autouse=True)
def _relocate_frozen_gate_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep production-shaped gate artifacts inside each pytest run root."""

    def load_local_contract(path: str | Path) -> object:
        contract = load_adjacent16_contract(path)
        run_root = Path(path).resolve().parent.parent
        return replace(
            contract,
            reconstruction_root=(run_root / "reconstruction").resolve(),
            stress_root=(run_root / "inputs" / "stress").resolve(),
            output_root=run_root,
        )

    monkeypatch.setattr(
        "met3r.adjacent16_gate.load_adjacent16_contract", load_local_contract
    )


def _write_frozen_manifest_fixture(root: Path) -> tuple[tuple[Path, ...], object, object]:
    contract = replace(
        load_adjacent16_contract("configs/adjacent16_core.json"),
        reconstruction_root=(root / "reconstruction").resolve(),
        stress_root=(root / "inputs" / "stress").resolve(),
        output_root=root.resolve(),
    )
    positions = {view: index for index, view in enumerate(contract.camera_order)}
    shoes = []
    root.mkdir(parents=True, exist_ok=True)
    for shoe_index, (shoe, camera_dataset_id) in enumerate(
        contract.raw["shoes"].items()
    ):
        views = []
        for quartet in contract.quartets:
            for view in quartet.view_ids:
                serial = shoe_index * 16 + len(views)
                source = root / "sources" / shoe / quartet.quartet_id / f"{view}.png"
                mask = source.with_name(f"{view}.mask.png")
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB", (1080, 1080), (int(view), shoe_index * 60, 127)
                ).save(source)
                Image.new("L", (1080, 1080), 128 + serial).save(mask)
                write_valid_stress(
                    contract.stress_root,
                    shoe,
                    quartet.quartet_id,
                    view,
                )
                stress_dir = (
                    contract.stress_root
                    / shoe
                    / quartet.quartet_id
                    / f"view_{view}"
                    / "deformation"
                )
                stress = stress_dir / "deformed.png"
                stress_mask = stress_dir / "deformed_mask.png"
                views.append(
                    AdjacentViewRecord(
                        shoe_id=shoe,
                        camera_dataset_id=camera_dataset_id,
                        quartet_id=quartet.quartet_id,
                        view_id=view,
                        orbit_position=positions[view],
                        source_path=source.resolve(),
                        source_mask_path=mask.resolve(),
                        stress_path=stress.resolve(),
                        stress_mask_path=stress_mask.resolve(),
                        deformation_json_path=(stress_dir / "deformation.json").resolve(),
                        metrics_json_path=(stress_dir / "metrics.json").resolve(),
                        task_json_path=(stress_dir / "task.json").resolve(),
                        processed_image_correlation=1.0,
                        camera_azimuth_deg=float(
                            (positions[view] * 7.5) % 360.0
                        ),
                        source_sha256=sha256_file(source),
                        source_mask_sha256=sha256_file(mask),
                        stress_sha256=sha256_file(stress),
                        stress_mask_sha256=sha256_file(stress_mask),
                        quality=DeformationQuality(
                            status="ok",
                            selected_attempt_id="attempt_01",
                            semantic_gate=True,
                            preservation_gate=True,
                            camera_preserved=True,
                            outsole_preserved=True,
                            fit_relation="equivalent",
                            task_status="ok",
                        ),
                    )
                )
        shoes.append(AdjacentShoeRecord(shoe, camera_dataset_id, tuple(views)))
    manifest = Adjacent16Manifest(
        contract.reconstruction_root, contract.stress_root, tuple(shoes)
    )
    design = build_adjacent16_design(manifest, contract)
    _write_manifests(root, manifest, design)
    manifests = root / "manifests"
    state_path = manifests / "image_states.jsonl"
    pair_path = manifests / "pair_conditions.jsonl"
    variant_path = manifests / "feature_variants.json"
    return (state_path, pair_path, variant_path), contract, design


def _registry_graph_fixture(tmp_path: Path) -> tuple[Path, object, object]:
    root = tmp_path / "run"
    _paths, contract, design = _write_frozen_manifest_fixture(root)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, "a" * 64)
    rows = _registry_rows(root)
    assert _validate_registry_graph(root, contract, "a" * 64, rows) is True
    return root, contract, rows


def test_registry_graph_rejects_mutated_stress_paths_and_hashes(
    tmp_path: Path,
) -> None:
    """Catch runtime/final graph validation discarding stress file identity."""

    root, contract, rows = _registry_graph_fixture(tmp_path)
    path = root / "manifests" / "adjacent16_manifest.json"
    original = path.read_bytes()
    mutations = (
        "stress_path",
        "stress_mask_path",
        "deformation_json_path",
        "metrics_json_path",
        "task_json_path",
        "stress_sha256",
        "stress_mask_sha256",
    )
    for field in mutations:
        manifest = json.loads(original)
        manifest["shoes"][0]["views"][0][field] = (
            "0" * 64
            if field.endswith("sha256")
            else str((root / f"does-not-exist-{field}").resolve())
        )
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        assert (
            _validate_registry_graph(root, contract, "a" * 64, rows) is False
        ), field
    path.write_bytes(original)
    manifest = json.loads(original)
    artifact = Path(manifest["shoes"][0]["views"][0]["stress_path"])
    artifact_bytes = artifact.read_bytes()
    artifact.unlink()
    assert _validate_registry_graph(root, contract, "a" * 64, rows) is False
    artifact.write_bytes(artifact_bytes)
    assert _validate_registry_graph(root, contract, "a" * 64, rows) is True


def test_source_manifest_keeps_clean_preflight_independent_of_stress_evidence(
    tmp_path: Path,
) -> None:
    """Catch stress-required runtime validation leaking into clean preflight."""

    root, contract, _rows = _registry_graph_fixture(tmp_path)
    path = root / "manifests" / "adjacent16_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shoes"][0]["views"][0]["stress_path"] = str(
        (root / "does-not-exist.png").resolve()
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    clean = _source_manifest(root, contract, require_stress=False)

    assert len(clean) == 16
    assert sum(len(records) for records in clean.values()) == 64
    with pytest.raises(ValueError):
        _source_manifest(root, contract, require_stress=True)


def test_registry_graph_rejects_stress_dimensions_audits_and_manifest_quality(
    tmp_path: Path,
) -> None:
    """Catch runtime/final validation accepting failed or detached audits."""

    root, contract, rows = _registry_graph_fixture(tmp_path)
    manifest_path = root / "manifests" / "adjacent16_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    view = json.loads(manifest_bytes)["shoes"][0]["views"][0]

    for label, artifact_field, mode, size in (
        ("stress-rgb-size", "stress_path", "RGB", (1079, 1080)),
        ("stress-mask-size", "stress_mask_path", "L", (1080, 1079)),
    ):
        artifact = Path(view[artifact_field])
        original = artifact.read_bytes()
        Image.new(mode, size).save(artifact)
        assert (
            _validate_registry_graph(root, contract, "a" * 64, rows) is False
        ), label
        artifact.write_bytes(original)

    audit_mutations = (
        ("deformation-status", "deformation_json_path", ("status",), "failed"),
        ("task-status", "task_json_path", ("status",), "failed"),
        (
            "semantic-gate",
            "deformation_json_path",
            ("attempts", 0, "semantic", "passes_gate"),
            False,
        ),
        (
            "preservation-gate",
            "deformation_json_path",
            ("attempts", 0, "preservation", "passes_gate"),
            False,
        ),
        (
            "camera-preserved",
            "deformation_json_path",
            ("attempts", 0, "semantic", "camera_preserved"),
            False,
        ),
        (
            "outsole-preserved",
            "deformation_json_path",
            ("attempts", 0, "semantic", "outsole_preserved"),
            False,
        ),
        (
            "selected-attempt",
            "deformation_json_path",
            ("selected_attempt_id",),
            "missing",
        ),
        (
            "fit-relation",
            "deformation_json_path",
            ("attempts", 0, "comparison", "relation"),
            "fabricated",
        ),
        (
            "metrics-cross-check",
            "metrics_json_path",
            ("after", "fit_score"),
            999.0,
        ),
    )
    for label, field, keys, replacement in audit_mutations:
        artifact = Path(view[field])
        original = artifact.read_bytes()
        payload = json.loads(original)
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
        artifact.write_text(json.dumps(payload))
        assert (
            _validate_registry_graph(root, contract, "a" * 64, rows) is False
        ), label
        artifact.write_bytes(original)

    manifest = json.loads(manifest_bytes)
    manifest["shoes"][0]["views"][0]["quality"]["semantic_gate"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert _validate_registry_graph(root, contract, "a" * 64, rows) is False
    manifest_path.write_bytes(manifest_bytes)
    assert _validate_registry_graph(root, contract, "a" * 64, rows) is True


def test_independent_frozen_manifest_fixture_is_exact_and_byte_stable(tmp_path: Path) -> None:
    """Catch test fixtures that silently shrink the frozen manifest design."""

    first, _contract, _design = _write_frozen_manifest_fixture(tmp_path / "one")
    second, _contract2, _design2 = _write_frozen_manifest_fixture(tmp_path / "two")
    states = [json.loads(line) for line in first[0].read_text().splitlines()]
    pairs = [json.loads(line) for line in first[1].read_text().splitlines()]
    source_manifest = json.loads(
        (tmp_path / "one/manifests/adjacent16_manifest.json").read_text()
    )
    source_views = [
        view for shoe in source_manifest["shoes"] for view in shoe["views"]
    ]

    assert set(source_manifest) == {"reconstruction_root", "stress_root", "shoes"}
    assert len(source_views) == 64
    assert len({row["source_path"] for row in source_views}) == 64
    assert len({row["source_mask_path"] for row in source_views}) == 64
    assert len({row["source_sha256"] for row in source_views}) == 64
    assert len({row["source_mask_sha256"] for row in source_views}) == 64
    assert Path(source_manifest["stress_root"]).is_relative_to(tmp_path / "one")
    assert all(
        all(
            row[field] is not None
            for field in (
                "stress_path",
                "stress_mask_path",
                "deformation_json_path",
                "metrics_json_path",
                "task_json_path",
                "stress_sha256",
                "stress_mask_sha256",
                "quality",
            )
        )
        for row in source_views
    )
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


def _metric_row(
    condition: dict[str, object],
    variant: object,
    config_hash: str,
    *,
    analysis_track: str,
    **extra: object,
) -> dict[str, object]:
    state_id = condition["image_state_id"]
    job_id = (
        f"feature__{analysis_track}__{state_id}__"
        f"{variant.backbone}__{variant.feature_track}"
    )
    return {
        **condition,
        "job_id": job_id,
        "analysis_track": analysis_track,
        "variant_key": variant.key,
        "feature_track": variant.feature_track,
        "backbone": variant.backbone,
        "input_size": variant.input_size,
        "feature_mode": variant.feature_mode,
        "resolution": variant.resolution,
        "config_sha256": config_hash,
        "camera_label": "fixed/precomputed DA3 cameras (not calibrated GT)",
        "camera_provenance": "fixed_precomputed_da3",
        "abstained": False,
        "score": .8,
        "s00": .8,
        "s10": .8,
        "s01": .8,
        "s11": .8,
        "s11_first_to_second_overlap": .8,
        "s11_second_to_first_overlap": .8,
        **{
            f"{prefix}_{field}": None
            for prefix in ("first", "second")
            for field in (
                "semantic_gate",
                "preservation_gate",
                "camera_preserved",
                "outsole_preserved",
                "fit_relation",
            )
        },
        **extra,
    }


def _write_frozen_metric_tables(root: Path, contract: object, design: object, config_hash: str) -> dict[str, Path]:
    controlled = [_metric_row(asdict(condition), variant, config_hash, analysis_track="controlled") for condition in design.controlled_conditions for variant in design.feature_variants]
    source_manifest = json.loads(
        (root / "manifests" / "adjacent16_manifest.json").read_text()
    )
    quality_by_view = {
        (shoe["shoe_id"], view["view_id"]): view["quality"]
        for shoe in source_manifest["shoes"]
        for view in shoe["views"]
    }
    stress = []
    for condition in design.stress_conditions:
        record = asdict(condition)
        quality_columns = {
            f"{prefix}_{field}": quality_by_view[
                condition.shoe_id, view_id
            ][field]
            for prefix, view_id in (
                ("first", condition.first_view),
                ("second", condition.second_view),
            )
            for field in (
                "semantic_gate",
                "preservation_gate",
                "camera_preserved",
                "outsole_preserved",
                "fit_relation",
            )
        }
        for variant in design.feature_variants:
            stress.append(
                _metric_row(
                    record,
                    variant,
                    config_hash,
                    analysis_track="stress",
                    **quality_columns,
                )
            )
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
                            calibration.append(_metric_row(condition, variant, config_hash, analysis_track="geometry_calibration", changed_view=changed, perturbation_kind=kind, magnitude=float(magnitude)))
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
    assert stress[
        [
            "first_semantic_gate",
            "first_preservation_gate",
            "first_camera_preserved",
            "first_outsole_preserved",
            "second_semantic_gate",
            "second_preservation_gate",
            "second_camera_preserved",
            "second_outsole_preserved",
        ]
    ].to_numpy(dtype=bool).all()
    assert set(stress["first_fit_relation"]) == set(stress["second_fit_relation"]) == {
        "equivalent"
    }


def test_metric_validation_rejects_stress_quality_that_differs_from_source_view(
    tmp_path: Path,
) -> None:
    """Catch final validation accepting worker quality detached from its audit."""

    root = tmp_path / "run"
    _paths, contract, design = _write_frozen_manifest_fixture(root)
    tables = _write_frozen_metric_tables(root, contract, design, "a" * 64)
    frames = {
        track: pd.read_parquet(path)
        for track, path in tables.items()
    }
    assert _validate_metric_keys(root, frames, contract) is True

    row = frames["stress"].index[0]
    columns = tuple(
        f"{prefix}_{field}"
        for prefix in ("first", "second")
        for field in (
            "semantic_gate",
            "preservation_gate",
            "camera_preserved",
            "outsole_preserved",
            "fit_relation",
        )
    )
    for column in columns:
        original = frames["stress"].loc[row, column]
        frames["stress"].loc[row, column] = (
            "worsened" if column.endswith("fit_relation") else not bool(original)
        )
        assert _validate_metric_keys(root, frames, contract) is False, column
        frames["stress"].loc[row, column] = original
    missing = dict(frames)
    missing["stress"] = frames["stress"].drop(columns=["second_fit_relation"])
    assert _validate_metric_keys(root, missing, contract) is False
    assert _validate_metric_keys(root, frames, contract) is True


def _final_gate_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "run"
    config = Path("configs/adjacent16_core.json").read_bytes()
    (root / "config").mkdir(parents=True)
    (root / "config/frozen_config.json").write_bytes(config)
    digest = __import__("hashlib").sha256(config).hexdigest()
    (root / "run_manifest.json").write_text(json.dumps({"experiment_id": "20260727_adjacent16_core_v1", "config_sha256": digest}))
    _paths, contract, design = _write_frozen_manifest_fixture(root)
    marker = root / "reports" / "marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    registry = JobRegistry(root / "registry/jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, digest)
    _materialize_owned_outputs(root, registry)
    with sqlite3.connect(registry.path) as connection:
        connection.execute("UPDATE jobs SET state='completed', attempts=1")
    metrics = _write_frozen_metric_tables(root, contract, design, digest)
    return {"root": root, "contract": contract, "design": design, "registry": registry, "metrics": metrics}


def _materialize_owned_outputs(root: Path, registry: JobRegistry) -> None:
    """Create lightweight, job-owned fixture outputs and matching manifests."""
    with sqlite3.connect(registry.path) as connection:
        rows = connection.execute("SELECT job_id, phase, payload_json FROM jobs").fetchall()
        for job_id, phase, raw_payload in rows:
            payload = json.loads(raw_payload)
            if phase == "geometry":
                directory = Path(payload["output_dir"])
                paths = (directory / "depth_geometry.npz", directory / "depth_geometry.json")
            else:
                track = "geometry_calibration" if payload["job_type"] == "geometry_calibration_feature" else payload["feature_track"]
                paths = (root / "metrics" / "parts" / track / f"{job_id}.parquet",)
            files = []
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                if phase == "geometry" and path.suffix == ".npz":
                    np.savez_compressed(
                        path,
                        depth=np.ones((1, 1, 1), dtype=np.float32),
                        confidence=np.ones((1, 1, 1), dtype=np.float32),
                        mask=np.ones((1, 1, 1), dtype=bool),
                        extrinsics=np.eye(4, dtype=np.float32)[None],
                        intrinsics=np.eye(3, dtype=np.float32)[None],
                        meta__artifact_type=np.asarray("fixed_camera_depth"),
                        meta__model_id=np.asarray("fixture-da3"),
                        meta__model_revision=np.asarray("fixture-revision"),
                        meta__runtime_fingerprint=np.asarray("fixture-runtime"),
                        meta__source_revision=np.asarray("fixture-source"),
                        meta__source_dirty=np.asarray(False),
                        meta__source_dirty_fingerprint=np.asarray(""),
                        meta__camera_provenance=np.asarray(
                            "fixed_precomputed_da3"
                        ),
                        meta__image_set_id=np.asarray(job_id),
                        meta__view_ids=np.asarray(["1"]),
                        meta__camera_groups=np.asarray(["upper"]),
                        meta__analysis_tier=np.asarray("primary"),
                    )
                else:
                    path.write_bytes(job_id.encode())
                files.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            connection.execute("UPDATE jobs SET output_manifest_json=? WHERE job_id=?", (json.dumps({"files": files}), job_id))


_SUPPORT_FIELDS = ("shoe_id", "quartet_id", "pair_id", "changed_view", "corruption", "severity")


def _variant_support_keys(frame: pd.DataFrame, variant: str) -> set[tuple[object, ...]]:
    rows = frame.loc[frame.variant_key == variant, list(_SUPPORT_FIELDS)]
    keys = {tuple(row) for row in rows.itertuples(index=False, name=None)}
    assert len(rows) == len(keys) == 2592
    return keys


def _abstain_variant_support_keys(
    frame: pd.DataFrame,
    variant: str,
    support_keys: set[tuple[object, ...]],
) -> None:
    variant_rows = frame.loc[frame.variant_key == variant]
    rows_by_key = {
        tuple(row[field] for field in _SUPPORT_FIELDS): index
        for index, row in variant_rows.iterrows()
    }
    assert len(rows_by_key) == 2592
    assert support_keys <= rows_by_key.keys()
    frame.loc[[rows_by_key[key] for key in support_keys], "abstained"] = True


def _nonclean_support_keys(frame: pd.DataFrame, variant: str) -> list[tuple[object, ...]]:
    keys = _variant_support_keys(frame, variant)
    nonclean = sorted(key for key in keys if key[4] != "clean")
    assert len(nonclean) == 2496
    return nonclean


def test_independent_frozen_fixture_passes_the_real_final_gate(tmp_path: Path) -> None:
    """Catch divergence between compact fixture data and final-gate production contracts."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)

    report = validate_adjacent16_final_gate(root)

    assert report.passed is True
    assert report.actual_rows == report.expected_rows == {"controlled": 31104, "geometry_calibration": 6528, "stress": 1152, "total": 38784}
    assert report.controlled_complete_series_coverage == report.all_four_shoe_primary_cell_rate == 1.0
    assert report.minimum_common_support_retention == report.stress_symmetric_pass_rate == 1.0
    assert report.failed_checks == report.unexplained_jobs == report.raw_feature_artifacts == ()
    report_path = root / "reports" / "adjacent16_final_gate.json"
    first_bytes = report_path.read_bytes()
    provenance = json.loads(first_bytes)
    assert all(
        provenance[name]
        for name in (
            "config_sha256",
            "registry_sha256",
            "manifest_sha256",
            "adjacent_manifest_sha256",
            "stress_audit_sha256",
            "metric_input_sha256",
            "output_sha256",
        )
    )
    assert validate_adjacent16_final_gate(root) == report
    assert report_path.read_bytes() == first_bytes
    controlled_path = fixture["metrics"]["controlled"]
    controlled = pd.read_parquet(controlled_path)
    controlled.loc[0, "score"] = .81
    controlled.to_parquet(controlled_path, index=False)
    source_manifest = json.loads(
        (root / "manifests" / "adjacent16_manifest.json").read_text()
    )
    task_path = Path(source_manifest["shoes"][0]["views"][0]["task_json_path"])
    task_path.write_text(
        json.dumps(json.loads(task_path.read_text()), indent=2, sort_keys=True) + "\n"
    )
    changed = validate_adjacent16_final_gate(root)
    changed_provenance = json.loads(report_path.read_text())
    assert changed.passed is True
    assert changed_provenance["metric_input_sha256"] != provenance["metric_input_sha256"]
    assert changed_provenance["stress_audit_sha256"] != provenance["stress_audit_sha256"]


def test_controlled_complete_series_uses_adjacent_whole_series_threshold(tmp_path: Path) -> None:
    """Catch rounding the 9,216-series 80% threshold or treating missing as zero."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"]["controlled"]
    frame = pd.read_parquet(path)
    key = ["shoe_id", "quartet_id", "pair_id", "changed_view", "corruption", "variant_key"]
    groups = list(frame.loc[frame.corruption != "clean"].groupby(key, sort=True).groups.values())
    assert len(groups) == 9216
    for index, rows in enumerate(groups[:1843]):
        row = next(iter(rows))
        if index == 0:
            frame.loc[row, "score"] = float("nan")
        else:
            frame.loc[row, "abstained"] = True
    frame.to_parquet(path, index=False)
    passing = validate_adjacent16_final_gate(root)
    assert passing.controlled_complete_series_coverage == pytest.approx(7373 / 9216)
    assert "controlled_complete_series" not in passing.failed_checks
    row = next(iter(groups[1843]))
    frame.loc[row, "abstained"] = True
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.controlled_complete_series_coverage == pytest.approx(7372 / 9216)
    assert "controlled_complete_series" in failing.failed_checks


def test_all_four_shoe_primary_cells_use_normalized_pair_boundaries(tmp_path: Path) -> None:
    """Require all shoes for a normalized physical cell at the exact 80% boundary."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    contract = fixture["contract"]
    path = fixture["metrics"]["controlled"]
    assert isinstance(root, Path)

    complete = validate_adjacent16_final_gate(root)
    assert complete.passed is True
    assert complete.all_four_shoe_primary_cell_rate == 1.0

    frame = pd.read_parquet(path)
    key = ["shoe_id", "quartet_id", "pair_id", "changed_view", "corruption", "variant_key"]
    groups = frame.loc[frame.corruption != "clean"].groupby(key, sort=True).groups
    expected_shoes = set(contract.raw["shoes"])
    physical_shoes: dict[tuple[object, ...], set[str]] = {}
    target_series: list[tuple[tuple[object, ...], int]] = []
    target_shoe = sorted(expected_shoes)[0]
    for series, rows in groups.items():
        shoe, quartet, pair_id, placement, corruption, variant = series
        pair_parts = str(pair_id).split("__")
        assert pair_parts[:2] == [shoe, quartet]
        physical = (quartet, pair_parts[-2], pair_parts[-1], placement, corruption, variant)
        physical_shoes.setdefault(physical, set()).add(shoe)
        if shoe == target_shoe:
            target_series.append((physical, next(iter(rows))))
    assert len(groups) == 9216
    assert len(physical_shoes) == len(target_series) == 2304
    assert all(shoes == expected_shoes for shoes in physical_shoes.values())

    target_series.sort()
    for _physical, row in target_series[:460]:
        frame.loc[row, "abstained"] = True
    frame.to_parquet(path, index=False)
    passing = validate_adjacent16_final_gate(root)
    assert passing.controlled_complete_series_coverage == pytest.approx((9216 - 460) / 9216)
    assert passing.controlled_complete_series_coverage > .80
    assert passing.all_four_shoe_primary_cell_rate == pytest.approx(1844 / 2304)
    assert passing.passed is True
    assert "all_four_shoe_primary_cells" not in passing.failed_checks

    _physical, row = target_series[460]
    frame.loc[row, "abstained"] = True
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.controlled_complete_series_coverage == pytest.approx((9216 - 461) / 9216)
    assert failing.controlled_complete_series_coverage > .80
    assert failing.all_four_shoe_primary_cell_rate == pytest.approx(1843 / 2304)
    assert failing.passed is False
    assert "all_four_shoe_primary_cells" in failing.failed_checks


def test_common_support_four_b37_backbones_uses_intersection_over_union(tmp_path: Path) -> None:
    """Catch a directional-overlap mean in the four-backbone comparison."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"]["controlled"]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    b37 = frame.loc[
        (frame.feature_mode == "bilinear") & (frame.resolution == 37),
        ["backbone", "variant_key"],
    ].drop_duplicates("variant_key")
    assert len(b37) == 4
    target = b37.loc[b37.backbone == "dino_vitb16", "variant_key"].item()
    missing = _nonclean_support_keys(frame, target)

    _abstain_variant_support_keys(frame, target, set(missing[:777]))
    frame.to_parquet(path, index=False)
    passing = validate_adjacent16_final_gate(root)
    assert passing.minimum_common_support_retention == pytest.approx(1815 / 2592)
    assert passing.controlled_complete_series_coverage > .80
    assert passing.all_four_shoe_primary_cell_rate > .80
    assert passing.passed is True
    assert "common_support_retention" not in passing.failed_checks

    _abstain_variant_support_keys(frame, target, {missing[777]})
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.minimum_common_support_retention == pytest.approx(1814 / 2592)
    assert failing.controlled_complete_series_coverage > .80
    assert failing.all_four_shoe_primary_cell_rate > .80
    assert failing.passed is False
    assert failing.failed_checks == ("common_support_retention",)


def test_common_support_five_l_bilinear_resolutions_uses_disjoint_missing_keys(tmp_path: Path) -> None:
    """Catch a five-way support calculation that omits one resolution or averages overlaps."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"]["controlled"]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    l_bilinear = frame.loc[
        (frame.backbone == "dinov3_vitl16_pixal3d") & (frame.feature_mode == "bilinear"),
        ["resolution", "variant_key"],
    ].drop_duplicates("variant_key").sort_values("resolution")
    assert tuple(l_bilinear.resolution) == (37, 74, 148, 296, 592)
    variants = tuple(l_bilinear.variant_key)
    missing = _nonclean_support_keys(frame, variants[0])
    assert all(set(_nonclean_support_keys(frame, variant)) == set(missing) for variant in variants[1:])
    chunks = (set(missing[:156]), set(missing[156:312]), set(missing[312:467]), set(missing[467:622]), set(missing[622:777]))
    assert len(set().union(*chunks)) == 777
    for variant, chunk in zip(variants, chunks, strict=True):
        _abstain_variant_support_keys(frame, variant, chunk)
    frame.to_parquet(path, index=False)
    passing = validate_adjacent16_final_gate(root)
    assert passing.minimum_common_support_retention == pytest.approx(1815 / 2592)
    assert passing.controlled_complete_series_coverage > .80
    assert passing.all_four_shoe_primary_cell_rate > .80
    assert passing.passed is True
    assert "common_support_retention" not in passing.failed_checks

    _abstain_variant_support_keys(frame, variants[-1], {missing[777]})
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.minimum_common_support_retention == pytest.approx(1814 / 2592)
    assert failing.controlled_complete_series_coverage > .80
    assert failing.all_four_shoe_primary_cell_rate > .80
    assert failing.passed is False
    assert failing.failed_checks == ("common_support_retention",)


def test_common_support_naf_bilinear_pair_uses_intersection_over_union(tmp_path: Path) -> None:
    """Catch one directional NAF-vs-bilinear overlap being treated as the pair result."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"]["controlled"]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    pair = frame.loc[
        (frame.backbone == "dinov3_vitl16_pixal3d") & (frame.resolution == 74),
        ["feature_mode", "variant_key"],
    ].drop_duplicates("variant_key")
    assert set(pair.feature_mode) == {"bilinear", "naf_normal_guide"}
    naf = pair.loc[pair.feature_mode == "naf_normal_guide", "variant_key"].item()
    bilinear = pair.loc[pair.feature_mode == "bilinear", "variant_key"].item()
    missing = _nonclean_support_keys(frame, naf)
    assert set(_nonclean_support_keys(frame, bilinear)) == set(missing)

    _abstain_variant_support_keys(frame, naf, set(missing[:777]))
    frame.to_parquet(path, index=False)
    passing = validate_adjacent16_final_gate(root)
    assert passing.minimum_common_support_retention == pytest.approx(1815 / 2592)
    assert passing.controlled_complete_series_coverage > .80
    assert passing.all_four_shoe_primary_cell_rate > .80
    assert passing.passed is True
    assert "common_support_retention" not in passing.failed_checks

    _abstain_variant_support_keys(frame, naf, {missing[777]})
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.minimum_common_support_retention == pytest.approx(1814 / 2592)
    assert failing.controlled_complete_series_coverage > .80
    assert failing.all_four_shoe_primary_cell_rate > .80
    assert failing.passed is False
    assert failing.failed_checks == ("common_support_retention",)


def test_stress_symmetric_pass_rate_counts_invalid_rows_in_the_frozen_denominator(tmp_path: Path) -> None:
    """Catch invalid stress rows being discarded or treated as a hard evidence failure."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"]["stress"]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    assert len(frame) == 1152
    passing_rows = frame.index[:807]
    abstained_rows = frame.index[807:922]
    subthreshold_rows = frame.index[922:1037]
    nonfinite_rows = frame.index[1037:]
    assert sum(len(rows) for rows in (passing_rows, abstained_rows, subthreshold_rows, nonfinite_rows)) == 1152
    frame.loc[passing_rows, ["s11_first_to_second_overlap", "s11_second_to_first_overlap"]] = .08
    frame.loc[abstained_rows, "abstained"] = True
    frame.loc[subthreshold_rows, "s11_first_to_second_overlap"] = .079
    frame.loc[nonfinite_rows, "s11_second_to_first_overlap"] = float("nan")
    frame.to_parquet(path, index=False)

    passing = validate_adjacent16_final_gate(root)
    assert passing.stress_symmetric_pass_rate == pytest.approx(807 / 1152)
    assert passing.passed is True
    assert "stress_evidence" not in passing.failed_checks
    assert "stress_symmetric_pass_rate" not in passing.failed_checks

    frame.loc[passing_rows[0], "s11_first_to_second_overlap"] = .079
    frame.to_parquet(path, index=False)
    failing = validate_adjacent16_final_gate(root)
    assert failing.actual_rows == failing.expected_rows
    assert failing.stress_symmetric_pass_rate == pytest.approx(806 / 1152)
    assert failing.passed is False
    assert failing.failed_checks == ("stress_symmetric_pass_rate",)


def test_final_gate_rejects_identity_swap_manifest_mutation_without_row_loss(tmp_path: Path) -> None:
    """Catch a cardinality-only manifest validation that accepts a changed frozen identity."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)
    path = root / "manifests" / "pair_conditions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    identity = next(
        row for row in rows
        if row["track"] == "controlled" and row["corruption"] == "identity_swap" and row["severity"] == 1.0
    )
    identity["severity"] = 9.0
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    report = validate_adjacent16_final_gate(root)
    assert report.actual_rows == report.expected_rows
    assert report.passed is False
    assert "metric_row_counts" not in report.failed_checks
    assert "input_validation" in report.failed_checks


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_thirteenth_variant",
        "stress_condition_state_id",
        "stress_condition_baseline",
        "stress_state_id",
        "state_order",
    ),
)
def test_final_gate_rejects_exact_manifest_graph_mutations(
    tmp_path: Path, mutation: str
) -> None:
    """Catch unordered, partial, or key-only final-manifest validation."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)
    if mutation == "duplicate_thirteenth_variant":
        path = root / "manifests/feature_variants.json"
        variants = json.loads(path.read_text())
        variants.append(dict(variants[0]))
        path.write_text(json.dumps(variants, indent=2, sort_keys=True) + "\n")
    elif mutation in {"stress_condition_state_id", "stress_condition_baseline"}:
        path = root / "manifests/pair_conditions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        stress = next(row for row in rows if row["track"] == "stress")
        if mutation == "stress_condition_state_id":
            stress["image_state_id"] = "stress__fabricated_state"
        else:
            stress["baseline_steps"] = 3 if stress["baseline_steps"] != 3 else 2
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )
    else:
        path = root / "manifests/image_states.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "stress_state_id":
            stress = next(row for row in rows if row["track"] == "stress")
            stress["state_id"] = "stress__fabricated_state"
        else:
            rows[0], rows[1] = rows[1], rows[0]
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )

    report = validate_adjacent16_final_gate(root)

    assert report.actual_rows == report.expected_rows
    assert report.passed is False
    assert report.failed_checks == ("registry_graph",)


@pytest.mark.parametrize("track", ("controlled", "stress", "geometry_calibration"))
def test_final_gate_rejects_duplicate_metric_keys_without_row_loss(tmp_path: Path, track: str) -> None:
    """Catch a metric validator that checks only the frozen row cardinality."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"][track]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    if track == "controlled":
        hard = frame.index[(frame.corruption == "identity_swap") & (frame.severity == 1.0)][0]
        easy = frame.index[(frame.corruption == "clean") & (frame.severity == 0.0)][0]
    elif track == "stress":
        hard, easy = frame.index[-1], frame.index[0]
    else:
        hard = frame.index[frame.perturbation_kind == "relative_depth_noise"][-1]
        easy = frame.index[frame.perturbation_kind == "rotation"][0]
    frame.loc[hard] = frame.loc[easy]
    frame.to_parquet(path, index=False)

    report = validate_adjacent16_final_gate(root)
    assert report.actual_rows == report.expected_rows
    assert report.passed is False
    assert "metric_row_counts" not in report.failed_checks
    assert "metric_row_keys" in report.failed_checks


@pytest.mark.parametrize(
    ("track", "column", "replacement"),
    (
        ("stress", "pair_id", "stress__constant_pair"),
        ("geometry_calibration", "pair_id", "calibration__constant_pair"),
        ("controlled", "job_id", "feature__fabricated_job"),
        ("controlled", "design_id", "C99999999"),
        ("controlled", "image_state_id", "controlled__fabricated_state"),
        ("controlled", "resolution", 999),
    ),
)
def test_final_gate_rejects_exact_metric_identity_mutations(
    tmp_path: Path, track: str, column: str, replacement: object
) -> None:
    """Catch count-preserving metric identity or variant-binding drift."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    path = fixture["metrics"][track]
    assert isinstance(root, Path)
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], column] = replacement
    frame.to_parquet(path, index=False)

    report = validate_adjacent16_final_gate(root)

    assert report.actual_rows == report.expected_rows
    assert report.passed is False
    assert report.failed_checks == ("metric_row_keys",)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    (
        pytest.param("unknown_job_id", "registry_graph", id="unknown-job-id"),
        pytest.param("semantic_payload", "registry_graph", id="semantic-payload"),
        pytest.param("pending", "unexplained_jobs", id="pending-job"),
        pytest.param("bad_output_hash", "registry_output_hashes", id="bad-output-hash"),
    ),
)
def test_final_gate_rejects_registry_and_output_mutations(
    tmp_path: Path,
    mutation: str,
    expected_failure: str,
) -> None:
    """Catch count-preserving registry changes before they can masquerade as a frozen run."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)
    registry_path = root / "registry" / "jobs.sqlite"
    with sqlite3.connect(registry_path) as connection:
        job_id, payload_json, output_manifest_json = connection.execute(
            "SELECT job_id, payload_json, output_manifest_json FROM jobs ORDER BY job_id LIMIT 1"
        ).fetchone()
        if mutation == "unknown_job_id":
            connection.execute("UPDATE jobs SET job_id = ? WHERE job_id = ?", ("unknown__frozen_job", job_id))
        elif mutation == "semantic_payload":
            payload = json.loads(payload_json)
            payload["analysis_track"] = "semantic_tamper"
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            connection.execute(
                "UPDATE jobs SET payload_json = ?, payload_hash = ? WHERE job_id = ?",
                (canonical, hashlib.sha256(canonical.encode()).hexdigest(), job_id),
            )
        elif mutation == "pending":
            connection.execute("UPDATE jobs SET state = 'pending' WHERE job_id = ?", (job_id,))
        else:
            manifest = json.loads(output_manifest_json)
            manifest["files"][0]["sha256"] = "0" * 64
            connection.execute(
                "UPDATE jobs SET output_manifest_json = ? WHERE job_id = ?",
                (json.dumps(manifest), job_id),
            )

    report = validate_adjacent16_final_gate(root)
    assert report.actual_rows == report.expected_rows
    assert report.passed is False
    assert report.failed_checks == (expected_failure,)
    if mutation == "pending":
        assert report.unexplained_jobs == (job_id,)
    if mutation == "semantic_payload":
        assert "input_validation" not in report.failed_checks


def test_final_gate_reports_exact_relative_raw_feature_artifact_paths(tmp_path: Path) -> None:
    """Catch raw feature storage under reports, regardless of its serialized extension."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)
    names = ("embedding.npy", "raw.safetensors", "weights.pt", "checkpoint.pth", "artifact.bin")
    for name in names:
        (root / "reports" / name).write_bytes(b"raw-feature-storage\n")

    report = validate_adjacent16_final_gate(root)
    assert report.passed is False
    assert report.failed_checks == ("raw_feature_artifact",)
    assert report.raw_feature_artifacts == tuple(sorted(f"reports/{name}" for name in names))


def test_final_gate_rejects_arbitrary_npz_outside_geometry_camera_allowlist(
    tmp_path: Path,
) -> None:
    """Catch arbitrary raw arrays hidden in a generically named NPZ."""

    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    assert isinstance(root, Path)
    camera = root / "cameras/allowlist_shoe/q0/cameras.npz"
    camera.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        camera,
        extrinsics=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32)[None],
        mask=np.ones((1, 1, 1), dtype=bool),
        view_ids=np.asarray(["1"]),
        camera_groups=np.asarray(["upper"]),
    )

    allowed = validate_adjacent16_final_gate(root)
    assert allowed.passed is True
    assert allowed.raw_feature_artifacts == ()

    canonical_geometry = next(root.glob("geometry/**/depth_geometry.npz"))
    with np.load(canonical_geometry, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    disguised = root / "geometry/evil/depth_geometry.npz"
    disguised.parent.mkdir(parents=True)
    np.savez_compressed(
        disguised,
        **arrays,
        embedding=np.ones((2, 3), dtype=np.float32),
    )
    disguised_report = validate_adjacent16_final_gate(root)
    assert disguised_report.passed is False
    assert disguised_report.failed_checks == ("raw_feature_artifact",)
    assert disguised_report.raw_feature_artifacts == (
        "geometry/evil/depth_geometry.npz",
    )
    disguised.unlink()

    raw = root / "reports/raw_embedding.npz"
    np.savez_compressed(raw, embedding=np.ones((2, 3), dtype=np.float32))
    rejected = validate_adjacent16_final_gate(root)

    assert rejected.passed is False
    assert rejected.failed_checks == ("raw_feature_artifact",)
    assert rejected.raw_feature_artifacts == ("reports/raw_embedding.npz",)


def test_final_gate_rejects_another_jobs_valid_output_part(tmp_path: Path) -> None:
    fixture = _final_gate_fixture(tmp_path)
    root = fixture["root"]
    with sqlite3.connect(root / "registry" / "jobs.sqlite") as connection:
        first, second = connection.execute("SELECT job_id, output_manifest_json FROM jobs WHERE phase='feature' ORDER BY job_id LIMIT 2").fetchall()
        manifest = json.loads(second[1])
        manifest["files"][0]["path"] = json.loads(first[1])["files"][0]["path"]
        manifest["files"][0]["sha256"] = json.loads(first[1])["files"][0]["sha256"]
        connection.execute("UPDATE jobs SET output_manifest_json=? WHERE job_id=?", (json.dumps(manifest), second[0]))
    report = validate_adjacent16_final_gate(root)
    assert "registry_output_hashes" in report.failed_checks


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


def test_timed_sentinel_manifest_restores_old_tree_when_staging_rename_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _frozen_config(tmp_path)
    _register_sentinel_jobs(tmp_path)
    directory = write_timed_sentinel_manifest(tmp_path)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    real_rename = Path.rename
    def fail_staging(self: Path, target: Path) -> Path:
        if self.name.startswith(".timed_sentinel.staging."):
            raise OSError("forced staging rename failure")
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", fail_staging)
    with pytest.raises(OSError, match="forced staging"):
        write_timed_sentinel_manifest(tmp_path)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    assert not list(directory.parent.glob(".timed_sentinel.backup.*"))
    assert not list(directory.parent.glob(".timed_sentinel.staging.*"))


def test_runtime_projection_rejects_reduced_registry_before_sentinel_timing(
    tmp_path: Path,
) -> None:
    """Catch projection code that estimates from pending or zero-duration work."""

    _frozen_config(tmp_path)
    _register_sentinel_jobs(tmp_path)

    report = validate_timed_runtime_projection(tmp_path)

    assert report.passed is False
    assert report.sentinel_job_ids == ()
    assert report.failed_checks == ("runtime_graph",)


def test_runtime_projection_rejects_reduced_registry_even_with_valid_timing(
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

    assert report.passed is False
    assert report.failed_checks == ("runtime_graph",)


def test_production_runtime_fixture_persists_the_frozen_sentinel_projection(tmp_path: Path) -> None:
    """Catch a runtime projection that samples the wrong jobs, sums GPUs, or loses provenance."""

    root = tmp_path / "run"
    config = Path("configs/adjacent16_core.json").read_bytes()
    digest = hashlib.sha256(config).hexdigest()
    (root / "config").mkdir(parents=True)
    (root / "config" / "frozen_config.json").write_bytes(config)
    (root / "run_manifest.json").write_text(json.dumps({"experiment_id": "20260727_adjacent16_core_v1", "config_sha256": digest}))
    _paths, contract, design = _write_frozen_manifest_fixture(root)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, digest)
    _materialize_owned_outputs(root, registry)
    assert registry.counts_by_phase() == {"feature": 3472, "geometry": 32}

    shoe = next(iter(contract.raw["shoes"]))
    clean = next(state for state in design.controlled_states if state.shoe_id == shoe and state.quartet_id == "q0" and state.corruption == "clean")
    changed = next(state for state in design.controlled_states if state.shoe_id == shoe and state.quartet_id == "q0" and state.corruption == "translation" and state.severity == .2 and state.changed_view == "1")
    stress = next(state for state in design.stress_states if state.shoe_id == shoe and state.quartet_id == "q0")
    backbones = tuple(dict.fromkeys(variant.backbone for variant in contract.feature_variants))
    expected: list[str] = []
    classes: dict[str, str] = {}
    seconds: dict[tuple[str, str], int] = {
        ("dino_vitb16", "clean"): 1, ("dino_vitb16", "changed"): 2, ("dino_vitb16", "stress"): 3,
        ("dinov2_vitb14", "clean"): 4, ("dinov2_vitb14", "changed"): 5, ("dinov2_vitb14", "stress"): 6,
        ("dinov3_vitb16", "clean"): 7, ("dinov3_vitb16", "changed"): 8, ("dinov3_vitb16", "stress"): 9,
        ("dinov3_vitl16_pixal3d", "clean"): 10, ("dinov3_vitl16_pixal3d", "changed"): 11,
        ("dinov3_vitl16_pixal3d", "stress"): 12, ("dinov3_vitl16_pixal3d", "calibration"): 13,
    }
    for backbone in backbones:
        for name, state in (("clean", clean), ("changed", changed), ("stress", stress)):
            job_id = f"feature__{'stress' if name == 'stress' else 'controlled'}__{state.state_id}__{backbone}__user_37_grid_track"
            expected.append(job_id)
            classes[job_id] = name
    calibration = f"feature__geometry_calibration__{clean.state_id}__dinov3_vitl16_pixal3d__user_37_grid_track"
    expected.append(calibration)
    classes[calibration] = "calibration"
    assert len(expected) == len(set(expected)) == 13
    assert {name: list(classes.values()).count(name) for name in ("clean", "changed", "stress", "calibration")} == {"clean": 4, "changed": 4, "stress": 4, "calibration": 1}

    directory = write_timed_sentinel_manifest(root)
    selected = tuple(job_id for path in sorted(directory.glob("*.json")) for job_id in json.loads(path.read_text())["job_ids"])
    assert set(selected) == set(expected)
    initial_files = {path.name: path.read_bytes() for path in directory.iterdir()}
    manifest_report = root / "reports" / "timed_sentinel_manifest.json"
    initial_manifest_report = manifest_report.read_bytes()
    (directory / "stale.json").write_text("{}")
    assert write_timed_sentinel_manifest(root) == directory
    assert not (directory / "stale.json").exists()
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == initial_files
    assert manifest_report.read_bytes() == initial_manifest_report

    start = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(registry.path) as connection:
        for job_id, name in classes.items():
            payload = json.loads(connection.execute("SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0])
            duration = seconds[str(payload["backbone"]), name]
            connection.execute(
                "UPDATE jobs SET state = 'completed', attempts = 1, started_at = ?, finished_at = ? WHERE job_id = ?",
                (start, f"2026-01-01T00:00:{duration:02d}+00:00", job_id),
            )
        connection.execute(
            "UPDATE jobs SET state = 'completed', attempts = 1, started_at = ?, finished_at = ? WHERE phase = 'geometry'",
            (start, "2026-01-01T01:00:00+00:00"),
        )

    report = validate_timed_runtime_projection(root)
    expected_loads = {0: 1728, 1: 4320, 2: 6912, 3: 9712}
    assert dict(zip(contract.physical_gpus, expected_loads.values(), strict=True)) == expected_loads
    assert report.passed is True
    assert report.sentinel_job_ids == tuple(expected)
    assert report.projected_geometry_hours == 1.0
    assert report.projected_feature_hours == pytest.approx(9712 / 3600)
    assert report.projected_total_hours == pytest.approx(1.0 + 1.25 * (9712 / 3600) + 2.0)
    assert report.projected_total_hours <= 24.0

    runtime_report = root / "reports" / "runtime_projection.json"
    persisted = json.loads(runtime_report.read_text())
    assert all(persisted[name] for name in ("config_sha256", "registry_sha256", "timed_sentinel_manifest_sha256", "input_sha256", "output_sha256"))
    assert persisted["sampled_class_seconds"] == {f"{backbone}/{name}": seconds[backbone, name] for backbone, name in sorted(seconds)}
    first_bytes = runtime_report.read_bytes()
    assert validate_timed_runtime_projection(root) == report
    assert runtime_report.read_bytes() == first_bytes


def _runtime_failure_fixture(tmp_path: Path, *, l_changed_seconds: float = 2.0) -> tuple[Path, JobRegistry, tuple[str, ...]]:
    root = tmp_path / "run"
    config = Path("configs/adjacent16_core.json").read_bytes()
    digest = hashlib.sha256(config).hexdigest()
    (root / "config").mkdir(parents=True)
    (root / "config" / "frozen_config.json").write_bytes(config)
    (root / "run_manifest.json").write_text(json.dumps({"experiment_id": "20260727_adjacent16_core_v1", "config_sha256": digest}))
    _paths, _contract, design = _write_frozen_manifest_fixture(root)
    registry = JobRegistry(root / "registry" / "jobs.sqlite")
    register_adjacent16_jobs(registry, root, design, digest)
    _materialize_owned_outputs(root, registry)
    ids = tuple(job_id for path in sorted(write_timed_sentinel_manifest(root).glob("*.json")) for job_id in json.loads(path.read_text())["job_ids"])
    start = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(registry.path) as connection:
        for job_id in ids:
            payload = json.loads(connection.execute("SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0])
            duration = l_changed_seconds if payload["backbone"] == "dinov3_vitl16_pixal3d" and "translation__0.200000" in payload["state_id"] else 1.0
            minutes, seconds = divmod(duration, 60.0)
            connection.execute("UPDATE jobs SET state='completed', attempts=1, started_at=?, finished_at=? WHERE job_id=?", (start, f"2026-01-01T00:{int(minutes):02}:{seconds:06.3f}+00:00", job_id))
        connection.execute("UPDATE jobs SET state='completed', attempts=1, started_at=?, finished_at=? WHERE phase='geometry'", (start, "2026-01-01T01:00:00+00:00"))
    return root, registry, ids


@pytest.mark.parametrize(("seconds", "passed"), ((72.6, True), (72.7, False)))
def test_runtime_projection_obeys_the_exact_24_hour_boundary(tmp_path: Path, seconds: float, passed: bool) -> None:
    root, _registry, _ids = _runtime_failure_fixture(tmp_path, l_changed_seconds=seconds)
    report = validate_timed_runtime_projection(root)
    assert report.projected_total_hours == pytest.approx(23.99 if passed else 24.0188888889)
    assert report.passed is passed
    assert report.failed_checks == (() if passed else ("projected_total_hours",))
    path = root / "reports" / "runtime_projection.json"
    assert json.loads(path.read_text())["passed"] is passed


@pytest.mark.parametrize("mutation", ("zero", "negative", "naive"))
def test_runtime_projection_checks_every_geometry_duration(tmp_path: Path, mutation: str) -> None:
    root, registry, _ids = _runtime_failure_fixture(tmp_path)
    with sqlite3.connect(registry.path) as connection:
        job_id = connection.execute("SELECT job_id FROM jobs WHERE phase='geometry' ORDER BY job_id LIMIT 1").fetchone()[0]
        finish = {"zero": "2026-01-01T00:00:00+00:00", "negative": "2025-12-31T23:59:59+00:00", "naive": "2026-01-01T00:00:01"}[mutation]
        connection.execute("UPDATE jobs SET finished_at=? WHERE job_id=?", (finish, job_id))
    report = validate_timed_runtime_projection(root)
    assert report.passed is False
    assert report.failed_checks == ("runtime_validation",)


@pytest.mark.parametrize("mutation", ("failed", "pending", "zero", "negative", "naive"))
def test_runtime_projection_rejects_bad_sentinel_evidence(tmp_path: Path, mutation: str) -> None:
    root, registry, ids = _runtime_failure_fixture(tmp_path)
    with sqlite3.connect(registry.path) as connection:
        job_id = ids[1]
        if mutation in {"failed", "pending"}:
            connection.execute("UPDATE jobs SET state=? WHERE job_id=?", (mutation, job_id))
        else:
            finish = {"zero": "2026-01-01T00:00:00+00:00", "negative": "2025-12-31T23:59:59+00:00", "naive": "2026-01-01T00:00:01"}[mutation]
            connection.execute("UPDATE jobs SET finished_at=? WHERE job_id=?", (finish, job_id))
    report = validate_timed_runtime_projection(root)
    assert report.passed is False
    assert ("sentinel_job_state" if mutation in {"failed", "pending"} else "runtime_validation") in report.failed_checks


@pytest.mark.parametrize("mutation", ("missing", "ambiguous", "unclassifiable"))
def test_runtime_projection_rejects_missing_ambiguous_or_unclassifiable_sentinels(tmp_path: Path, mutation: str) -> None:
    """Catch selection evidence that no longer resolves to one frozen runtime class."""

    root, registry, ids = _runtime_failure_fixture(tmp_path)
    changed = ids[1::3]
    states = {row["state_id"]: row for row in (json.loads(line) for line in (root / "manifests" / "image_states.jsonl").read_text().splitlines())}
    with sqlite3.connect(registry.path) as connection:
        payloads = [json.loads(connection.execute("SELECT payload_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]) for job_id in changed]
        assert all(states[payload["state_id"]][field] == value for payload in payloads for field, value in (("quartet_id", "q0"), ("corruption", "translation"), ("severity", .2), ("changed_view", "1")))
        target = changed[0]
        if mutation == "missing":
            connection.execute("DELETE FROM jobs WHERE job_id=?", (target,))
        elif mutation == "ambiguous":
            payload_json = connection.execute("SELECT payload_json FROM jobs WHERE job_id=?", (target,)).fetchone()[0]
            payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
            connection.execute("INSERT INTO jobs(job_id, phase, state, payload_json, payload_hash) VALUES (?, 'feature', 'pending', ?, ?)", ("feature__ambiguous_sentinel", payload_json, payload_hash))
        else:
            payload = json.loads(connection.execute("SELECT payload_json FROM jobs WHERE job_id=?", (target,)).fetchone()[0])
            payload["job_type"] = "unclassifiable"
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            connection.execute("UPDATE jobs SET payload_json=?, payload_hash=? WHERE job_id=?", (canonical, hashlib.sha256(canonical.encode()).hexdigest(), target))
    report = validate_timed_runtime_projection(root)
    assert report.passed is False
    assert report.failed_checks == ("runtime_graph",)


def test_final_gate_reports_missing_inputs_as_failed_evidence(tmp_path: Path) -> None:
    """Catch a final gate that turns absent analysis artifacts into zero evidence."""

    _frozen_config(tmp_path)

    report = validate_adjacent16_final_gate(tmp_path)

    assert report.passed is False
    assert report.actual_rows == {"controlled": 0, "geometry_calibration": 0, "stress": 0, "total": 0}
    assert "registry_counts" in report.failed_checks
