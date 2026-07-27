from __future__ import annotations

import hashlib
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
from met3r.adjacent16_gate import validate_adjacent16_final_gate


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
    assert all(provenance[name] for name in ("config_sha256", "registry_sha256", "manifest_sha256", "metric_input_sha256", "output_sha256"))
    assert validate_adjacent16_final_gate(root) == report
    assert report_path.read_bytes() == first_bytes
    controlled_path = fixture["metrics"]["controlled"]
    controlled = pd.read_parquet(controlled_path)
    controlled.loc[0, "score"] = .81
    controlled.to_parquet(controlled_path, index=False)
    changed = validate_adjacent16_final_gate(root)
    changed_provenance = json.loads(report_path.read_text())
    assert changed.passed is True
    assert changed_provenance["metric_input_sha256"] != provenance["metric_input_sha256"]


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
    assert report.failed_checks == ("runtime_validation",)


def test_final_gate_reports_missing_inputs_as_failed_evidence(tmp_path: Path) -> None:
    """Catch a final gate that turns absent analysis artifacts into zero evidence."""

    _frozen_config(tmp_path)

    report = validate_adjacent16_final_gate(tmp_path)

    assert report.passed is False
    assert report.actual_rows == {"controlled": 0, "geometry_calibration": 0, "stress": 0, "total": 0}
    assert "registry_counts" in report.failed_checks
