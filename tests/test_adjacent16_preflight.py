from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import met3r.adjacent16_gate as adjacent16_gate
from met3r.adjacent16_gate import validate_adjacent16_geometry_preflight
from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import (
    Adjacent16Manifest,
    AdjacentShoeRecord,
    AdjacentViewRecord,
)
from met3r.adjacent16_design import build_adjacent16_design
from met3r.adjacent16_prepare import _write_manifests
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.known_camera import KnownCameraConfig, score_symmetric_geometry_support


def _identity_camera(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [
            [4.0, 0.0, (width - 1) / 2],
            [0.0, 4.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    return intrinsics, torch.eye(4).unsqueeze(0)


def test_geometry_support_uses_directional_target_foreground_and_symmetric_abstention() -> None:
    height = width = 4
    depth = torch.ones(1, height, width)
    first_mask = torch.ones(1, height, width, dtype=torch.bool)
    second_mask = torch.zeros(1, height, width, dtype=torch.bool)
    second_mask[0, 0, 0] = True
    intrinsics, extrinsics = _identity_camera(height, width)

    support = score_symmetric_geometry_support(
        depth,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=first_mask,
        second_mask=second_mask,
        config=KnownCameraConfig(min_overlap_fraction=0.08),
    )

    torch.testing.assert_close(
        support.first_to_second.overlap_fraction,
        torch.tensor([1 / 16]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        support.second_to_first.overlap_fraction,
        torch.tensor([1.0]),
        rtol=0,
        atol=0,
    )
    assert support.first_to_second.abstained.tolist() == [True]
    assert support.second_to_first.abstained.tolist() == [False]
    assert support.abstained.tolist() == [True]


def test_geometry_support_diagnostics_are_finite_for_valid_correspondences() -> None:
    height = width = 4
    first_depth = torch.ones(1, height, width)
    second_depth = torch.ones(1, height, width)
    second_depth[0, 0, 0] = 0.0
    mask = torch.ones(1, height, width, dtype=torch.bool)
    intrinsics, extrinsics = _identity_camera(height, width)

    support = score_symmetric_geometry_support(
        first_depth,
        second_depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        first_mask=mask,
        second_mask=mask,
        config=KnownCameraConfig(min_overlap_fraction=0.0),
    )

    for direction in (support.first_to_second, support.second_to_first):
        assert torch.isfinite(direction.relative_depth_error[direction.valid_mask]).all()
        assert torch.isfinite(direction.cycle_error_px[direction.valid_mask]).all()


def test_geometry_preflight_fails_closed_when_the_frozen_run_is_missing(
    tmp_path: Path,
) -> None:
    """Catch a gate that treats a missing preflight input as passing evidence."""

    report = validate_adjacent16_geometry_preflight(tmp_path / "missing")

    assert report.passed is False
    assert report.pair_units == 0
    assert "input_validation" in report.failed_checks


def test_preflight_pair_uses_confidence_percentile_and_the_eight_percent_pass_boundary() -> None:
    """Catch applying the 20% median gate as an individual pair threshold."""

    depth = torch.ones((2, 4, 4))
    confidence = torch.stack((
        torch.linspace(1.0, 2.0, 16).reshape(4, 4),
        torch.ones((4, 4)),
    ))
    mask = torch.stack((
        torch.ones((4, 4), dtype=torch.bool),
        torch.tensor([[False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, True, True]]),
    ))
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]] * 2
    )
    extrinsics = torch.eye(4).repeat(2, 1, 1)

    scored = adjacent16_gate._score_preflight_pair(
        depth, confidence, mask, intrinsics, extrinsics, 0, 1,
    )

    assert scored["minimum_directional_overlap"] == pytest.approx(2 / 14)
    assert scored["symmetric_pass"] is True
    assert scored["minimum_directional_overlap"] < 0.20


def _orbit_extrinsic(azimuth_deg: float) -> np.ndarray:
    radians = np.deg2rad(azimuth_deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [
            [sine, 0.0, -cosine, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [cosine, 0.0, sine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _canonical_preflight_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    config = Path("configs/adjacent16_core.json").read_bytes()
    (root / "config").mkdir(parents=True)
    (root / "config/frozen_config.json").write_bytes(config)
    config_hash = hashlib.sha256(config).hexdigest()
    (root / "run_manifest.json").write_text(json.dumps({"experiment_id": "20260727_adjacent16_core_v1", "config_sha256": config_hash}))
    raw = json.loads(config)
    contract = load_adjacent16_contract(root / "config/frozen_config.json")
    registry = JobRegistry(root / "registry/jobs.sqlite")
    views_by_q = {item["quartet_id"]: item["view_ids"] for item in raw["quartets"]}
    mask = np.ones((4, 504, 504), dtype=bool)
    pixel_y, pixel_x = np.meshgrid(
        np.arange(504, dtype=np.float32),
        np.arange(504, dtype=np.float32),
        indexing="ij",
    )
    normalized_x = (pixel_x - 251.5) / 504.0
    normalized_y = (pixel_y - 251.5) / 504.0
    spherical_depth = 1.0 / np.sqrt(
        normalized_x * normalized_x + normalized_y * normalized_y + 1.0
    )
    depth = np.broadcast_to(spherical_depth, (4, 504, 504)).copy()
    confidence = np.broadcast_to(np.linspace(0.0, 1.0, 504, dtype=np.float32), (4, 504, 504)).copy()
    shoes: list[AdjacentShoeRecord] = []
    global_positions = {
        view_id: position for position, view_id in enumerate(raw["camera_order"])
    }
    for shoe_index, (shoe, camera_dataset_id) in enumerate(raw["shoes"].items()):
        manifest_views: list[AdjacentViewRecord] = []
        for quartet, views in views_by_q.items():
            source_records: dict[str, AdjacentViewRecord] = {}
            for view in views:
                source_serial = shoe_index * 16 + len(manifest_views)
                orbit_position = global_positions[view]
                azimuth = float((7.5 * orbit_position) % 360.0)
                rgb = root / "sources" / shoe / quartet / f"{view}.png"
                source_mask = root / "sources" / shoe / quartet / f"{view}.mask.png"
                rgb.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (1080, 1080),
                    (int(view) % 255, shoe_index * 60, 127),
                ).save(rgb)
                Image.new("L", (1080, 1080), 128 + source_serial).save(source_mask)
                record = AdjacentViewRecord(
                    shoe_id=shoe,
                    camera_dataset_id=camera_dataset_id,
                    quartet_id=quartet,
                    view_id=view,
                    orbit_position=orbit_position,
                    source_path=rgb.resolve(),
                    source_mask_path=source_mask.resolve(),
                    stress_path=None,
                    stress_mask_path=None,
                    deformation_json_path=None,
                    metrics_json_path=None,
                    task_json_path=None,
                    processed_image_correlation=1.0,
                    camera_azimuth_deg=azimuth,
                    source_sha256=sha256_file(rgb),
                    source_mask_sha256=sha256_file(source_mask),
                    stress_sha256=None,
                    stress_mask_sha256=None,
                    quality=None,
                )
                manifest_views.append(record)
                source_records[view] = record
            camera_dir = root / "cameras" / shoe / quartet
            camera_dir.mkdir(parents=True)
            extrinsics = np.stack(
                [_orbit_extrinsic(source_records[view].camera_azimuth_deg) for view in views]
            )
            intrinsics = np.broadcast_to(np.array([[504.0, 0.0, 251.5], [0.0, 504.0, 251.5], [0.0, 0.0, 1.0]]), (4, 3, 3)).copy()
            np.savez_compressed(camera_dir / "cameras.npz", extrinsics=extrinsics, intrinsics=intrinsics, mask=mask, view_ids=np.asarray(views), camera_groups=np.asarray(["upper"] * 4))
            (camera_dir / "images.json").write_text(json.dumps([str(source_records[view].source_path) for view in views]))
            (camera_dir / "metadata.json").write_text(json.dumps({"camera_mode": "fixed/precomputed DA3 cameras (not calibrated GT)", "camera_provenance": "dense_camera_outputs", "camera_groups": ["upper"] * 4, "view_ids": views, "source_hashes": {view: {"rgb": source_records[view].source_sha256, "mask": source_records[view].source_mask_sha256} for view in views}}))
            state = f"controlled__{shoe}__{quartet}__clean"
            output = root / "geometry" / "calibration" / state
            output.mkdir(parents=True)
            np.savez_compressed(output / "depth_geometry.npz", depth=depth, confidence=confidence, mask=mask, extrinsics=extrinsics, intrinsics=intrinsics, meta__artifact_type=np.asarray("fixed_camera_depth"), meta__camera_provenance=np.asarray("fixed_precomputed_da3"), meta__view_ids=np.asarray(views), meta__camera_groups=np.asarray(["upper"] * 4))
            (output / "depth_geometry.json").write_text("{}")
            payload = {"job_type": "geometry_calibration", "shoe_id": shoe, "quartet_id": quartet, "config_sha256": config_hash, "camera_path": str((camera_dir / "cameras.npz").resolve()), "output_dir": str(output.resolve())}
            job_id = f"geometry__geometry_calibration__{state}"
            registry.register(JobDefinition(job_id, "geometry", payload))
            with sqlite3.connect(registry.path) as connection:
                connection.execute("UPDATE jobs SET state='completed', output_manifest_json=? WHERE job_id=?", (json.dumps({"files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in (output / "depth_geometry.npz", output / "depth_geometry.json")]}), job_id))
        shoes.append(AdjacentShoeRecord(shoe, camera_dataset_id, tuple(manifest_views)))
    manifest = Adjacent16Manifest(
        reconstruction_root=contract.reconstruction_root,
        stress_root=contract.stress_root,
        shoes=tuple(shoes),
    )
    _write_manifests(root, manifest, build_adjacent16_design(manifest, contract))
    return root


def _rewrite_npz(path: Path, update: object) -> None:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    assert callable(update)
    update(arrays)
    np.savez_compressed(path, **arrays)


def _refresh_geometry_output_manifest(
    root: Path, shoe_id: str, quartet_id: str
) -> None:
    state_id = f"controlled__{shoe_id}__{quartet_id}__clean"
    output = root / "geometry" / "calibration" / state_id
    paths = (output / "depth_geometry.npz", output / "depth_geometry.json")
    manifest = {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in paths
        ]
    }
    with sqlite3.connect(root / "registry/jobs.sqlite") as connection:
        connection.execute(
            "UPDATE jobs SET output_manifest_json=? WHERE job_id=?",
            (
                json.dumps(manifest),
                f"geometry__geometry_calibration__{state_id}",
            ),
        )


def test_geometry_preflight_accepts_canonical_task6_task7_fixture_and_is_idempotent(tmp_path: Path) -> None:
    """Catch rejecting the canonical camera mask or non-atomic preflight persistence."""

    root = _canonical_preflight_run(tmp_path)
    manifest = json.loads(
        (root / "manifests/adjacent16_manifest.json").read_text()
    )
    assert set(manifest) == {"reconstruction_root", "stress_root", "shoes"}
    assert len(manifest["shoes"]) == 4
    assert all(
        set(shoe) == {"shoe_id", "camera_dataset_id", "views"}
        and len(shoe["views"]) == 16
        for shoe in manifest["shoes"]
    )
    manifest_views = [
        view for shoe in manifest["shoes"] for view in shoe["views"]
    ]
    assert len(manifest_views) == 64
    assert all(
        set(view)
        == {
            "shoe_id",
            "camera_dataset_id",
            "quartet_id",
            "view_id",
            "orbit_position",
            "source_path",
            "source_mask_path",
            "stress_path",
            "stress_mask_path",
            "deformation_json_path",
            "metrics_json_path",
            "task_json_path",
            "processed_image_correlation",
            "camera_azimuth_deg",
            "source_sha256",
            "source_mask_sha256",
            "stress_sha256",
            "stress_mask_sha256",
            "quality",
        }
        for view in manifest_views
    )
    assert len({view["source_path"] for view in manifest_views}) == 64
    assert len({view["source_mask_path"] for view in manifest_views}) == 64
    assert len({view["source_sha256"] for view in manifest_views}) == 64
    assert len({view["source_mask_sha256"] for view in manifest_views}) == 64

    first = validate_adjacent16_geometry_preflight(root)
    table = pd.read_parquet(root / "reports/geometry_preflight.parquet")
    first_bytes = (root / "reports/geometry_preflight.parquet").read_bytes()
    report_bytes = (root / "reports/geometry_preflight.json").read_bytes()

    assert first.passed is True
    assert first.pair_units == 96
    assert first.symmetric_pass_rate == 1.0
    assert first.median_min_directional_overlap >= 0.20
    assert all(rate == 1.0 for rate in first.per_shoe_pass_rate.values())
    assert tuple(table.columns) == adjacent16_gate._PREFLIGHT_COLUMNS
    assert not table.duplicated(["shoe_id", "quartet_id", "pair_id"]).any()
    assert table["baseline_steps"].value_counts().to_dict() == {1: 48, 2: 32, 3: 16}

    second = validate_adjacent16_geometry_preflight(root)
    assert second == first
    assert (root / "reports/geometry_preflight.parquet").read_bytes() == first_bytes
    assert (root / "reports/geometry_preflight.json").read_bytes() == report_bytes

    provenance = json.loads(report_bytes)
    shoe = next(iter(json.loads(Path("configs/adjacent16_core.json").read_text())["shoes"]))
    geometry = root / "geometry" / "calibration" / f"controlled__{shoe}__q0__clean" / "depth_geometry.npz"
    with np.load(geometry) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    arrays["confidence"] = arrays["confidence"].copy()
    arrays["confidence"][0, 0, 0] = 0.123
    np.savez_compressed(geometry, **arrays)
    with sqlite3.connect(root / "registry/jobs.sqlite") as connection:
        geometry_json = geometry.with_suffix(".json")
        job_id = f"geometry__geometry_calibration__controlled__{shoe}__q0__clean"
        connection.execute("UPDATE jobs SET output_manifest_json=? WHERE job_id = ?", (json.dumps({"files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in (geometry, geometry_json)]}), job_id))
    changed = validate_adjacent16_geometry_preflight(root)
    changed_provenance = json.loads((root / "reports/geometry_preflight.json").read_text())
    assert changed.passed is True
    assert changed_provenance["input_sha256"] != provenance["input_sha256"]


@pytest.mark.parametrize(
    "mutation",
    ("missing_source_mask", "fabricated_manifest_mask_hash", "source_camera_mask"),
)
def test_geometry_preflight_binds_manifest_sources_to_camera_artifacts(
    tmp_path: Path, mutation: str
) -> None:
    """Catch accepting camera evidence detached from the frozen source manifest."""

    root = _canonical_preflight_run(tmp_path)
    manifest_path = root / "manifests/adjacent16_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["shoes"][0]["views"][0]
    shoe_id, quartet_id = record["shoe_id"], record["quartet_id"]
    camera = root / "cameras" / shoe_id / quartet_id / "cameras.npz"
    geometry = (
        root
        / "geometry"
        / "calibration"
        / f"controlled__{shoe_id}__{quartet_id}__clean"
        / "depth_geometry.npz"
    )

    if mutation == "missing_source_mask":
        Path(record["source_mask_path"]).unlink()
    elif mutation == "fabricated_manifest_mask_hash":
        record["source_mask_sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    else:
        def remove_one_foreground(arrays: dict[str, np.ndarray]) -> None:
            arrays["mask"] = arrays["mask"].copy()
            arrays["mask"][0, 0, 0] = False

        _rewrite_npz(camera, remove_one_foreground)
        _rewrite_npz(geometry, remove_one_foreground)
        _refresh_geometry_output_manifest(root, shoe_id, quartet_id)

    report = validate_adjacent16_geometry_preflight(root)

    assert report.passed is False
    assert report.pair_units == 0
    assert report.failed_checks == ("input_validation",)


@pytest.mark.parametrize(
    "mutation",
    (
        "all_one_step_labels",
        "identical_extrinsics",
        "fabricated_extrinsics",
        "wrong_artifact_type",
    ),
)
def test_geometry_preflight_binds_pose_and_pair_baselines(
    tmp_path: Path, mutation: str
) -> None:
    """Catch accepting pair labels or depth geometry detached from fixed poses."""

    root = _canonical_preflight_run(tmp_path)
    manifest = json.loads(
        (root / "manifests/adjacent16_manifest.json").read_text()
    )
    records = manifest["shoes"][0]["views"][:4]
    shoe_id, quartet_id = records[0]["shoe_id"], records[0]["quartet_id"]
    camera = root / "cameras" / shoe_id / quartet_id / "cameras.npz"
    geometry = (
        root
        / "geometry"
        / "calibration"
        / f"controlled__{shoe_id}__{quartet_id}__clean"
        / "depth_geometry.npz"
    )

    if mutation == "all_one_step_labels":
        conditions_path = root / "manifests/pair_conditions.jsonl"
        conditions = [
            json.loads(line)
            for line in conditions_path.read_text().splitlines()
            if line
        ]
        for condition in conditions:
            if (
                condition["track"] == "controlled"
                and condition["corruption"] == "clean"
            ):
                condition["baseline_steps"] = 1
        conditions_path.write_text(
            "".join(
                json.dumps(condition, sort_keys=True, separators=(",", ":")) + "\n"
                for condition in conditions
            )
        )
    elif mutation in {"identical_extrinsics", "fabricated_extrinsics"}:
        if mutation == "identical_extrinsics":
            replacement = np.broadcast_to(
                _orbit_extrinsic(records[0]["camera_azimuth_deg"]), (4, 4, 4)
            ).copy()
        else:
            replacement = np.stack(
                [
                    _orbit_extrinsic(
                        (float(record["camera_azimuth_deg"]) + 1.0) % 360.0
                    )
                    for record in records
                ]
            )

        def replace_extrinsics(arrays: dict[str, np.ndarray]) -> None:
            arrays["extrinsics"] = replacement

        _rewrite_npz(camera, replace_extrinsics)
        _rewrite_npz(geometry, replace_extrinsics)
        _refresh_geometry_output_manifest(root, shoe_id, quartet_id)
    else:
        def replace_artifact_type(arrays: dict[str, np.ndarray]) -> None:
            arrays["meta__artifact_type"] = np.asarray("dino_features")

        _rewrite_npz(geometry, replace_artifact_type)
        _refresh_geometry_output_manifest(root, shoe_id, quartet_id)

    report = validate_adjacent16_geometry_preflight(root)

    assert report.passed is False
    assert report.pair_units == 0
    assert report.failed_checks == ("input_validation",)


@pytest.mark.parametrize("mutation", ("missing", "extra", "mask", "order", "hash", "source"))
def test_geometry_preflight_rejects_canonical_camera_and_provenance_mutations(
    tmp_path: Path, mutation: str,
) -> None:
    """Catch accepting malformed Task6/Task7 inputs after a canonical preflight."""

    root = _canonical_preflight_run(tmp_path)
    shoe = next(iter(json.loads(Path("configs/adjacent16_core.json").read_text())["shoes"]))
    camera = root / "cameras" / shoe / "q0" / "cameras.npz"
    geometry = root / "geometry" / "calibration" / f"controlled__{shoe}__q0__clean" / "depth_geometry.npz"
    if mutation in {"missing", "extra", "order"}:
        with np.load(camera) as artifact:
            arrays = {name: artifact[name] for name in artifact.files}
        if mutation == "missing":
            arrays.pop("mask")
        elif mutation == "extra":
            arrays["unexpected"] = np.asarray(1)
        else:
            arrays["view_ids"] = arrays["view_ids"][::-1]
        np.savez_compressed(camera, **arrays)
    elif mutation == "mask":
        with np.load(geometry) as artifact:
            arrays = {name: artifact[name] for name in artifact.files}
        arrays["mask"] = arrays["mask"].copy()
        arrays["mask"][0, 0, 0] = False
        np.savez_compressed(geometry, **arrays)
        _refresh_geometry_output_manifest(root, shoe, "q0")
    elif mutation == "source":
        path = camera.parent / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["source_hashes"]["1"]["rgb"] = "f" * 64
        path.write_text(json.dumps(metadata))
    else:
        registry = root / "registry/jobs.sqlite"
        with sqlite3.connect(registry) as connection:
            connection.execute("UPDATE jobs SET output_manifest_json=? WHERE job_id LIKE ?", (json.dumps({"files": [{"path": str(geometry), "sha256": "0" * 64}]}), "%q0__clean"))

    report = validate_adjacent16_geometry_preflight(root)

    assert report.passed is False
    assert report.pair_units == 0
    assert report.failed_checks == ("input_validation",)


@pytest.mark.parametrize(("mode", "expected"), (
    ("shoe", "per_shoe_pair_coverage"),
    ("global", "symmetric_pass_rate"),
    ("median", "median_directional_overlap"),
))
def test_geometry_preflight_enforces_frozen_coverage_and_median_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, expected: str,
) -> None:
    """Catch conflating individual 8% passes with 80%/90%/20% gate thresholds."""

    root = _canonical_preflight_run(tmp_path)
    calls = 0

    def boundary(*_args: object) -> dict[str, float | bool]:
        nonlocal calls
        index = calls
        calls += 1
        failed = (mode == "shoe" and index < 5) or (mode == "global" and index < 10)
        minimum = 0.19 if mode == "median" else 0.30
        return {
            "first_to_second_overlap": minimum, "second_to_first_overlap": minimum,
            "minimum_directional_overlap": minimum, "overlap_asymmetry": 0.0,
            "common_valid_fraction": minimum, "cycle_error_px_median": 0.0,
            "relative_depth_error_median": 0.0, "symmetric_pass": not failed,
        }

    monkeypatch.setattr(adjacent16_gate, "_score_preflight_pair", boundary)
    report = validate_adjacent16_geometry_preflight(root)

    assert report.pair_units == 96
    assert expected in report.failed_checks
    if mode == "median":
        assert report.symmetric_pass_rate == 1.0
