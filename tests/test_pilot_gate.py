from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import met3r.pilot_gate as pilot_gate
from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.pca_features import PCABasis, pca_fingerprint
from met3r.pilot_gate import (
    aggregate_all_metric_parts,
    aggregate_metric_parts,
    validate_pilot,
)
from met3r.upper_data import EXPECTED_SHOES, UPPER_VIEWS
from met3r.upper_design import (
    FeatureVariant,
    build_upper_design,
    feature_variants,
)


BACKBONE = "dinov3_vitl16_pixal3d"
REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_surplus_cq7743_300",
)
TRACKS = ("user_37_grid_track", "current_pixal3d_code_track")


def _config() -> dict[str, object]:
    return json.loads(
        Path("configs/upper_dino_fidelity.json").read_text(encoding="utf-8")
    )


def _pilot_variants(config: dict[str, object]) -> tuple[FeatureVariant, ...]:
    return tuple(item for item in feature_variants(config) if item.backbone == BACKBONE)


def _design_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        shoes=tuple(
            SimpleNamespace(
                shoe_id=shoe_id,
                views=tuple(
                    SimpleNamespace(view_id=view_id)
                    for view_id in UPPER_VIEWS
                ),
            )
            for shoe_id in EXPECTED_SHOES
        )
    )


def _states_and_conditions(
    config: dict[str, object],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    design = build_upper_design(_design_manifest(), config, scope="pilot")
    states = [asdict(state) for state in design.image_states]
    conditions = pd.DataFrame(asdict(row) for row in design.pair_conditions)
    assert len(states) == 32
    assert len(conditions) == 44
    return states, conditions


def _metric_rows(
    conditions: pd.DataFrame,
    variants: tuple[FeatureVariant, ...],
    state: dict[str, object],
    job_id: str,
    feature_track: str,
    config_sha: str,
) -> pd.DataFrame:
    attached = conditions.loc[conditions["image_state_id"] == state["state_id"]]
    track_variants = [item for item in variants if item.feature_track == feature_track]
    rows: list[dict[str, object]] = []
    for condition in attached.to_dict("records"):
        for variant in track_variants:
            ordinal = len(rows) + 1
            rows.append(
                {
                    **condition,
                    "job_id": job_id,
                    "analysis_track": "controlled",
                    "feature_track": variant.feature_track,
                    "variant_key": variant.key,
                    "backbone": variant.backbone,
                    "backbone_revision": REVISION,
                    "input_size": variant.input_size,
                    "feature_mode": variant.feature_mode,
                    "resolution": variant.resolution,
                    "config_sha256": config_sha,
                    "camera_provenance": "fixed_precomputed_da3",
                    "camera_label": (
                        "fixed/precomputed DA3 cameras (not calibrated GT)"
                    ),
                    "abstained": False,
                    "score": 0.5 + ordinal / 10_000,
                    "s00": 0.50,
                    "s10": 0.51,
                    "s01": 0.52,
                    "s11": 0.53,
                    "feature_effect": 0.01,
                    "geometry_effect": 0.02,
                    "interaction": 0.0,
                    "s11_first_to_second_score": 0.53,
                    "s11_second_to_first_score": 0.53,
                    "s11_first_to_second_overlap": 1.0,
                    "s11_second_to_first_overlap": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _set_job_state(
    registry: JobRegistry,
    job_id: str,
    state: str,
    *,
    files: tuple[Path, ...] = (),
    error: str | None = None,
) -> None:
    manifest = (
        json.dumps(
            {
                "files": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in files
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if files
        else None
    )
    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET state = ?, output_manifest_json = ?, attempts = 1, error = ?
            WHERE job_id = ?
            """,
            (
                state,
                manifest,
                (
                    error
                    if error is not None
                    else "explicit test abstention"
                    if state == "abstained"
                    else None
                ),
                job_id,
            ),
        )


def _refresh_job_manifest(root: Path, job_id: str, part: Path) -> None:
    registry = JobRegistry(root / "registry/jobs.sqlite")
    _set_job_state(registry, job_id, "completed", files=(part,))


def _part_for_job(root: Path, job_id: str, feature_track: str) -> Path:
    return root / "metrics" / "parts" / feature_track / f"{job_id}.parquet"


@pytest.fixture
def valid_pilot_run(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    for relative in (
        "config",
        "manifests",
        "registry",
        "metrics/parts",
        f"pca/{BACKBONE}",
        "reports",
        "states",
        "pca_fit_images",
        "geometry",
        "visualizations",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = _config()
    config_path = root / "config/frozen_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (root / "run_manifest.json").write_text(
        json.dumps({"scope": "pilot", "config_sha256": config_sha}),
        encoding="utf-8",
    )
    states, conditions = _states_and_conditions(config)
    variants = _pilot_variants(config)
    assert len(variants) == 30
    (root / "manifests/image_states.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states),
        encoding="utf-8",
    )
    conditions.to_parquet(root / "manifests/pair_conditions.parquet", index=False)
    (root / "manifests/feature_variants.json").write_text(
        json.dumps([asdict(item) for item in variants]),
        encoding="utf-8",
    )
    (root / "manifests/upper_manifest.json").write_text(
        json.dumps(
            {
                "reconstruction_root": config["reconstruction_root"],
                "shoes": [
                    {
                        "shoe_id": shoe_id,
                        "views": [
                            {"view_id": view_id}
                            for view_id in UPPER_VIEWS
                        ],
                    }
                    for shoe_id in EXPECTED_SHOES
                ],
            }
        ),
        encoding="utf-8",
    )

    basis_path = root / f"pca/{BACKBONE}/basis.npz"
    incomplete = PCABasis(
        mean=np.zeros(3, dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        lower=np.zeros(3, dtype=np.float32),
        upper=np.ones(3, dtype=np.float32),
        fingerprint="",
    )
    basis = replace(incomplete, fingerprint=pca_fingerprint(incomplete))
    basis.save(basis_path)
    fit_images: list[dict[str, str]] = []
    for shoe_id in EXPECTED_SHOES:
        shoe_dir = root / "pca_fit_images" / shoe_id
        shoe_dir.mkdir()
        for view_id in UPPER_VIEWS:
            fit_image = shoe_dir / f"{view_id}.png"
            fit_image.write_bytes(f"{shoe_id}/{view_id}".encode())
            fit_images.append(
                {
                    "path": str(fit_image.resolve()),
                    "sha256": sha256_file(fit_image),
                }
            )
    (root / f"pca/{BACKBONE}/basis.json").write_text(
        json.dumps(
            {
                "artifact_type": "fixed_pca_basis",
                "backbone": BACKBONE,
                "backbone_revision": REVISION,
                "config_sha256": config_sha,
                "seed": 42,
                "fingerprint": basis.fingerprint,
                "fit_image_count": 16,
                "fit_sample_count": 32,
                "input_sizes": [512, 592],
                "fit_images": fit_images,
            }
        ),
        encoding="utf-8",
    )

    registry = JobRegistry(root / "registry/jobs.sqlite")
    for state in states:
        geometry_id = f"geometry__{state['state_id']}"
        registry.register(
            JobDefinition(
                geometry_id,
                "geometry",
                {
                    "job_type": "geometry",
                    "state_id": state["state_id"],
                    "track": "controlled",
                    "shoe_id": state["shoe_id"],
                    "config_sha256": config_sha,
                },
            )
        )
        _set_job_state(registry, geometry_id, "abstained")
        for feature_track in TRACKS:
            job_id = (
                f"feature__controlled__{state['state_id']}__"
                f"{BACKBONE}__{feature_track}"
            )
            part = _part_for_job(root, job_id, feature_track)
            part.parent.mkdir(parents=True, exist_ok=True)
            _metric_rows(
                conditions,
                variants,
                state,
                job_id,
                feature_track,
                config_sha,
            ).to_parquet(part, index=False)
            registry.register(
                JobDefinition(
                    job_id,
                    "feature",
                    {
                        "job_type": "feature",
                        "analysis_track": "controlled",
                        "state_id": state["state_id"],
                        "track": "controlled",
                        "shoe_id": state["shoe_id"],
                        "backbone": BACKBONE,
                        "feature_track": feature_track,
                        "config_sha256": config_sha,
                    },
                )
            )
            _set_job_state(registry, job_id, "completed", files=(part,))

    sentinel_id = (
        f"feature__controlled__controlled__{SHOES[0]}__clean__"
        f"{BACKBONE}__user_37_grid_track"
    )
    sentinel_part = _part_for_job(root, sentinel_id, "user_37_grid_track")
    (root / "reports/sentinel_rerun.json").write_text(
        json.dumps(
            {
                "passed": True,
                "job_id": sentinel_id,
                "backbone": BACKBONE,
                "design_ids_match": True,
                "metrics_match": True,
                "rtol": 1e-5,
                "atol": 1e-6,
                "original_part_sha256": sha256_file(sentinel_part),
                "rerun_part_sha256": sha256_file(sentinel_part),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_gate_passes_complete_operational_pilot(valid_pilot_run: Path) -> None:
    report = validate_pilot(valid_pilot_run)
    assert report.passed
    assert report.expected_metric_rows == 1320
    assert report.actual_metric_rows == 1320
    assert report.clean_directional_coverage >= 0.90
    assert report.raw_feature_artifacts == ()


def test_gate_accepts_task7_nullable_string_condition_columns(
    valid_pilot_run: Path,
) -> None:
    for part in (valid_pilot_run / "metrics/parts").rglob("*.parquet"):
        frame = pd.read_parquet(part)
        frame["changed_view"] = frame["changed_view"].astype("string")
        frame.to_parquet(part, index=False)
        _refresh_job_manifest(
            valid_pilot_run,
            str(frame["job_id"].iloc[0]),
            part,
        )
    sentinel_path = valid_pilot_run / "reports/sentinel_rerun.json"
    sentinel = json.loads(sentinel_path.read_text())
    sentinel_part = _part_for_job(
        valid_pilot_run,
        sentinel["job_id"],
        "user_37_grid_track",
    )
    sentinel["original_part_sha256"] = sha256_file(sentinel_part)
    sentinel["rerun_part_sha256"] = sha256_file(sentinel_part)
    sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")
    assert validate_pilot(valid_pilot_run).passed


def test_optional_normalization_handles_pandas_na() -> None:
    assert pilot_gate._normalized_optional(pd.NA) is None


def test_gate_rejects_missing_or_tampered_design_manifest(
    valid_pilot_run: Path,
) -> None:
    (valid_pilot_run / "manifests/feature_variants.json").unlink()
    report = validate_pilot(valid_pilot_run)
    assert not report.passed
    assert "design_manifests" in report.failed_checks


@pytest.mark.parametrize(
    "extra",
    (
        "not-an-object",
        None,
        [],
        {"wrong": "keys"},
    ),
)
def test_gate_rejects_malformed_feature_variant_list_entry(
    valid_pilot_run: Path,
    extra: object,
) -> None:
    path = valid_pilot_run / "manifests/feature_variants.json"
    variants = json.loads(path.read_text())
    variants.append(extra)
    path.write_text(json.dumps(variants), encoding="utf-8")

    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


def test_gate_rejects_extra_feature_variant_object_key(
    valid_pilot_run: Path,
) -> None:
    path = valid_pilot_run / "manifests/feature_variants.json"
    variants = json.loads(path.read_text())
    variants[0]["unexpected"] = True
    path.write_text(json.dumps(variants), encoding="utf-8")

    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        ("direction", lambda state: -int(state["direction"])),
        ("seed", lambda state: int(state["seed"]) + 1),
        ("donor_shoe_id", lambda state: SHOES[0]),
        ("donor_view", lambda state: "bottom"),
    ),
)
def test_gate_rejects_noncanonical_task3_state_provenance(
    valid_pilot_run: Path,
    field: str,
    mutate: object,
) -> None:
    path = valid_pilot_run / "manifests/image_states.jsonl"
    states = [json.loads(line) for line in path.read_text().splitlines()]
    index = next(
        index
        for index, state in enumerate(states)
        if (
            state["corruption"] == "identity_swap"
            if field.startswith("donor_")
            else state["corruption"] != "clean"
        )
    )
    states[index][field] = mutate(states[index])
    path.write_text(
        "".join(json.dumps(state) + "\n" for state in states),
        encoding="utf-8",
    )
    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


def _replace_design_id(root: Path, old: str, new: str) -> None:
    conditions_path = root / "manifests/pair_conditions.parquet"
    conditions = pd.read_parquet(conditions_path)
    conditions.loc[conditions["design_id"] == old, "design_id"] = new
    conditions.to_parquet(conditions_path, index=False)
    for part in (root / "metrics/parts").rglob("*.parquet"):
        frame = pd.read_parquet(part)
        if bool((frame["design_id"] == old).any()):
            frame.loc[frame["design_id"] == old, "design_id"] = new
            frame.to_parquet(part, index=False)
            _refresh_job_manifest(root, str(frame["job_id"].iloc[0]), part)
    sentinel_path = root / "reports/sentinel_rerun.json"
    sentinel = json.loads(sentinel_path.read_text())
    sentinel_part = _part_for_job(
        root, sentinel["job_id"], "user_37_grid_track"
    )
    sentinel["original_part_sha256"] = sha256_file(sentinel_part)
    sentinel["rerun_part_sha256"] = sha256_file(sentinel_part)
    sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")


def test_gate_rejects_noncanonical_numeric_design_id_assignment(
    valid_pilot_run: Path,
) -> None:
    _replace_design_id(valid_pilot_run, "D00000001", "D00000999")
    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


def test_gate_rejects_reordered_task3_states_and_conditions(
    valid_pilot_run: Path,
) -> None:
    states_path = valid_pilot_run / "manifests/image_states.jsonl"
    states = states_path.read_text().splitlines()
    states_path.write_text("\n".join(reversed(states)) + "\n")
    conditions_path = valid_pilot_run / "manifests/pair_conditions.parquet"
    conditions = pd.read_parquet(conditions_path)
    conditions.iloc[::-1].reset_index(drop=True).to_parquet(
        conditions_path, index=False
    )
    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


def test_gate_rejects_noncanonical_state_id_even_when_ownership_is_rewritten(
    valid_pilot_run: Path,
) -> None:
    states_path = valid_pilot_run / "manifests/image_states.jsonl"
    states = [json.loads(line) for line in states_path.read_text().splitlines()]
    state = next(item for item in states if item["corruption"] != "clean")
    old = state["state_id"]
    new = f"{old}__invented"
    state["state_id"] = new
    states_path.write_text(
        "".join(json.dumps(item) + "\n" for item in states),
        encoding="utf-8",
    )
    conditions_path = valid_pilot_run / "manifests/pair_conditions.parquet"
    conditions = pd.read_parquet(conditions_path)
    conditions.loc[conditions["image_state_id"] == old, "image_state_id"] = new
    conditions.to_parquet(conditions_path, index=False)
    registry_path = valid_pilot_run / "registry/jobs.sqlite"
    with sqlite3.connect(registry_path) as connection:
        rows = connection.execute(
            "SELECT job_id, payload_json FROM jobs"
        ).fetchall()
        for job_id, payload_json in rows:
            payload = json.loads(payload_json)
            if payload.get("state_id") != old:
                continue
            payload["state_id"] = new
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            connection.execute(
                """
                UPDATE jobs SET payload_json = ?, payload_hash = ?
                WHERE job_id = ?
                """,
                (encoded, hashlib.sha256(encoded.encode()).hexdigest(), job_id),
            )
    for part in (valid_pilot_run / "metrics/parts").rglob("*.parquet"):
        frame = pd.read_parquet(part)
        if bool((frame["image_state_id"] == old).any()):
            frame.loc[frame["image_state_id"] == old, "image_state_id"] = new
            frame.to_parquet(part, index=False)
            _refresh_job_manifest(
                valid_pilot_run, str(frame["job_id"].iloc[0]), part
            )
    assert "design_manifests" in validate_pilot(valid_pilot_run).failed_checks


def test_gate_rejects_arbitrary_design_and_variant_keys(
    valid_pilot_run: Path,
) -> None:
    part = sorted((valid_pilot_run / "metrics/parts").rglob("*.parquet"))[0]
    frame = pd.read_parquet(part)
    frame.loc[0, "design_id"] = "D99999999"
    frame.loc[1, "variant_key"] = "invented-variant"
    frame.to_parquet(part, index=False)
    job_id = str(frame.loc[0, "job_id"])
    _refresh_job_manifest(valid_pilot_run, job_id, part)
    report = validate_pilot(valid_pilot_run)
    assert "metric_primary_keys" in report.failed_checks
    assert "design_ownership" in report.failed_checks


def test_gate_rejects_cross_job_row_ownership_with_exact_global_keys(
    valid_pilot_run: Path,
) -> None:
    parts = sorted(
        (valid_pilot_run / "metrics/parts/user_37_grid_track").glob("*.parquet")
    )
    first = pd.read_parquet(parts[0])
    second_index = next(
        index
        for index, path in enumerate(parts[1:], start=1)
        if pd.read_parquet(path)["image_state_id"].iloc[0]
        != first["image_state_id"].iloc[0]
    )
    second = pd.read_parquet(parts[second_index])
    first_row = first.iloc[0].copy()
    second_row = second.iloc[0].copy()
    first.iloc[0] = second_row
    second.iloc[0] = first_row
    first.to_parquet(parts[0], index=False)
    second.to_parquet(parts[second_index], index=False)
    _refresh_job_manifest(valid_pilot_run, str(first_row["job_id"]), parts[0])
    _refresh_job_manifest(
        valid_pilot_run,
        str(second_row["job_id"]),
        parts[second_index],
    )
    report = validate_pilot(valid_pilot_run)
    assert report.actual_metric_rows == 1320
    assert "design_ownership" in report.failed_checks


def test_gate_fails_raw_features_and_storage_overage(
    valid_pilot_run: Path,
) -> None:
    (valid_pilot_run / "features.npz").write_bytes(b"forbidden")
    report = validate_pilot(
        valid_pilot_run,
        projected_bytes=301 * 1024**3,
    )
    assert "raw_feature_artifact" in report.failed_checks
    assert "storage_limit" in report.failed_checks


def test_gate_rejects_whitespace_only_abstention_reason(
    valid_pilot_run: Path,
) -> None:
    with sqlite3.connect(valid_pilot_run / "registry/jobs.sqlite") as connection:
        connection.execute(
            """
            UPDATE jobs SET error = '   '
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE state = 'abstained'
                ORDER BY job_id LIMIT 1
            )
            """
        )
    report = validate_pilot(valid_pilot_run)
    assert "job_completion" in report.failed_checks


def test_forced_sentinel_calls_worker_and_compares_new_output(
    valid_pilot_run: Path,
) -> None:
    calls: list[tuple[Path, str, str]] = []

    def runner(root: str | Path, backbone: str, device: str) -> object:
        run_root = Path(root)
        calls.append((run_root, backbone, device))
        registry = JobRegistry(run_root / "registry/jobs.sqlite")
        job = registry.claim_next(
            "feature:test", "feature", {"backbone": BACKBONE}
        )
        assert job is not None
        part = _part_for_job(
            run_root,
            job.job_id,
            str(job.payload["feature_track"]),
        )
        frame = pd.read_parquet(part)
        frame.to_parquet(part, index=False)
        registry.complete(
            job.job_id,
            {
                "files": [
                    {
                        "path": str(part.resolve()),
                        "sha256": sha256_file(part),
                    }
                ]
            },
        )
        return object()

    report = validate_pilot(
        valid_pilot_run,
        rerun_sentinel=True,
        sentinel_runner=runner,
    )
    assert report.passed
    assert calls == [(valid_pilot_run.resolve(), BACKBONE, "cuda:0")]


def test_storage_projection_splits_basis_and_per_job_pca(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def measured(path: Path) -> int:
        if path == valid_pilot_run / "pca":
            return 6_500
        return 0

    monkeypatch.setattr(pilot_gate, "_directory_bytes", measured)
    monkeypatch.setattr(pilot_gate, "_pca_basis_bytes", lambda path: 100)
    projection = pilot_gate._storage_projection(valid_pilot_run, 1320)
    categories = projection["categories"]
    assert categories["pca_basis"]["measured_bytes"] == 100
    assert categories["pca_basis"]["projected_bytes"] == 400
    assert categories["pca_job_outputs"]["measured_bytes"] == 6_400
    assert categories["pca_job_outputs"]["projected_bytes"] == 174_000


@pytest.mark.parametrize(
    ("generation_count", "generation_bytes"),
    (
        (2, 2_640),
        (0, 0),
    ),
)
def test_storage_projection_accounts_for_exact_generation_retention(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_count: int,
    generation_bytes: int,
) -> None:
    def measured(path: Path) -> int:
        if path == valid_pilot_run / "metrics/parts":
            return 1_320
        return 0

    monkeypatch.setattr(pilot_gate, "_directory_bytes", measured)
    monkeypatch.setattr(pilot_gate, "_pca_basis_bytes", lambda path: 0)
    monkeypatch.setattr(
        pilot_gate,
        "_aggregate_generation_stats",
        lambda path: (generation_count, generation_bytes),
        raising=False,
    )

    projection = pilot_gate._storage_projection(valid_pilot_run, 1_320)
    categories = projection["categories"]
    parts = categories["metric_parts"]
    generations = categories["aggregate_generations"]
    assert parts["measured_bytes"] == 1_320
    assert parts["full_units"] == pilot_gate.FULL_DESIGN_COUNTS["metric_rows"]
    assert parts["projected_bytes"] == pilot_gate.FULL_DESIGN_COUNTS["metric_rows"]
    assert generations["measured_bytes"] == generation_bytes
    assert generations["generation_count"] == generation_count
    assert generations["retention_limit"] == 2
    assert generations["projected_generation_count"] == 2
    assert generations["projected_bytes"] == (
        pilot_gate.FULL_DESIGN_COUNTS["metric_rows"] * 2
    )


def test_storage_gate_counts_both_retained_aggregate_generations(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_copy_bytes = 1_000_000_000

    def measured(path: Path) -> int:
        if path == valid_pilot_run / "metrics/parts":
            return pilot_copy_bytes
        return 0

    monkeypatch.setattr(pilot_gate, "_directory_bytes", measured)
    monkeypatch.setattr(pilot_gate, "_pca_basis_bytes", lambda path: 0)
    monkeypatch.setattr(
        pilot_gate,
        "_aggregate_generation_stats",
        lambda path: (2, pilot_copy_bytes * 2),
        raising=False,
    )

    report = validate_pilot(valid_pilot_run)
    projection = json.loads(
        (
            valid_pilot_run / "reports/pilot_storage_projection.json"
        ).read_text()
    )
    parts = projection["categories"]["metric_parts"]["projected_bytes"]
    generations = projection["categories"]["aggregate_generations"]
    one_generation = generations["projected_bytes"] // 2
    limit = 300 * 1024**3

    assert parts + one_generation <= limit
    assert parts + generations["projected_bytes"] > limit
    assert generations["generation_count"] == 2
    assert generations["retention_limit"] == 2
    assert "storage_limit" in report.failed_checks


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_type", "not-a-basis"),
        ("seed", 7),
        ("input_sizes", [592]),
        ("fit_image_count", 1),
        ("fit_sample_count", 0),
    ),
)
def test_gate_rejects_wrong_task7_pca_manifest_contract(
    valid_pilot_run: Path,
    field: str,
    value: object,
) -> None:
    path = valid_pilot_run / f"pca/{BACKBONE}/basis.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "pca_fingerprint" in validate_pilot(valid_pilot_run).failed_checks


@pytest.mark.parametrize("mode", ("empty", "arbitrary"))
def test_gate_rejects_noncanonical_task7_pca_fit_images(
    valid_pilot_run: Path,
    mode: str,
) -> None:
    path = valid_pilot_run / f"pca/{BACKBONE}/basis.json"
    manifest = json.loads(path.read_text())
    if mode == "empty":
        manifest["fit_images"] = []
    else:
        arbitrary = valid_pilot_run / "states/arbitrary.png"
        arbitrary.write_bytes(b"arbitrary")
        manifest["fit_images"] = [
            {
                "path": str(arbitrary.resolve()),
                "sha256": sha256_file(arbitrary),
            }
        ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "pca_fingerprint" in validate_pilot(valid_pilot_run).failed_checks


def test_gate_rejects_metric_part_in_wrong_feature_track_directory(
    valid_pilot_run: Path,
) -> None:
    sentinel = json.loads(
        (valid_pilot_run / "reports/sentinel_rerun.json").read_text()
    )["job_id"]
    source = next(
        path
        for path in (
            valid_pilot_run / "metrics/parts/user_37_grid_track"
        ).glob("*.parquet")
        if path.stem != sentinel
    )
    destination = (
        valid_pilot_run
        / "metrics/parts/current_pixal3d_code_track"
        / source.name
    )
    destination.parent.mkdir(exist_ok=True)
    source.rename(destination)
    _refresh_job_manifest(valid_pilot_run, source.stem, destination)
    assert "design_ownership" in validate_pilot(valid_pilot_run).failed_checks


@pytest.mark.parametrize(
    ("track", "expected_columns"),
    (
        ("stress", {"design_id", "variant_key"}),
        (
            "geometry_calibration",
            {
                "design_id",
                "variant_key",
                "changed_view",
                "perturbation_kind",
                "magnitude",
            },
        ),
    ),
)
def test_empty_aggregate_has_stable_typed_schema(
    valid_pilot_run: Path,
    track: str,
    expected_columns: set[str],
) -> None:
    output = aggregate_metric_parts(valid_pilot_run, track)
    table = pilot_gate.pq.read_table(output)
    assert expected_columns <= set(table.column_names)
    assert table.schema.field("design_id").type == pilot_gate.pa.string()
    assert table.schema.field("variant_key").type == pilot_gate.pa.string()
    if track == "geometry_calibration":
        assert table.schema.field("magnitude").type == pilot_gate.pa.float64()


def test_aggregation_rejects_duplicate_primary_keys_without_replacing_output(
    valid_pilot_run: Path,
) -> None:
    output = aggregate_metric_parts(valid_pilot_run, "controlled")
    prior = output.read_bytes()
    source = sorted((valid_pilot_run / "metrics/parts").rglob("*.parquet"))[0]
    duplicate = (
        valid_pilot_run
        / "metrics/parts/user_37_grid_track/feature__duplicate.parquet"
    )
    pd.read_parquet(source).iloc[:1].to_parquet(duplicate, index=False)
    registry = JobRegistry(valid_pilot_run / "registry/jobs.sqlite")
    registry.register(
        JobDefinition(
            "feature__duplicate",
            "feature",
            {
                "job_type": "feature",
                "analysis_track": "controlled",
                "track": "controlled",
                "state_id": "duplicate",
                "shoe_id": SHOES[0],
                "backbone": BACKBONE,
                "feature_track": "user_37_grid_track",
                "config_sha256": hashlib.sha256(
                    (valid_pilot_run / "config/frozen_config.json").read_bytes()
                ).hexdigest(),
            },
        )
    )
    _set_job_state(registry, "feature__duplicate", "completed", files=(duplicate,))
    with pytest.raises(ValueError, match="duplicate metric primary key"):
        aggregate_metric_parts(valid_pilot_run, "controlled")
    assert output.read_bytes() == prior


def _register_auxiliary_metric_job(
    root: Path,
    *,
    job_type: str,
    analysis_track: str,
    state_track: str,
    directory: str,
) -> tuple[str, Path]:
    job_id = f"feature__{job_type}__exact_path"
    part = root / "metrics/parts" / directory / f"{job_id}.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        "design_id": "D00000001",
        "variant_key": "variant",
    }
    if job_type == "geometry_calibration":
        row.update(
            changed_view="left",
            perturbation_kind="rotation",
            magnitude=1.0,
        )
    pd.DataFrame([row]).to_parquet(part, index=False)
    registry = JobRegistry(root / "registry/jobs.sqlite")
    registry.register(
        JobDefinition(
            job_id,
            "feature",
            {
                "job_type": job_type,
                "analysis_track": analysis_track,
                "track": state_track,
                "state_id": f"{job_type}-state",
                "shoe_id": SHOES[0],
                "backbone": BACKBONE,
                "feature_track": "user_37_grid_track",
                "config_sha256": "test",
            },
        )
    )
    _set_job_state(registry, job_id, "completed", files=(part,))
    return job_id, part


@pytest.mark.parametrize(
    ("job_type", "track", "expected_directory", "wrong_directory"),
    (
        ("stress", "stress", "user_37_grid_track", "real_deformation_stress"),
        (
            "geometry_calibration",
            "geometry_calibration",
            "geometry_calibration",
            "user_37_grid_track",
        ),
    ),
)
def test_aggregation_rejects_auxiliary_part_outside_task7_exact_directory(
    valid_pilot_run: Path,
    job_type: str,
    track: str,
    expected_directory: str,
    wrong_directory: str,
) -> None:
    state_track = "stress" if job_type == "stress" else "controlled"
    job_id, source = _register_auxiliary_metric_job(
        valid_pilot_run,
        job_type=job_type,
        analysis_track=track,
        state_track=state_track,
        directory=expected_directory,
    )
    destination = (
        valid_pilot_run
        / "metrics/parts"
        / wrong_directory
        / source.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    _refresh_job_manifest(valid_pilot_run, job_id, destination)

    with pytest.raises(ValueError, match="exact Task 7 path"):
        aggregate_metric_parts(valid_pilot_run, track)


def test_aggregation_rejects_job_type_analysis_track_mismatch(
    valid_pilot_run: Path,
) -> None:
    _register_auxiliary_metric_job(
        valid_pilot_run,
        job_type="stress",
        analysis_track="controlled",
        state_track="stress",
        directory="user_37_grid_track",
    )

    with pytest.raises(ValueError, match="payload routing"):
        aggregate_metric_parts(valid_pilot_run, "stress")


def _register_calibration_duplicate(root: Path) -> None:
    registry = JobRegistry(root / "registry/jobs.sqlite")
    base = {
        "design_id": "D00000001",
        "variant_key": "variant",
        "changed_view": "left",
        "perturbation_kind": "rotation",
        "magnitude": 1.0,
    }
    for index in range(2):
        job_id = f"feature__geometry_calibration__{index}"
        part = root / "metrics/parts/geometry_calibration" / f"{job_id}.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([base]).to_parquet(part, index=False)
        registry.register(
            JobDefinition(
                job_id,
                "feature",
                {
                    "job_type": "geometry_calibration",
                    "analysis_track": "geometry_calibration",
                    "track": "controlled",
                    "state_id": f"state-{index}",
                    "shoe_id": SHOES[0],
                    "backbone": BACKBONE,
                    "feature_track": "user_37_grid_track",
                    "config_sha256": "test",
                },
            )
        )
        _set_job_state(registry, job_id, "completed", files=(part,))


def test_aggregate_all_later_track_failure_preserves_entire_prior_generation(
    valid_pilot_run: Path,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    outputs = (
        valid_pilot_run / "metrics/controlled_fidelity.parquet",
        valid_pilot_run / "metrics/geometry_calibration.parquet",
        valid_pilot_run / "metrics/real_deformation_stress.parquet",
        valid_pilot_run / "metrics/summary_preview.csv",
    )
    before = {path: path.read_bytes() for path in outputs}
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    frame.loc[0, "score"] += 0.1
    frame.to_parquet(controlled_part, index=False)
    _refresh_job_manifest(
        valid_pilot_run,
        str(frame.loc[0, "job_id"]),
        controlled_part,
    )
    _register_calibration_duplicate(valid_pilot_run)

    with pytest.raises(ValueError, match="duplicate metric primary key"):
        aggregate_all_metric_parts(valid_pilot_run)

    assert {path: path.read_bytes() for path in outputs} == before
    assert not list(
        (valid_pilot_run / "metrics/aggregate_generations").glob(".staging-*")
    )


def test_aggregate_all_recovers_interrupted_install_without_mixed_generation(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    frame.loc[0, "score"] += 0.2
    frame.to_parquet(controlled_part, index=False)
    _refresh_job_manifest(
        valid_pilot_run,
        str(frame.loc[0, "job_id"]),
        controlled_part,
    )

    class SimulatedCrash(BaseException):
        pass

    old_generation = pilot_gate.resolve_aggregate_generation(valid_pilot_run)
    old_bytes = {
        path.name: path.read_bytes() for path in old_generation.iterdir()
    }
    original = pilot_gate._publish_aggregate_pointer
    monkeypatch.setattr(
        pilot_gate,
        "_publish_aggregate_pointer",
        lambda *args: (_ for _ in ()).throw(SimulatedCrash()),
    )
    with pytest.raises(SimulatedCrash):
        aggregate_all_metric_parts(valid_pilot_run)
    assert pilot_gate.resolve_aggregate_generation(valid_pilot_run) == old_generation
    assert {
        path.name: path.read_bytes() for path in old_generation.iterdir()
    } == old_bytes

    monkeypatch.setattr(pilot_gate, "_publish_aggregate_pointer", original)
    aggregate_all_metric_parts(valid_pilot_run)
    assert pilot_gate.resolve_aggregate_generation(valid_pilot_run) != old_generation
    assert not list(
        (valid_pilot_run / "metrics/aggregate_generations").glob(".staging-*")
    )
    aggregate = pd.read_parquet(
        valid_pilot_run / "metrics/controlled_fidelity.parquet"
    )
    changed = aggregate.loc[
        (aggregate["job_id"] == frame.loc[0, "job_id"])
        & (aggregate["design_id"] == frame.loc[0, "design_id"])
        & (aggregate["variant_key"] == frame.loc[0, "variant_key"]),
        "score",
    ]
    assert changed.tolist() == [frame.loc[0, "score"]]


def test_reader_resolves_one_atomic_aggregate_generation_at_pointer_switch(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    old_generation = pilot_gate.resolve_aggregate_generation(valid_pilot_run)
    old_snapshot = {
        path.name: path.read_bytes() for path in old_generation.iterdir()
    }
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    frame.loc[0, "score"] += 0.25
    frame.to_parquet(controlled_part, index=False)
    _refresh_job_manifest(
        valid_pilot_run,
        str(frame.loc[0, "job_id"]),
        controlled_part,
    )

    observed: list[Path] = []
    original = pilot_gate._publish_aggregate_pointer

    def observe(metrics_root: Path, generation: Path) -> None:
        observed.append(pilot_gate.resolve_aggregate_generation(valid_pilot_run))
        original(metrics_root, generation)
        observed.append(pilot_gate.resolve_aggregate_generation(valid_pilot_run))

    monkeypatch.setattr(pilot_gate, "_publish_aggregate_pointer", observe)
    aggregate_all_metric_parts(valid_pilot_run)

    assert observed[0] == old_generation
    assert observed[1] != old_generation
    assert {
        path.name: path.read_bytes() for path in observed[0].iterdir()
    } == old_snapshot
    assert set(path.name for path in observed[1].iterdir()) == set(old_snapshot)
    assert all(
        (valid_pilot_run / "metrics" / name).is_symlink()
        for name in old_snapshot
    )


def test_aggregate_publication_retains_only_current_and_previous_generation(
    valid_pilot_run: Path,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    snapshots = [pilot_gate.resolve_aggregate_generation(valid_pilot_run)]
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    for delta in (0.31, 0.32):
        frame.loc[0, "score"] += delta
        frame.to_parquet(controlled_part, index=False)
        _refresh_job_manifest(
            valid_pilot_run,
            str(frame.loc[0, "job_id"]),
            controlled_part,
        )
        aggregate_all_metric_parts(valid_pilot_run)
        snapshots.append(
            pilot_gate.resolve_aggregate_generation(valid_pilot_run)
        )

    generations = {
        path.resolve()
        for path in (
            valid_pilot_run / "metrics/aggregate_generations"
        ).iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert pilot_gate.AGGREGATE_GENERATION_RETENTION == 2
    assert generations == set(snapshots[-2:])
    assert not snapshots[0].exists()
    assert all(
        {
            path.name for path in generation.iterdir()
        } == set(pilot_gate._aggregate_output_names())
        for generation in snapshots[-2:]
    )
    generation_count, generation_bytes = pilot_gate._aggregate_generation_stats(
        valid_pilot_run / "metrics/aggregate_generations"
    )
    assert generation_count == 2
    assert generation_bytes == sum(
        pilot_gate._directory_bytes(generation)
        for generation in snapshots[-2:]
    )
    projection = pilot_gate._storage_projection(valid_pilot_run, 1_320)
    category = projection["categories"]["aggregate_generations"]
    assert category["generation_count"] == 2
    assert category["measured_bytes"] == generation_bytes
    assert category["retention_limit"] == 2


@pytest.mark.parametrize("second_mode", ("all", "single"))
def test_aggregate_publishers_are_serialized(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_mode: str,
) -> None:
    original = pilot_gate._aggregate_frame
    guard = threading.Lock()
    first_entered = threading.Event()
    release = threading.Event()
    concurrent = threading.Event()
    errors: list[BaseException] = []
    active = 0

    def observed(root: Path, track: str) -> pd.DataFrame:
        nonlocal active
        with guard:
            active += 1
            if active > 1:
                concurrent.set()
            first_entered.set()
        release.wait(timeout=5)
        try:
            return original(root, track)
        finally:
            with guard:
                active -= 1

    def publish(mode: str) -> None:
        try:
            if mode == "all":
                aggregate_all_metric_parts(valid_pilot_run)
            else:
                aggregate_metric_parts(valid_pilot_run, "controlled")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(pilot_gate, "_aggregate_frame", observed)
    first = threading.Thread(target=publish, args=("all",))
    second = threading.Thread(target=publish, args=(second_mode,))
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert not concurrent.wait(timeout=0.25)
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []


def test_single_track_aggregation_preserves_published_generation_aliases(
    valid_pilot_run: Path,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    old_generation = pilot_gate.resolve_aggregate_generation(valid_pilot_run)
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    frame.loc[0, "score"] += 0.275
    frame.to_parquet(controlled_part, index=False)
    _refresh_job_manifest(
        valid_pilot_run,
        str(frame.loc[0, "job_id"]),
        controlled_part,
    )

    output = aggregate_metric_parts(valid_pilot_run, "controlled")

    assert output == valid_pilot_run / "metrics/controlled_fidelity.parquet"
    assert pilot_gate.resolve_aggregate_generation(valid_pilot_run) != old_generation
    assert all(
        (valid_pilot_run / "metrics" / name).is_symlink()
        for name in pilot_gate._aggregate_output_names()
    )
    changed = pd.read_parquet(output).loc[
        lambda rows: (
            (rows["job_id"] == frame.loc[0, "job_id"])
            & (rows["design_id"] == frame.loc[0, "design_id"])
            & (rows["variant_key"] == frame.loc[0, "variant_key"])
        ),
        "score",
    ]
    assert changed.tolist() == [frame.loc[0, "score"]]


def test_aggregate_all_preview_failure_preserves_entire_prior_generation(
    valid_pilot_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate_all_metric_parts(valid_pilot_run)
    outputs = (
        valid_pilot_run / "metrics/controlled_fidelity.parquet",
        valid_pilot_run / "metrics/geometry_calibration.parquet",
        valid_pilot_run / "metrics/real_deformation_stress.parquet",
        valid_pilot_run / "metrics/summary_preview.csv",
    )
    before = {path: path.read_bytes() for path in outputs}
    controlled_part = sorted(
        (valid_pilot_run / "metrics/parts").rglob("*.parquet")
    )[0]
    frame = pd.read_parquet(controlled_part)
    frame.loc[0, "score"] += 0.3
    frame.to_parquet(controlled_part, index=False)
    _refresh_job_manifest(
        valid_pilot_run,
        str(frame.loc[0, "job_id"]),
        controlled_part,
    )

    monkeypatch.setattr(
        pilot_gate,
        "_write_aggregate_preview",
        lambda *args: (_ for _ in ()).throw(OSError("preview failed")),
    )
    with pytest.raises(OSError, match="preview failed"):
        aggregate_all_metric_parts(valid_pilot_run)

    assert {path: path.read_bytes() for path in outputs} == before
    assert not list(
        (valid_pilot_run / "metrics/aggregate_generations").glob(".staging-*")
    )
