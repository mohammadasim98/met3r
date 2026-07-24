from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

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
from met3r.upper_design import FeatureVariant, feature_variants


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


def _states_and_conditions() -> tuple[list[dict[str, object]], pd.DataFrame]:
    states: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    design_index = 1
    selected = {
        "translation": (0.12, 0.20),
        "exposure": (0.50, 0.75),
        "identity_swap": (1.00,),
    }
    for shoe in SHOES:
        clean_id = f"controlled__{shoe}__clean"
        states.append(
            {
                "state_id": clean_id,
                "track": "controlled",
                "shoe_id": shoe,
                "changed_view": None,
                "corruption": "clean",
                "severity": 0.0,
                "direction": 1,
                "seed": 0,
            }
        )
        state_specs = [(clean_id, None, "clean", 0.0)]
        for corruption, severities in selected.items():
            for severity in severities:
                for view in ("left", "back", "right"):
                    state_id = (
                        f"controlled__{shoe}__{view}__{corruption}__"
                        f"{severity:.6f}__+1"
                    )
                    states.append(
                        {
                            "state_id": state_id,
                            "track": "controlled",
                            "shoe_id": shoe,
                            "changed_view": view,
                            "corruption": corruption,
                            "severity": severity,
                            "direction": 1,
                            "seed": 42,
                        }
                    )
                    state_specs.append((state_id, view, corruption, severity))
        for state_id, changed_view, corruption, severity in state_specs:
            pairs = []
            if changed_view in {None, "left", "back"}:
                pairs.append(("left", "back"))
            if changed_view in {None, "left", "right"}:
                pairs.append(("left", "right"))
            for first_view, second_view in pairs:
                conditions.append(
                    {
                        "design_id": f"D{design_index:08d}",
                        "track": "controlled",
                        "shoe_id": shoe,
                        "pair_id": f"{first_view}__{second_view}",
                        "first_view": first_view,
                        "second_view": second_view,
                        "changed_view": changed_view,
                        "image_state_id": state_id,
                        "corruption": corruption,
                        "severity": severity,
                    }
                )
                design_index += 1
    assert len(states) == 32
    assert len(conditions) == 44
    return states, pd.DataFrame(conditions)


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
    states, conditions = _states_and_conditions()
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
    fit_image = root / "states/input.png"
    fit_image.write_bytes(b"input")
    (root / f"pca/{BACKBONE}/basis.json").write_text(
        json.dumps(
            {
                "artifact_type": "fixed_pca_basis",
                "backbone": BACKBONE,
                "backbone_revision": REVISION,
                "config_sha256": config_sha,
                "fingerprint": basis.fingerprint,
                "fit_images": [
                    {
                        "path": str(fit_image.resolve()),
                        "sha256": sha256_file(fit_image),
                    }
                ],
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
    duplicate = source.parent / "duplicate.parquet"
    pd.read_parquet(source).iloc[:1].to_parquet(duplicate, index=False)
    registry = JobRegistry(valid_pilot_run / "registry/jobs.sqlite")
    registry.register(
        JobDefinition(
            "feature__duplicate",
            "feature",
            {
                "job_type": "feature",
                "analysis_track": "controlled",
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
    assert not (valid_pilot_run / "metrics/.aggregate-transaction").exists()


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

    original = pilot_gate._rename_aggregate_path
    calls = 0

    def crash(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise SimulatedCrash()
        original(source, destination)

    monkeypatch.setattr(pilot_gate, "_rename_aggregate_path", crash)
    with pytest.raises(SimulatedCrash):
        aggregate_all_metric_parts(valid_pilot_run)
    assert (valid_pilot_run / "metrics/.aggregate-transaction").is_dir()

    monkeypatch.setattr(pilot_gate, "_rename_aggregate_path", original)
    aggregate_all_metric_parts(valid_pilot_run)
    assert not (valid_pilot_run / "metrics/.aggregate-transaction").exists()
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
    assert not (valid_pilot_run / "metrics/.aggregate-transaction").exists()
