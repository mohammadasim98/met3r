from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from met3r.job_registry import JobDefinition, JobRegistry, sha256_file
from met3r.pca_features import PCABasis, pca_fingerprint
from met3r.pilot_gate import aggregate_metric_parts, validate_pilot


BACKBONE = "dinov3_vitl16_pixal3d"
REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_surplus_cq7743_300",
)


def _config() -> dict[str, object]:
    return {
        "camera_label": "fixed/precomputed DA3 cameras (not calibrated GT)",
        "views": ["left", "back", "right", "front"],
        "seed": 42,
        "stage_size": 504,
        "metric": {
            "relative_depth_tolerance": 0.03,
            "min_overlap_fraction": 0.10,
            "confidence_percentile": 10.0,
        },
        "persistent_storage_limit_gb": 300,
        "physical_gpus": [4, 5, 6, 7],
        "pilot": {
            "shoes": list(SHOES),
            "pairs": ["left__back", "left__right"],
            "backbones": [BACKBONE],
        },
        "models": {
            BACKBONE: {"revision": REVISION},
            "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
            "da3_revision": "8615eefb62f2db4f8d6ebaa59160086981672829",
            "naf_revision": "37f2dfc180f2de53d98bd601109c0da0dd6b0f43",
        },
    }


def _metric_frame(rows: int = 1320) -> pd.DataFrame:
    indexes = np.arange(rows)
    clean = indexes < 60
    return pd.DataFrame(
        {
            "design_id": [f"D{index // 30 + 1:08d}" for index in indexes],
            "variant_key": [f"variant-{index % 30:02d}" for index in indexes],
            "job_id": ["feature__sentinel"] * rows,
            "analysis_track": ["controlled"] * rows,
            "track": ["controlled"] * rows,
            "shoe_id": [SHOES[index % 2] for index in indexes],
            "pair_id": [
                "left__back" if index % 2 == 0 else "left__right"
                for index in indexes
            ],
            "first_view": ["left"] * rows,
            "second_view": [
                "back" if index % 2 == 0 else "right" for index in indexes
            ],
            "changed_view": [None] * rows,
            "corruption": np.where(clean, "clean", "translation"),
            "severity": np.where(clean, 0.0, 0.12),
            "feature_track": [
                "user_37_grid_track" if index % 2 == 0
                else "current_pixal3d_code_track"
                for index in indexes
            ],
            "backbone": [BACKBONE] * rows,
            "backbone_revision": [REVISION] * rows,
            "config_sha256": ["CONFIG_HASH"] * rows,
            "camera_provenance": ["fixed_precomputed_da3"] * rows,
            "camera_label": [
                "fixed/precomputed DA3 cameras (not calibrated GT)"
            ]
            * rows,
            "abstained": [False] * rows,
            "score": np.linspace(0.1, 0.9, rows),
            "s00": np.linspace(0.1, 0.9, rows),
            "s10": np.linspace(0.11, 0.91, rows),
            "s01": np.linspace(0.12, 0.92, rows),
            "s11": np.linspace(0.13, 0.93, rows),
            "feature_effect": np.zeros(rows),
            "geometry_effect": np.zeros(rows),
            "interaction": np.zeros(rows),
            "s11_first_to_second_score": np.linspace(0.1, 0.9, rows),
            "s11_second_to_first_score": np.linspace(0.1, 0.9, rows),
            "s11_first_to_second_overlap": np.ones(rows),
            "s11_second_to_first_overlap": np.ones(rows),
        }
    )


def _set_job_state(
    registry: JobRegistry,
    job_id: str,
    state: str,
    *,
    files: tuple[Path, ...] = (),
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
            SET state = ?, output_manifest_json = ?, attempts = 1,
                error = ?
            WHERE job_id = ?
            """,
            (
                state,
                manifest,
                "explicit test abstention" if state == "abstained" else None,
                job_id,
            ),
        )


@pytest.fixture
def valid_pilot_run(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    for relative in (
        "config",
        "manifests",
        "registry",
        "metrics/parts/user_37_grid_track",
        "pca/dinov3_vitl16_pixal3d",
        "reports",
        "states",
        "geometry",
        "visualizations",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config_path = root / "config/frozen_config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (root / "run_manifest.json").write_text(
        json.dumps({"scope": "pilot", "config_sha256": config_sha}),
        encoding="utf-8",
    )
    frame = _metric_frame()
    frame["config_sha256"] = config_sha
    part = (
        root
        / "metrics/parts/user_37_grid_track"
        / "feature__sentinel.parquet"
    )
    frame.to_parquet(part, index=False)
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
    for index in range(32):
        job_id = f"geometry__{index:02d}"
        registry.register(
            JobDefinition(
                job_id,
                "geometry",
                {
                    "job_type": "geometry",
                    "shoe_id": SHOES[index % 2],
                    "config_sha256": config_sha,
                },
            )
        )
        _set_job_state(registry, job_id, "abstained")
    for index in range(64):
        job_id = "feature__sentinel" if index == 0 else f"feature__{index:02d}"
        registry.register(
            JobDefinition(
                job_id,
                "feature",
                {
                    "job_type": "feature",
                    "analysis_track": "controlled",
                    "shoe_id": SHOES[(index // 2) % 2],
                    "backbone": BACKBONE,
                    "feature_track": (
                        "user_37_grid_track"
                        if index % 2 == 0
                        else "current_pixal3d_code_track"
                    ),
                    "config_sha256": config_sha,
                },
            )
        )
        _set_job_state(
            registry,
            job_id,
            "completed" if index == 0 else "abstained",
            files=(part,) if index == 0 else (),
        )
    sentinel = {
        "passed": True,
        "job_id": "feature__sentinel",
        "backbone": BACKBONE,
        "design_ids_match": True,
        "metrics_match": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "original_part_sha256": sha256_file(part),
        "rerun_part_sha256": sha256_file(part),
    }
    (root / "reports/sentinel_rerun.json").write_text(
        json.dumps(sentinel), encoding="utf-8"
    )
    return root


def test_gate_passes_complete_operational_pilot(valid_pilot_run: Path) -> None:
    report = validate_pilot(valid_pilot_run)
    assert report.passed
    assert report.expected_metric_rows == 1320
    assert report.actual_metric_rows == 1320
    assert report.clean_directional_coverage >= 0.90
    assert report.raw_feature_artifacts == ()
    projection = json.loads(
        (valid_pilot_run / "reports/pilot_storage_projection.json").read_text()
    )
    assert projection["method"] == "artifact_category_full_design_projection"
    assert {"run_metadata", "states", "geometry", "metrics", "visualizations"} <= set(
        projection["categories"]
    )


def test_gate_fails_raw_features_and_storage_overage(
    valid_pilot_run: Path,
) -> None:
    (valid_pilot_run / "features.npz").write_bytes(b"forbidden")
    report = validate_pilot(
        valid_pilot_run,
        projected_bytes=301 * 1024**3,
    )
    assert not report.passed
    assert "raw_feature_artifact" in report.failed_checks
    assert "storage_limit" in report.failed_checks


def test_gate_rejects_nonfinite_nonabstained_metric(
    valid_pilot_run: Path,
) -> None:
    part = next((valid_pilot_run / "metrics/parts").rglob("*.parquet"))
    frame = pd.read_parquet(part)
    frame.loc[0, "score"] = np.inf
    frame.to_parquet(part, index=False)
    report = validate_pilot(valid_pilot_run)
    assert "registry_hashes" in report.failed_checks
    assert "finite_metrics" in report.failed_checks


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
        part = next((run_root / "metrics/parts").rglob("*.parquet"))
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


def test_aggregation_rejects_duplicate_primary_keys_without_replacing_output(
    valid_pilot_run: Path,
) -> None:
    output = aggregate_metric_parts(valid_pilot_run, "controlled")
    prior = output.read_bytes()
    source = next((valid_pilot_run / "metrics/parts").rglob("*.parquet"))
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
