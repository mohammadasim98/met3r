from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import met3r.adjacent16_visualization as adjacent16_visualization
from met3r.adjacent16_visualization import (
    build_adjacent16_audit_manifest,
    render_adjacent16_audit,
)
from met3r.camera_metadata import CAMERA_MODE_LABEL
from met3r.features import NAF_REVISION


LOW_VARIANT = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__bilinear__37"
)
HIGH_VARIANT = (
    "user_37_grid_track__dinov3_vitl16_pixal3d__592__naf_normal_guide__592"
)


def _numeric_row(
    design_id: str,
    *,
    track: str,
    baseline_steps: int,
    corruption: str,
    severity: float,
    variant_key: str,
    overlap: float,
    effect: float,
    valid: bool = True,
) -> dict[str, object]:
    high = variant_key == HIGH_VARIANT
    view = str(baseline_steps + 1)
    return {
        "design_id": design_id,
        "track": track,
        "shoe_id": f"shoe-{design_id[1]}",
        "quartet_id": "q0",
        "pair_id": f"shoe-{design_id[1]}__q0__1__{view}",
        "first_view": "1",
        "second_view": view,
        "baseline_steps": baseline_steps,
        "angular_separation_deg": 7.5 * baseline_steps,
        "changed_view": None if corruption in {"clean", "deformed"} else "1",
        "image_state_id": f"{track}__{design_id}",
        "corruption": corruption,
        "severity": severity,
        "variant_key": variant_key,
        "feature_track": "user_37_grid_track",
        "backbone": "dinov3_vitl16_pixal3d",
        "backbone_revision": "fixture-model-revision",
        "input_size": 592,
        "feature_mode": "naf_normal_guide" if high else "bilinear",
        "resolution": 592 if high else 37,
        "s00": 0.1,
        "s10": 0.1 + effect if valid else float("nan"),
        "s01": 0.1,
        "s11": 0.2 if valid else float("nan"),
        "score": 0.2 if valid else float("nan"),
        "feature_effect": effect if valid else float("nan"),
        "overlap": overlap if valid else float("nan"),
        "s11_first_to_second_overlap": overlap if valid else float("nan"),
        "s11_second_to_first_overlap": overlap if valid else float("nan"),
        "abstained": not valid,
    }


def _pair(
    design_id: str,
    *,
    track: str = "controlled",
    baseline_steps: int = 1,
    corruption: str = "clean",
    severity: float = 0.0,
    low_overlap: float = 0.5,
    high_overlap: float | None = None,
    low_effect: float = 0.1,
    high_effect: float = 0.2,
    low_valid: bool = True,
    high_valid: bool = True,
) -> list[dict[str, object]]:
    return [
        _numeric_row(
            design_id,
            track=track,
            baseline_steps=baseline_steps,
            corruption=corruption,
            severity=severity,
            variant_key=LOW_VARIANT,
            overlap=low_overlap,
            effect=low_effect,
            valid=low_valid,
        ),
        _numeric_row(
            design_id,
            track=track,
            baseline_steps=baseline_steps,
            corruption=corruption,
            severity=severity,
            variant_key=HIGH_VARIANT,
            overlap=low_overlap if high_overlap is None else high_overlap,
            effect=high_effect,
            valid=high_valid,
        ),
    ]


def _write_numeric_generation(root: Path) -> Path:
    generation = root / "metrics/aggregate_generations/successful-generation"
    generation.mkdir(parents=True)
    controlled: list[dict[str, object]] = []
    for design_id, steps, overlap in (
        ("C099", 1, 0.10),
        ("C100", 1, 0.10),
        ("C101", 1, 0.40),
        ("C102", 1, 0.90),
        ("C103", 1, 0.90),
        ("C200", 2, 0.20),
        ("C201", 2, 0.50),
        ("C202", 2, 0.80),
        ("C300", 3, 0.30),
        ("C301", 3, 0.60),
        ("C302", 3, 0.70),
    ):
        controlled.extend(
            _pair(
                design_id,
                baseline_steps=steps,
                low_overlap=overlap + 0.05,
                high_overlap=overlap,
            )
        )
    for design_id, corruption, severity, low, high, high_valid in (
        ("C400", "translation", 0.10, 0.10, 0.20, True),
        ("C410", "translation", 0.20, 0.10, 0.15, True),
        ("C411", "translation", 0.20, 0.10, 0.50, True),
        ("C412", "translation", 0.20, 0.10, 9.00, False),
        ("C510", "thin_plate_warp", 1.00, 0.10, 0.20, True),
        ("C511", "thin_plate_warp", 1.00, 0.10, 8.00, False),
        ("C610", "jpeg", 1.00, 0.10, 0.20, True),
        ("C611", "jpeg", 1.00, 0.10, 0.90, True),
        ("C710", "identity_swap", 1.00, 0.10, 0.20, True),
        ("C711", "identity_swap", 1.00, 0.10, 0.70, True),
    ):
        controlled.extend(
            _pair(
                design_id,
                corruption=corruption,
                severity=severity,
                low_effect=low,
                high_effect=high,
                high_valid=high_valid,
            )
        )

    stress: list[dict[str, object]] = []
    for design_id, steps, low, high, high_valid in (
        ("S100", 1, 0.10, 0.30, True),
        ("S101", 1, 0.10, 0.90, True),
        ("S102", 1, 0.10, 9.00, False),
        ("S200", 2, 0.10, 0.70, True),
        ("S300", 3, 0.10, 0.40, True),
        ("S301", 3, 0.10, 9.00, False),
    ):
        stress.extend(
            _pair(
                design_id,
                track="stress",
                baseline_steps=steps,
                corruption="deformed",
                severity=1.0,
                low_effect=low,
                high_effect=high,
                high_valid=high_valid,
            )
        )

    pd.DataFrame(controlled).to_parquet(
        generation / "controlled_fidelity.parquet",
        index=False,
    )
    pd.DataFrame(
        columns=("design_id", "variant_key"),
    ).to_parquet(generation / "geometry_calibration.parquet", index=False)
    pd.DataFrame(stress).to_parquet(
        generation / "real_deformation_stress.parquet",
        index=False,
    )
    (generation / "summary_preview.csv").write_text(
        "analysis_track\n",
        encoding="utf-8",
    )
    (root / "metrics/current").symlink_to(
        Path("aggregate_generations") / generation.name,
        target_is_directory=True,
    )
    return generation


def _role_designs(manifest: dict[str, object]) -> list[tuple[str, object, str]]:
    cases = manifest["cases"]
    assert isinstance(cases, list)
    return [
        (str(case["role"]), case["stratum"], str(case["design_id"]))
        for case in cases
    ]


def test_audit_selection_matches_hand_derived_numeric_cases(tmp_path: Path) -> None:
    """Catch extrema, median, validity, severity, or fallback selection drift."""

    _write_numeric_generation(tmp_path)

    manifest_path = build_adjacent16_audit_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert _role_designs(manifest) == [
        ("clean_overlap_minimum", 1, "C099"),
        ("clean_overlap_median_nearest", 1, "C101"),
        ("clean_overlap_maximum", 1, "C102"),
        ("clean_overlap_minimum", 2, "C200"),
        ("clean_overlap_median_nearest", 2, "C201"),
        ("clean_overlap_maximum", 2, "C202"),
        ("clean_overlap_minimum", 3, "C300"),
        ("clean_overlap_median_nearest", 3, "C301"),
        ("clean_overlap_maximum", 3, "C302"),
        ("maximum_corruption_jointly_valid", "identity_swap", "C710"),
        (
            "maximum_corruption_largest_finite_disagreement",
            "identity_swap",
            "C711",
        ),
        ("maximum_corruption_jointly_valid", "jpeg", "C610"),
        ("maximum_corruption_largest_finite_disagreement", "jpeg", "C611"),
        ("maximum_corruption_jointly_valid", "thin_plate_warp", "C510"),
        (
            "maximum_corruption_low_valid_high_invalid",
            "thin_plate_warp",
            "C511",
        ),
        ("maximum_corruption_jointly_valid", "translation", "C410"),
        (
            "maximum_corruption_low_valid_high_invalid",
            "translation",
            "C412",
        ),
        ("stress_largest_finite_disagreement", 1, "S101"),
        ("stress_low_valid_high_invalid", 1, "S102"),
        ("stress_largest_finite_disagreement", 2, "S200"),
        ("stress_largest_finite_disagreement", 3, "S300"),
        ("stress_low_valid_high_invalid", 3, "S301"),
    ]
    assert manifest["aggregate_generation"] == (
        "metrics/aggregate_generations/successful-generation"
    )
    assert manifest["variant_keys"] == {
        "low": LOW_VARIANT,
        "high": HIGH_VARIANT,
    }


def test_audit_selection_is_byte_stable_under_aggregate_row_shuffle(
    tmp_path: Path,
) -> None:
    """Catch selector outcomes or manifest order depending on Parquet row order."""

    generation = _write_numeric_generation(tmp_path)
    first = build_adjacent16_audit_manifest(tmp_path).read_bytes()
    for name, random_state in (
        ("controlled_fidelity.parquet", 91),
        ("real_deformation_stress.parquet", 17),
    ):
        path = generation / name
        frame = pd.read_parquet(path)
        frame.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        ).to_parquet(path, index=False)

    second = build_adjacent16_audit_manifest(tmp_path).read_bytes()

    assert second == first


class _TinyBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(images, (37, 37))
        fourth = pooled.mean(dim=1, keepdim=True)
        return torch.cat((pooled, fourth), dim=1)


class _TinyNAF(nn.Module):
    def forward(
        self,
        guide: torch.Tensor,
        native: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        guide_value = F.adaptive_avg_pool2d(
            guide.mean(dim=1, keepdim=True),
            size,
        )
        return F.interpolate(
            native,
            size=size,
            mode="bilinear",
            align_corners=False,
        ) + guide_value


class _WrongShapeNAF(nn.Module):
    def forward(
        self,
        guide: torch.Tensor,
        native: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        del guide, size
        return F.interpolate(native, size=(8, 8), mode="nearest")


def _write_state(
    root: Path,
    *,
    state_id: str,
    track: str,
    corruption: str,
    color: int,
) -> dict[str, object]:
    view_ids = ["1", "2", "3", "4"]
    state = {
        "state_id": state_id,
        "track": track,
        "shoe_id": "shoe",
        "quartet_id": "q0",
        "view_ids": view_ids,
        "changed_view": "1" if corruption == "jpeg" else None,
        "corruption": corruption,
        "severity": 1.0 if corruption != "clean" else 0.0,
        "direction": 1,
        "seed": 0,
        "donor_shoe_id": None,
        "donor_view": None,
    }
    directory = root / "states" / state_id
    (directory / "images").mkdir(parents=True)
    (directory / "masks").mkdir()
    for index, view_id in enumerate(view_ids):
        value = np.full((504, 504, 3), color + index, dtype=np.uint8)
        Image.fromarray(value, mode="RGB").save(
            directory / "images" / f"{view_id}.png"
        )
        Image.fromarray(
            np.full((504, 504), 255, dtype=np.uint8),
            mode="L",
        ).save(directory / "masks" / f"{view_id}.png")
    metadata = {
        "state": state,
        "stage_size": 504,
        "source_hashes": {},
        "view_ids": view_ids,
    }
    (directory / "state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state


def _write_geometry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view_ids = np.asarray(("1", "2", "3", "4"))
    np.savez_compressed(
        path,
        depth=np.ones((4, 504, 504), dtype=np.float32),
        mask=np.ones((4, 504, 504), dtype=bool),
        intrinsics=np.broadcast_to(
            np.asarray(
                ((400.0, 0.0, 251.5), (0.0, 400.0, 251.5), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            ),
            (4, 3, 3),
        ).copy(),
        extrinsics=np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (4, 4, 4),
        ).copy(),
        meta__view_ids=view_ids,
        meta__camera_groups=np.asarray(("upper",) * 4),
        meta__camera_provenance=np.asarray("fixed_precomputed_da3"),
    )
    path.with_name("depth_geometry.json").write_text(
        json.dumps(
            {
                "camera_mode": CAMERA_MODE_LABEL,
                "camera_provenance": "fixed_precomputed_da3",
                "process_res": 504,
                "view_ids": view_ids.tolist(),
                "camera_groups": ["upper"] * 4,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_render_generation(root: Path) -> None:
    generation = root / "metrics/aggregate_generations/render-generation"
    generation.mkdir(parents=True)
    controlled: list[dict[str, object]] = []
    for design_id, state_id, corruption, low_effect, high_effect in (
        ("C100", "controlled__shoe__q0__clean", "clean", 0.1, 0.2),
        ("C200", "controlled__shoe__q0__jpeg", "jpeg", 0.1, 0.7),
    ):
        pair = _pair(
            design_id,
            corruption=corruption,
            severity=0.0 if corruption == "clean" else 1.0,
            low_effect=low_effect,
            high_effect=high_effect,
        )
        for row in pair:
            row.update(
                {
                    "shoe_id": "shoe",
                    "pair_id": "shoe__q0__1__2",
                    "image_state_id": state_id,
                }
            )
        controlled.extend(pair)
    stress = _pair(
        "S100",
        track="stress",
        corruption="deformed",
        severity=1.0,
        low_effect=0.1,
        high_effect=0.5,
    )
    for row in stress:
        row.update(
            {
                "shoe_id": "shoe",
                "pair_id": "shoe__q0__1__2",
                "image_state_id": "stress__shoe__q0__deformed",
            }
        )
    pd.DataFrame(controlled).to_parquet(
        generation / "controlled_fidelity.parquet",
        index=False,
    )
    pd.DataFrame(columns=("design_id", "variant_key")).to_parquet(
        generation / "geometry_calibration.parquet",
        index=False,
    )
    pd.DataFrame(stress).to_parquet(
        generation / "real_deformation_stress.parquet",
        index=False,
    )
    (generation / "summary_preview.csv").write_text(
        "analysis_track\n",
        encoding="utf-8",
    )
    (root / "metrics/current").symlink_to(
        Path("aggregate_generations") / generation.name,
        target_is_directory=True,
    )


def _write_render_run(root: Path) -> None:
    _write_render_generation(root)
    states = [
        _write_state(
            root,
            state_id="controlled__shoe__q0__clean",
            track="controlled",
            corruption="clean",
            color=20,
        ),
        _write_state(
            root,
            state_id="controlled__shoe__q0__jpeg",
            track="controlled",
            corruption="jpeg",
            color=60,
        ),
        _write_state(
            root,
            state_id="stress__shoe__q0__deformed",
            track="stress",
            corruption="deformed",
            color=100,
        ),
    ]
    manifests = root / "manifests"
    manifests.mkdir()
    (manifests / "image_states.jsonl").write_text(
        "".join(json.dumps(state, sort_keys=True) + "\n" for state in states),
        encoding="utf-8",
    )
    pair_conditions = [
        {
            "design_id": "C100",
            "track": "controlled",
            "shoe_id": "shoe",
            "quartet_id": "q0",
            "pair_id": "shoe__q0__1__2",
            "first_view": "1",
            "second_view": "2",
            "baseline_steps": 1,
            "angular_separation_deg": 7.5,
            "changed_view": None,
            "image_state_id": "controlled__shoe__q0__clean",
            "corruption": "clean",
            "severity": 0.0,
        },
        {
            "design_id": "C200",
            "track": "controlled",
            "shoe_id": "shoe",
            "quartet_id": "q0",
            "pair_id": "shoe__q0__1__2",
            "first_view": "1",
            "second_view": "2",
            "baseline_steps": 1,
            "angular_separation_deg": 7.5,
            "changed_view": "1",
            "image_state_id": "controlled__shoe__q0__jpeg",
            "corruption": "jpeg",
            "severity": 1.0,
        },
        {
            "design_id": "S100",
            "track": "stress",
            "shoe_id": "shoe",
            "quartet_id": "q0",
            "pair_id": "shoe__q0__1__2",
            "first_view": "1",
            "second_view": "2",
            "baseline_steps": 1,
            "angular_separation_deg": 7.5,
            "changed_view": None,
            "image_state_id": "stress__shoe__q0__deformed",
            "corruption": "deformed",
            "severity": 1.0,
        },
    ]
    (manifests / "pair_conditions.jsonl").write_text(
        "".join(
            json.dumps(condition, sort_keys=True) + "\n"
            for condition in pair_conditions
        ),
        encoding="utf-8",
    )
    _write_geometry(
        root
        / "geometry/calibration/controlled__shoe__q0__clean/depth_geometry.npz"
    )
    _write_geometry(
        root / "geometry/stress/stress__shoe__q0__deformed/depth_geometry.npz"
    )

    source_root = root / "clean_sources"
    source_root.mkdir()
    fit_images: list[dict[str, object]] = []
    for index in range(64):
        path = source_root / f"{index:02d}.png"
        pixels = np.full((12, 12, 3), 10 + index, dtype=np.uint8)
        pixels[index % 12, :, index % 3] = 250
        Image.fromarray(pixels, mode="RGB").save(path)
        fit_images.append(
            {
                "shoe_id": f"shoe-{index // 16}",
                "quartet_id": f"q{(index % 16) // 4}",
                "view_id": str(index % 4),
                "source_path": str(path),
            }
        )
    fit_root = root / "pca_fit_images"
    fit_root.mkdir()
    (fit_root / "clean_source_paths.json").write_text(
        json.dumps(fit_images, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = {
        "experiment_id": "20260727_adjacent16_core_v1",
        "stage_size": 504,
        "seed": 42,
        "camera_label": CAMERA_MODE_LABEL,
        "quartets": [{"quartet_id": "q0", "view_ids": ["1", "2", "3", "4"]}],
        "metric": {
            "relative_depth_tolerance": 0.03,
            "min_overlap_fraction": 0.08,
            "confidence_percentile": 10.0,
        },
        "models": {
            "dinov3_vitl16_pixal3d": {
                "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881"
            },
            "naf_revision": NAF_REVISION,
        },
        "visualization": {"policy": "posthoc_audit_only"},
        "visualization_variant_keys": [LOW_VARIANT, HIGH_VARIANT],
    }
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "frozen_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_hash = hashlib.sha256(
        (config_dir / "frozen_config.json").read_bytes()
    ).hexdigest()
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "20260727_adjacent16_core_v1",
                "scope": "full",
                "config_sha256": config_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generation = root / "metrics/aggregate_generations/render-generation"
    for name in (
        "controlled_fidelity.parquet",
        "real_deformation_stress.parquet",
    ):
        path = generation / name
        frame = pd.read_parquet(path)
        frame["config_sha256"] = config_hash
        frame["backbone_revision"] = (
            "3c276edd87d6f6e569ff0c4400e086807d0f3881"
        )
        frame.to_parquet(path, index=False)


def _patch_tiny_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "met3r.adjacent16_visualization.load_feature_backbone",
        lambda _name: _TinyBackbone(),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_visualization.load_naf_upsampler",
        lambda **_kwargs: _TinyNAF(),
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_no_forbidden_raw_artifacts(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in {".pt", ".pth"}
        assert path.name not in {
            "features.npy",
            "features.npz",
            "basis.npz",
        }
        if path.suffix.lower() != ".npz":
            continue
        with np.load(path, allow_pickle=False) as artifact:
            assert "components" not in artifact.files
            assert "pca_components" not in artifact.files
            artifact_type = next(
                (
                    str(np.asarray(artifact[name]).item())
                    for name in ("artifact_type", "meta__artifact_type")
                    if name in artifact.files
                ),
                None,
            )
            assert artifact_type != "dino_features"


def test_render_audit_replays_selected_cases_with_one_in_memory_shared_pca(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch per-image PCA, wrong variants, raw persistence, or incomplete provenance."""

    _write_render_run(tmp_path)
    _patch_tiny_models(monkeypatch)

    first_outputs = render_adjacent16_audit(tmp_path, device="cpu")
    directory = tmp_path / "reports/audit_visualizations"
    first_snapshot = _snapshot(directory)
    source_manifest = tmp_path / "pca_fit_images/clean_source_paths.json"
    reversed_sources = list(
        reversed(json.loads(source_manifest.read_text(encoding="utf-8")))
    )
    source_manifest.write_text(
        json.dumps(reversed_sources, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second_outputs = render_adjacent16_audit(tmp_path, device="cpu")
    second_snapshot = _snapshot(directory)
    render_manifest = json.loads(
        (directory / "pca_render_manifest.json").read_text(encoding="utf-8")
    )

    assert first_snapshot == second_snapshot
    assert tuple(path.relative_to(tmp_path) for path in first_outputs) == tuple(
        path.relative_to(tmp_path) for path in second_outputs
    )
    assert {
        path.name for path in directory.iterdir()
    } == {
        "audit_manifest.json",
        "pca",
        "directional",
        "pca_render_manifest.json",
    }
    assert all(
        path.suffix == ".png"
        for subdirectory in ("pca", "directional")
        for path in (directory / subdirectory).iterdir()
    )
    assert len(list((directory / "pca").glob("*.png"))) == 3 * 2 * 4
    assert len(list((directory / "directional").glob("*.png"))) == 3 * 2 * 2
    assert render_manifest["artifact_type"] == "adjacent16_in_memory_shared_pca"
    assert len(render_manifest["fit_images"]) == 64
    assert len(render_manifest["pca_fingerprint"]) == 64
    assert len(render_manifest["fit_policy_fingerprint"]) == 64
    assert render_manifest["global_bounds"].keys() == {"lower", "upper"}
    assert render_manifest["model_revision"] == (
        "3c276edd87d6f6e569ff0c4400e086807d0f3881"
    )
    assert render_manifest["naf_revision"] == NAF_REVISION
    assert render_manifest["fit_token_count"] == 64 * 37 * 37
    assert render_manifest["fit_sample_count"] == 50_000
    assert render_manifest["max_tokens"] == 50_000
    assert render_manifest["aggregate_generation"] == (
        "metrics/aggregate_generations/render-generation"
    )
    assert render_manifest["config_sha256"] == hashlib.sha256(
        (tmp_path / "config/frozen_config.json").read_bytes()
    ).hexdigest()
    assert render_manifest["audit_manifest_sha256"] == hashlib.sha256(
        (directory / "audit_manifest.json").read_bytes()
    ).hexdigest()
    aggregate_inputs = {
        record["path"]: record["sha256"]
        for record in render_manifest["aggregate_inputs"]
    }
    assert set(aggregate_inputs) == {
        "metrics/aggregate_generations/render-generation/controlled_fidelity.parquet",
        "metrics/aggregate_generations/render-generation/real_deformation_stress.parquet",
    }
    for relative, digest in aggregate_inputs.items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
    assert {
        output["path"] for output in render_manifest["outputs"]
    } == {
        str(path.relative_to(directory))
        for path in (*first_outputs,)
        if path.suffix == ".png"
    }
    for record in (
        *render_manifest["fit_images"],
        *render_manifest["outputs"],
    ):
        path = (
            Path(record["path"])
            if Path(record["path"]).is_absolute()
            else directory / record["path"]
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    serialized = json.dumps(render_manifest, sort_keys=True)
    assert "components" not in serialized
    assert "per_image_features" not in serialized
    _assert_no_forbidden_raw_artifacts(tmp_path)
    for path in (directory / "pca").glob("*.png"):
        with Image.open(path) as image:
            expected = 592 if HIGH_VARIANT in path.name else 37
            assert image.size == (expected, expected)
    with pytest.raises(ValueError, match="rendered audit snapshot"):
        build_adjacent16_audit_manifest(tmp_path)


def test_audit_selection_rejects_low_high_changed_view_mismatch(
    tmp_path: Path,
) -> None:
    """Catch exact design IDs pairing rows owned by different changed views."""

    generation = _write_numeric_generation(tmp_path)
    path = generation / "controlled_fidelity.parquet"
    frame = pd.read_parquet(path)
    frame.loc[
        (frame["design_id"] == "C410") & (frame["variant_key"] == HIGH_VARIANT),
        "changed_view",
    ] = "2"
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="changed_view"):
        build_adjacent16_audit_manifest(tmp_path)


def test_render_audit_rejects_aggregate_case_not_owned_by_pair_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch selected aggregate identity drifting from its canonical condition."""

    _write_render_run(tmp_path)
    path = tmp_path / "manifests/pair_conditions.jsonl"
    conditions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    next(
        condition for condition in conditions if condition["design_id"] == "C200"
    )["changed_view"] = "2"
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in conditions),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        adjacent16_visualization,
        "load_feature_backbone",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("model load must follow owner validation")
        ),
    )

    with pytest.raises(ValueError, match="canonical pair condition"):
        render_adjacent16_audit(tmp_path, device="cpu")


def test_render_audit_rejects_aggregate_model_or_config_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch selected numeric rows detached from the frozen model contract."""

    _write_render_run(tmp_path)
    path = (
        tmp_path
        / "metrics/aggregate_generations/render-generation/controlled_fidelity.parquet"
    )
    frame = pd.read_parquet(path)
    frame.loc[frame["design_id"] == "C200", "config_sha256"] = "changed"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        adjacent16_visualization,
        "load_feature_backbone",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("model load must follow aggregate validation")
        ),
    )

    with pytest.raises(ValueError, match="aggregate provenance"):
        render_adjacent16_audit(tmp_path, device="cpu")


def test_render_audit_resolves_one_aggregate_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch selection from one generation being hashed as a later generation."""

    _write_render_run(tmp_path)
    generations = tmp_path / "metrics/aggregate_generations"
    replacement = generations / "replacement-generation"
    shutil.copytree(generations / "render-generation", replacement)
    changed = replacement / "controlled_fidelity.parquet"
    frame = pd.read_parquet(changed)
    frame["config_sha256"] = "replacement"
    frame.to_parquet(changed, index=False)
    real_build = adjacent16_visualization._build_audit_payload

    def switch_after_selection(
        root: Path,
        generation: Path | None = None,
    ) -> dict[str, object]:
        payload = real_build(root)
        current = root / "metrics/current"
        current.unlink()
        current.symlink_to(
            Path("aggregate_generations") / replacement.name,
            target_is_directory=True,
        )
        return payload

    monkeypatch.setattr(
        adjacent16_visualization,
        "_build_audit_payload",
        switch_after_selection,
    )
    monkeypatch.setattr(
        adjacent16_visualization,
        "_fit_source_records",
        lambda _root: (_ for _ in ()).throw(
            RuntimeError("single aggregate snapshot retained")
        ),
    )

    with pytest.raises(RuntimeError, match="single aggregate snapshot retained"):
        render_adjacent16_audit(tmp_path, device="cpu")


def test_replay_rejects_feature_outputs_not_exactly_l37_and_l592() -> None:
    """Catch labels claiming frozen resolutions for wrong spatial tensors."""

    images = torch.zeros(4, 3, 592, 592)
    native = torch.zeros(4, 4, 37, 37)

    with pytest.raises(ValueError, match="592.*spatial"):
        list(
            adjacent16_visualization._paired_variant_features(
                native,
                images,
                native.clone(),
                images.clone(),
                _WrongShapeNAF(),
            )
        )
    with pytest.raises(ValueError, match="native.*37"):
        next(
            adjacent16_visualization._paired_variant_features(
                torch.zeros(4, 4, 32, 32),
                images,
                torch.zeros(4, 4, 32, 32),
                images,
                _TinyNAF(),
            )
        )


@pytest.mark.parametrize(
    ("relative_path", "arrays"),
    (
        ("leaked.pt", None),
        ("nested/leaked.pth", None),
        ("features.npy", np.asarray([1.0])),
        ("features.npz", {"values": np.asarray([1.0])}),
        (
            "other.npz",
            {"meta__artifact_type": np.asarray("dino_features")},
        ),
        (
            "byte_marker.npz",
            {"artifact_type": np.asarray(b"dino_features")},
        ),
        ("basis.npz", {"values": np.asarray([1.0])}),
        ("other_basis.npz", {"components": np.eye(3)}),
        ("pca_components.npy", np.eye(3)),
    ),
)
def test_render_audit_rejects_existing_raw_feature_or_basis_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    arrays: object,
) -> None:
    """Catch post-hoc rendering proceeding while prohibited feature state exists."""

    _write_render_run(tmp_path)
    _patch_tiny_models(monkeypatch)
    legacy = tmp_path / "reports/audit_visualizations"
    legacy.mkdir(parents=True)
    (legacy / "audit_manifest.json").write_text(
        '{"legacy": true}\n',
        encoding="utf-8",
    )
    before = _snapshot(legacy)
    leaked = tmp_path / relative_path
    leaked.parent.mkdir(parents=True, exist_ok=True)
    if arrays is None:
        leaked.write_bytes(b"raw")
    elif leaked.suffix == ".npy":
        np.save(leaked, arrays)
    else:
        assert isinstance(arrays, dict)
        np.savez_compressed(leaked, **arrays)

    with pytest.raises(ValueError, match="prohibited"):
        render_adjacent16_audit(tmp_path, device="cpu")

    assert _snapshot(legacy) == before


def test_render_audit_failure_preserves_previous_published_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a replay exception deleting or partially replacing prior audit PNGs."""

    _write_render_run(tmp_path)
    _patch_tiny_models(monkeypatch)
    directory = tmp_path / "reports/audit_visualizations"
    (directory / "pca").mkdir(parents=True)
    (directory / "directional").mkdir()
    (directory / "audit_manifest.json").write_text(
        '{"legacy": true}\n',
        encoding="utf-8",
    )
    (directory / "pca_render_manifest.json").write_text(
        '{"legacy": true}\n',
        encoding="utf-8",
    )
    (directory / "pca/old.png").write_bytes(b"old-pca")
    (directory / "directional/old.png").write_bytes(b"old-directional")
    before = _snapshot(directory)

    monkeypatch.setattr(
        "met3r.adjacent16_visualization.save_direction_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected diagnostic failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        render_adjacent16_audit(tmp_path, device="cpu")

    assert _snapshot(directory) == before
    assert not tuple(
        (tmp_path / "reports").glob(".audit_visualizations.staging.*")
    )
    assert not tuple(
        (tmp_path / "reports").glob(".audit_visualizations.backup.*")
    )


def test_audit_publication_treats_backup_cleanup_as_post_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch recoverable backup cleanup reporting a false publication failure."""

    reports = tmp_path / "reports"
    destination = reports / "audit_visualizations"
    destination.mkdir(parents=True)
    (destination / "value.json").write_text('{"value": 1}\n', encoding="utf-8")
    staging = reports / ".audit_visualizations.staging.first"
    staging.mkdir()
    (staging / "value.json").write_text('{"value": 2}\n', encoding="utf-8")
    real_rmtree = adjacent16_visualization.shutil.rmtree

    def retain_backup(path: object, *args: object, **kwargs: object) -> None:
        if Path(path).name.startswith(".audit_visualizations.backup."):
            raise OSError("injected recoverable cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        adjacent16_visualization.shutil,
        "rmtree",
        retain_backup,
    )
    adjacent16_visualization._publish_audit(staging, destination)

    assert json.loads((destination / "value.json").read_text()) == {"value": 2}
    assert len(tuple(reports.glob(".audit_visualizations.backup.*"))) == 1

    monkeypatch.setattr(adjacent16_visualization.shutil, "rmtree", real_rmtree)
    next_staging = reports / ".audit_visualizations.staging.second"
    next_staging.mkdir()
    (next_staging / "value.json").write_text(
        '{"value": 3}\n',
        encoding="utf-8",
    )
    adjacent16_visualization._publish_audit(next_staging, destination)

    assert json.loads((destination / "value.json").read_text()) == {"value": 3}
    assert not tuple(reports.glob(".audit_visualizations.backup.*"))
