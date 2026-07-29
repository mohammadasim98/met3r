import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import met3r.upper_data as upper_data
from met3r.upper_data import (
    CAMERA_DATASET_BY_SHOE,
    EXPECTED_SHOES,
    UPPER_VIEWS,
    build_upper_manifest,
)


ROOT = Path("/root/dev/ByteSize/Reconstruction")


def test_real_upper_manifest_has_exactly_sixteen_views() -> None:
    manifest = build_upper_manifest(ROOT)
    assert tuple(shoe.shoe_id for shoe in manifest.shoes) == EXPECTED_SHOES
    assert all(tuple(view.view_id for view in shoe.views) == UPPER_VIEWS for shoe in manifest.shoes)
    assert sum(len(shoe.views) for shoe in manifest.shoes) == 16
    assert all(view.source_image.size == (1080, 1080) for shoe in manifest.shoes for view in shoe.views)
    assert min(view.processed_image_correlation for shoe in manifest.shoes for view in shoe.views) >= 0.99


def test_real_deformation_quality_matches_selected_attempt() -> None:
    manifest = build_upper_manifest(ROOT)
    for shoe in manifest.shoes:
        for view in shoe.views:
            quality = view.quality
            assert quality.status == "low_quality"
            assert quality.selected_attempt_id
            assert quality.fit_relation in {"improved", "equivalent", "worsened"}
            assert isinstance(quality.semantic_gate, bool)
            assert isinstance(quality.preservation_gate, bool)
            assert isinstance(quality.camera_preserved, bool)
            assert isinstance(quality.outsole_preserved, bool)


def test_manifest_json_contains_only_absolute_paths_and_scalars(tmp_path: Path) -> None:
    manifest = build_upper_manifest(ROOT)

    destination = manifest.write_json(tmp_path / "manifest.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert Path(payload["reconstruction_root"]).is_absolute()
    assert Path(payload["shoes"][0]["views"][0]["source_path"]).is_absolute()
    assert isinstance(payload["shoes"][0]["views"][0]["processed_image_correlation"], float)


def _valid_deformation_payload() -> dict[str, object]:
    selected = {
        "attempt_id": "attempt_01",
        "semantic": {
            "passes_gate": False,
            "camera_preserved": True,
            "background_preserved": True,
            "outsole_preserved": True,
            "last_is_hidden": True,
            "same_shoe_identity": 0.98,
            "design_preservation": 0.96,
            "photorealism": 0.94,
            "deformation_plausibility": 0.92,
            "confidence": 0.97,
            "quality_score": 95.0,
        },
        "preservation": {
            "passes_gate": False,
            "center_shift_normalized": 0.02,
            "bottom_shift_normalized": 0.01,
            "width_ratio": 1.04,
        },
        "comparison": {"relation": "improved"},
        "metrics": {"fit_score": 90.0},
    }
    return {
        "status": "low_quality",
        "selected_attempt_id": "attempt_01",
        "attempts": [selected],
    }


def test_quality_retains_noncompliance_as_continuous_reliability_metadata() -> None:
    quality = upper_data._quality(
        _valid_deformation_payload(),
        {"status": "accepted", "generation_valid": True},
    )

    assert quality.generation_valid is True
    assert quality.strict_compliant is False
    assert quality.moderate_alignment is True
    assert quality.alignment_class == "moderate"
    assert quality.background_preserved is True
    assert quality.center_shift_normalized == pytest.approx(0.02)
    assert quality.bottom_shift_normalized == pytest.approx(0.01)
    assert quality.width_ratio == pytest.approx(1.04)
    assert quality.same_shoe_identity == pytest.approx(0.98)


@pytest.fixture
def upper_root(tmp_path: Path) -> Path:
    for shoe_id in EXPECTED_SHOES:
        camera_id = CAMERA_DATASET_BY_SHOE[shoe_id]
        camera_dir = tmp_path / "custom_dataset" / camera_id / "upper"
        camera_dir.mkdir(parents=True)
        np.savez(
            camera_dir / "processed_images.npz",
            processed_images=np.zeros((48, 2, 2, 3), dtype=np.uint8),
        )
        for view_id in UPPER_VIEWS:
            deformation_dir = (
                tmp_path
                / "six_view_experiment_result_precomputed"
                / "multiview"
                / shoe_id
                / view_id
                / "deformation"
            )
            deformation_dir.mkdir(parents=True)
            for name, mode in (
                ("source.png", "RGB"),
                ("deformed.png", "RGB"),
                ("source_mask.png", "L"),
                ("deformed_mask.png", "L"),
            ):
                Image.new(mode, (1080, 1080)).save(deformation_dir / name)
            deformation = _valid_deformation_payload()
            selected = deformation["attempts"][0]
            (deformation_dir / "deformation.json").write_text(json.dumps(deformation))
            (deformation_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "after": selected["metrics"],
                        "comparison": selected["comparison"],
                        "preservation": selected["preservation"],
                        "verdict": deformation["status"],
                    }
                )
            )
            (deformation_dir / "task.json").write_text(json.dumps({"status": "ok"}))
    return tmp_path


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("semantic", "passes_gate"), "false"),
        (("preservation", "passes_gate"), 0),
        (("semantic", "camera_preserved"), 1),
        (("semantic", "outsole_preserved"), None),
    ],
)
def test_quality_rejects_non_boolean_audit_values(
    path: tuple[str, str], value: object
) -> None:
    deformation = _valid_deformation_payload()
    deformation["attempts"][0][path[0]][path[1]] = value

    with pytest.raises(ValueError, match="must be a boolean"):
        upper_data._quality(deformation, {"status": "ok"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", ""),
        ("selected_attempt_id", ""),
    ],
)
def test_quality_rejects_empty_required_deformation_strings(field: str, value: str) -> None:
    deformation = _valid_deformation_payload()
    deformation[field] = value

    with pytest.raises(ValueError, match="non-empty string"):
        upper_data._quality(deformation, {"status": "ok"})


def test_quality_rejects_unknown_fit_relation() -> None:
    deformation = _valid_deformation_payload()
    deformation["attempts"][0]["comparison"]["relation"] = "same"

    with pytest.raises(ValueError, match="fit relation"):
        upper_data._quality(deformation, {"status": "ok"})


def test_quality_rejects_non_string_task_status() -> None:
    with pytest.raises(ValueError, match="task status"):
        upper_data._quality(_valid_deformation_payload(), {"status": False})


@pytest.mark.parametrize("key", ("after", "comparison", "preservation", "verdict"))
def test_metrics_rejects_selected_attempt_mismatch(key: str) -> None:
    deformation = _valid_deformation_payload()
    selected = deformation["attempts"][0]
    metrics = {
        "after": selected["metrics"],
        "comparison": selected["comparison"],
        "preservation": selected["preservation"],
        "verdict": deformation["status"],
    }
    metrics[key] = {"wrong": True} if key != "verdict" else "ok"

    with pytest.raises(ValueError, match="metrics"):
        upper_data._validate_metrics(metrics, deformation, selected)


def test_metrics_rejects_type_coercion_when_cross_checking() -> None:
    deformation = _valid_deformation_payload()
    selected = deformation["attempts"][0]
    selected["metrics"] = {"fit_score": 1}
    metrics = {
        "after": {"fit_score": True},
        "comparison": selected["comparison"],
        "preservation": selected["preservation"],
        "verdict": deformation["status"],
    }

    with pytest.raises(ValueError, match="metrics after"):
        upper_data._validate_metrics(metrics, deformation, selected)


def test_manifest_rejects_malformed_selected_attempt(upper_root: Path) -> None:
    deformation_path = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation/deformation.json"
    )
    deformation = json.loads(deformation_path.read_text())
    deformation["selected_attempt_id"] = "missing"
    deformation_path.write_text(json.dumps(deformation))

    with pytest.raises(ValueError, match="selected attempt"):
        build_upper_manifest(upper_root)


def test_manifest_rejects_missing_artifact(upper_root: Path) -> None:
    path = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation/deformed.png"
    )
    path.unlink()

    with pytest.raises(FileNotFoundError, match="deformed image"):
        build_upper_manifest(upper_root)


def test_manifest_rejects_mismatched_mask_size(upper_root: Path) -> None:
    path = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation/source_mask.png"
    )
    Image.new("L", (1079, 1080)).save(path)

    with pytest.raises(ValueError, match="identical dimensions"):
        build_upper_manifest(upper_root)


def test_manifest_rejects_non_1080_images(upper_root: Path) -> None:
    deformation_dir = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation"
    )
    for name, mode in (
        ("source.png", "RGB"),
        ("deformed.png", "RGB"),
        ("source_mask.png", "L"),
        ("deformed_mask.png", "L"),
    ):
        Image.new(mode, (2, 2)).save(deformation_dir / name)

    with pytest.raises(ValueError, match="1080x1080"):
        build_upper_manifest(upper_root)


def test_manifest_rejects_correlation_below_gate(upper_root: Path) -> None:
    path = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation/source.png"
    )
    Image.new("RGB", (1080, 1080), color="white").save(path)

    with pytest.raises(ValueError, match="correlation"):
        build_upper_manifest(upper_root)


@pytest.mark.parametrize("replacement", ([], {"after": {"fit_score": 0.0}}))
def test_manifest_rejects_malformed_or_mismatched_metrics(
    upper_root: Path, replacement: object
) -> None:
    metrics_path = (
        upper_root
        / "six_view_experiment_result_precomputed/multiview"
        / EXPECTED_SHOES[0]
        / UPPER_VIEWS[0]
        / "deformation/metrics.json"
    )
    metrics_path.write_text(json.dumps(replacement))

    with pytest.raises(ValueError, match="metrics"):
        build_upper_manifest(upper_root)
