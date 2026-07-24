from __future__ import annotations

import json
from pathlib import Path

from met3r.design import build_design_rows


def test_known_camera_config_and_protocol_use_the_exact_camera_label() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "known_camera_bytesize.json").read_text())
    protocol = (root / "KNOWN_CAMERA_EXPERIMENT.md").read_text()

    label = "fixed/precomputed DA3 cameras (not calibrated GT)"
    assert config["dataset"]["camera_mode"] == label
    assert label in protocol


def test_design_expands_feature_corruption_and_pair_axes() -> None:
    config = {
        "feature_tracks": {
            "track": {
                "dino": {"input_size": 32, "native_grid": 2},
                "common_evaluation_grids": [2, 4],
            }
        },
        "upsampling_modes": ["bilinear", "naf_normal_guide"],
        "corruptions": {"translation": [0.0, 0.5]},
    }
    manifest = {
        "shoes": [
            {
                "shoe_id": "shoe-a",
                "views": [
                    {"view_id": "left", "camera_group": "upper"},
                    {"view_id": "right", "camera_group": "upper"},
                    {"view_id": "sole", "camera_group": "bottom"},
                ],
            }
        ]
    }

    rows = build_design_rows(config, manifest)

    assert any(
        row["feature_mode"] == "native"
        and row["resolution"] == 2
        and row["analysis_tier"] == "primary"
        for row in rows
    )
    assert any(
        row["feature_mode"] == "naf_normal_guide"
        and row["resolution"] == 4
        for row in rows
    )
    assert any(
        row["pair_scope"] == "cross_group"
        and row["analysis_tier"] == "sensitivity_only"
        for row in rows
    )
    assert {row["severity"] for row in rows} == {0.0, 0.5}
    assert len({row["design_id"] for row in rows}) == len(rows)
    assert all(row["design_role"] == "audit_only" for row in rows)
    assert all(row["first_index"] < row["second_index"] for row in rows)
    assert all(row["corruption_seed"] == 42 for row in rows)


def test_design_current_track_uses_explicit_naf_targets() -> None:
    config = {
        "feature_tracks": {
            "pixal": {
                "dinov3": {
                    "input_size": 512,
                    "native_grid": 32,
                    "naf_targets": [128, 512],
                }
            }
        },
        "upsampling_modes": ["bilinear", "naf_normal_guide"],
        "corruptions": {"clean": [0.0]},
    }
    manifest = {
        "shoes": [
            {
                "shoe_id": "shoe-a",
                "views": [
                    {"view_id": "left", "camera_group": "upper"},
                    {"view_id": "right", "camera_group": "upper"},
                ],
            }
        ]
    }

    rows = build_design_rows(config, manifest)

    assert {(row["feature_mode"], row["resolution"]) for row in rows} == {
        ("native", 32),
        ("bilinear", 128),
        ("bilinear", 512),
        ("naf_normal_guide", 128),
        ("naf_normal_guide", 512),
    }
