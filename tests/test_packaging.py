from __future__ import annotations

import subprocess
import sys
import zipfile
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).parents[1]


def test_known_camera_packaging_includes_parquet_dependency() -> None:
    requirements = (REPOSITORY / "requirements-known-camera.txt").read_text().splitlines()
    setup_source = (REPOSITORY / "setup.py").read_text()

    assert "pyarrow>=14" in requirements
    assert '"pyarrow>=14",' in setup_source


def test_frozen_upper_configuration_binds_every_experiment_value() -> None:
    config = json.loads((REPOSITORY / "configs/upper_dino_fidelity.json").read_text())

    assert config["experiment_id"] == "20260724_upper_v2"
    assert config["reconstruction_root"] == "/root/dev/ByteSize/Reconstruction"
    assert (
        config["output_root"]
        == "/root/dev/ByteSize/Reconstruction/met3r_upper_experiments/20260724_upper_v2"
    )
    assert config["da3_repository"] == "/root/dev/ByteSize/Reconstruction/third_party/Depth-Anything-3"
    assert config["camera_label"] == "fixed/precomputed DA3 cameras (not calibrated GT)"
    assert config["views"] == ["left", "back", "right", "front"]
    assert config["seed"] == 42
    assert config["stage_size"] == 504
    assert config["metric"] == {
        "relative_depth_tolerance": 0.03,
        "min_overlap_fraction": 0.08,
        "confidence_percentile": 10.0,
    }
    assert config["persistent_storage_limit_gb"] == 300
    assert config["physical_gpus"] == [4, 5, 6, 7]
    assert config["corruptions"] == {
        "translation": [0.03, 0.06, 0.12, 0.20],
        "thin_plate_warp": [0.25, 0.50, 0.75, 1.00],
        "exposure": [0.15, 0.30, 0.50, 0.75],
        "blur": [0.20, 0.40, 0.70, 1.00],
        "jpeg": [0.25, 0.50, 0.75, 1.00],
        "identity_swap": [1.00],
    }
    assert config["pilot"] == {
        "shoes": [
            "air_huarache_run_ultra_111449",
            "air_max_90_surplus_cq7743_300",
        ],
        "pairs": ["right__front", "left__right"],
        "corruptions": {
            "translation": [0.12, 0.20],
            "exposure": [0.50, 0.75],
            "identity_swap": [1.00],
        },
        "backbones": ["dinov3_vitl16_pixal3d"],
    }
    assert config["models"] == {
        "dino_vitb16": {"revision": "7c446df5b9f45747937fb0d72314eb9f7b66930a"},
        "dinov2_vitb14": {"revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415"},
        "dinov3_vitb16": {"revision": "5931719e67bbdb9737e363e781fb0c67687896bc"},
        "dinov3_vitl16_pixal3d": {
            "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881"
        },
        "naf_revision": "37f2dfc180f2de53d98bd601109c0da0dd6b0f43",
        "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
        "da3_revision": "8615eefb62f2db4f8d6ebaa59160086981672829",
    }
    assert config["feature_tracks"] == {
        "user_37_grid_track": {
            "dino_vitb16": {"input_size": 592, "native_grid": 37},
            "dinov2_vitb14": {"input_size": 588, "native_grid": 42},
            "dinov3_vitb16": {"input_size": 592, "native_grid": 37},
            "dinov3_vitl16_pixal3d": {"input_size": 592, "native_grid": 37},
            "common_evaluation_grids": [37, 74, 148, 296, 592],
        },
        "current_pixal3d_code_track": {
            "dinov3_vitl16_pixal3d": {
                "input_size": 512,
                "native_grid": 32,
                "naf_targets": [128, 512],
            }
        },
    }
    assert config["upsampling_modes"] == [
        "bilinear",
        "naf_normal_guide",
        "naf_blurred_guide",
        "naf_shuffled_guide",
    ]
    assert config["geometry_calibration"] == {
        "rotation_degrees": [0.25, 0.5, 1.0, 2.0, 5.0],
        "relative_translation": [0.0025, 0.005, 0.01, 0.02],
        "depth_scale": [1.01, 1.02, 1.05, 1.1],
        "relative_depth_noise_std": [0.005, 0.01, 0.02, 0.05],
    }
    assert "bottom" not in json.dumps(config)
    assert "part_transfer" not in config["corruptions"]


@pytest.mark.skipif(
    not (REPOSITORY / "mast3r" / "mast3r" / "__init__.py").is_file(),
    reason="MASt3R/DUSt3R submodules are not initialized",
)
def test_wheel_keeps_legacy_nested_packages(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            str(REPOSITORY),
            "-w",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("met3r-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "met3r/met3r.py" in names
    assert "mast3r/mast3r/__init__.py" in names
    assert "mast3r/dust3r/dust3r/__init__.py" in names
    assert "mast3r/dust3r/croco/models/curope/__init__.py" in names
