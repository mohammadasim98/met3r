"""Synthetic dense-camera sources for adjacent16 manifest tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from met3r.adjacent16_contract import Adjacent16Contract


def write_dense_source(
    root: Path,
    contract: Adjacent16Contract,
    *,
    orbit_gaps_deg: tuple[float, ...] = (7.5,) * 48,
) -> None:
    """Write four complete 48-view sources with deliberately nonnumeric orbit order."""

    for shoe_offset, display_name in enumerate(contract.raw["shoes"].values()):
        assert isinstance(display_name, str)
        group = root / "sample_result" / display_name / "upper"
        inputs = group / "glb_masked_white_bg" / "masked_inputs"
        masks = group / "mask"
        outputs = group / "glb_masked_white_bg" / "outputs"
        inputs.mkdir(parents=True)
        masks.mkdir(parents=True)
        outputs.mkdir(parents=True)
        processed = []
        for index in range(48):
            image = _view_image(index, shoe_offset)
            image.save(inputs / f"{index}.jpg", quality=100, subsampling=0)
            Image.new("L", (1080, 1080), color=index).save(masks / f"{index + 1}.jpg")
            with Image.open(inputs / f"{index}.jpg") as saved:
                processed.append(
                    np.asarray(
                        saved.resize((2, 2), resample=Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                )
        np.savez(outputs / "processed_images.npz", processed_images=np.stack(processed))
        np.savez(outputs / "intrinsics.npz", intrinsics=np.broadcast_to(np.eye(3), (48, 3, 3)))
        np.savez(
            outputs / "extrinsics.npz",
            extrinsics=_orbit_extrinsics(contract.camera_order, orbit_gaps_deg),
        )


def write_valid_stress(root: Path, shoe_id: str, quartet_id: str, view_id: str) -> None:
    """Write a strict deformation audit for one selected synthetic view."""

    deformation = root / shoe_id / quartet_id / f"view_{view_id}" / "deformation"
    deformation.mkdir(parents=True)
    for name, mode in (("deformed.png", "RGB"), ("deformed_mask.png", "L")):
        Image.new(mode, (1080, 1080)).save(deformation / name)
    selected = {
        "attempt_id": "attempt_01",
        "semantic": {
            "passes_gate": True,
            "camera_preserved": True,
            "outsole_preserved": True,
        },
        "preservation": {"passes_gate": True},
        "comparison": {"relation": "equivalent"},
        "metrics": {"fit_score": 1.0},
    }
    payload = {"status": "ok", "selected_attempt_id": "attempt_01", "attempts": [selected]}
    (deformation / "deformation.json").write_text(json.dumps(payload))
    (deformation / "metrics.json").write_text(
        json.dumps(
            {
                "after": selected["metrics"],
                "comparison": selected["comparison"],
                "preservation": selected["preservation"],
                "verdict": "ok",
            }
        )
    )
    (deformation / "task.json").write_text(json.dumps({"status": "ok"}))


def _view_image(index: int, shoe_offset: int) -> Image.Image:
    colors = np.array(
        [
            [index * 5 % 256, shoe_offset * 61 % 256, 10],
            [40, index * 11 % 256, shoe_offset * 71 % 256],
            [shoe_offset * 37 % 256, 90, index * 3 % 256],
            [index * 13 % 256, 160, shoe_offset * 23 % 256],
        ],
        dtype=np.uint8,
    )
    pixels = np.empty((1080, 1080, 3), dtype=np.uint8)
    pixels[:540, :540] = colors[0]
    pixels[:540, 540:] = colors[1]
    pixels[540:, :540] = colors[2]
    pixels[540:, 540:] = colors[3]
    return Image.fromarray(pixels, mode="RGB")


def _orbit_extrinsics(
    camera_order: tuple[str, ...], orbit_gaps_deg: tuple[float, ...]
) -> np.ndarray:
    """48 world-to-camera matrices whose camera centers circle the origin."""

    if len(orbit_gaps_deg) != 48 or not np.isclose(sum(orbit_gaps_deg), 360.0):
        raise ValueError("orbit gaps must contain 48 values summing to 360 degrees")
    values = np.empty((48, 3, 4), dtype=np.float64)
    angle_deg = 0.0
    for orbit_position, view_id in enumerate(camera_order):
        index = int(view_id)
        angle = np.deg2rad(angle_deg)
        center = np.array([np.cos(angle), 0.0, np.sin(angle)])
        forward = center
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        rotation = np.stack((right, np.array([0.0, 1.0, 0.0]), forward))
        values[index, :, :3] = rotation
        values[index, :, 3] = -rotation @ center
        angle_deg += orbit_gaps_deg[orbit_position]
    return values
