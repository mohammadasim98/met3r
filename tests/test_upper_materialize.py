from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from met3r.upper_data import build_upper_manifest
from met3r.upper_design import build_upper_design
from met3r.upper_materialize import materialize_image_state


ROOT = Path("/root/dev/ByteSize/Reconstruction")


@pytest.fixture(scope="session")
def real_upper_manifest():
    return build_upper_manifest(ROOT)


@pytest.fixture(scope="session")
def frozen_config() -> dict[str, object]:
    return json.loads(
        (Path(__file__).resolve().parents[1] / "configs/upper_dino_fidelity.json").read_text()
    )


def test_materialize_stages_binary_white_background_and_identity_donor(
    tmp_path: Path, real_upper_manifest, frozen_config
) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    state = next(
        item
        for item in design.changed_states
        if item.shoe_id == "air_huarache_run_ultra_111449"
        and item.changed_view == "left"
        and item.corruption == "identity_swap"
    )

    result = materialize_image_state(state, real_upper_manifest, tmp_path)

    image = np.asarray(Image.open(result.state_dir / "images" / "left.png").convert("RGB"))
    mask = np.asarray(Image.open(result.state_dir / "masks" / "left.png").convert("L"))
    assert image.shape[:2] == (504, 504)
    assert mask.shape == (504, 504)
    assert set(np.unique(mask)) <= {0, 255}
    assert np.all(image[mask == 0] == 255)

    donor = next(shoe for shoe in real_upper_manifest.shoes if shoe.shoe_id == state.donor_shoe_id)
    donor_view = next(view for view in donor.views if view.view_id == "left")
    with Image.open(donor_view.source_path) as donor_image:
        expected_rgb = np.asarray(
            donor_image.convert("RGB").resize((504, 504), Image.Resampling.LANCZOS)
        )
    with Image.open(donor_view.source_mask_path) as donor_mask:
        expected_mask = np.asarray(
            donor_mask.convert("L").resize((504, 504), Image.Resampling.NEAREST)
        ) > 127
    expected = np.where(expected_mask[..., None], expected_rgb, 255).astype(np.uint8)
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(result.masks[0], expected_mask)
    assert result.metadata_path.is_file()


@pytest.mark.parametrize(
    ("corruption", "severity", "direction"),
    [
        ("exposure", 0.5, -1),
        ("blur", 1.0, 1),
        ("jpeg", 0.75, 1),
        ("thin_plate_warp", 0.75, 1),
    ],
)
def test_materialize_forces_white_background_after_every_corruption(
    tmp_path: Path,
    real_upper_manifest,
    frozen_config,
    corruption: str,
    severity: float,
    direction: int,
) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    base_state = next(
        state
        for state in design.changed_states
        if state.shoe_id == "air_huarache_run_ultra_111449"
        and state.changed_view == "left"
        and state.corruption == corruption
        and state.severity == severity
    )
    state = replace(
        base_state,
        state_id=f"{base_state.state_id}__background_regression",
        direction=direction,
    )

    result = materialize_image_state(state, real_upper_manifest, tmp_path)

    for image_path, mask_path, returned_mask in zip(
        result.image_paths, result.mask_paths, result.masks
    ):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        saved_mask = np.asarray(Image.open(mask_path).convert("L"))
        assert image.shape[:2] == (504, 504)
        assert saved_mask.shape == (504, 504)
        assert set(np.unique(saved_mask)) <= {0, 255}
        np.testing.assert_array_equal(returned_mask, saved_mask > 127)
        assert np.all(image[~returned_mask] == 255)
