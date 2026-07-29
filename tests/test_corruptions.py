from __future__ import annotations

import numpy as np
import pytest

from met3r.corruptions import (
    CorruptionSpec,
    apply_corruption,
    apply_corruption_with_mask,
)


def _image() -> np.ndarray:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 10:22] = [20, 100, 220]
    image[13:18, 14:19] = [240, 20, 20]
    return image


def test_translation_has_monotonic_pixel_displacement() -> None:
    source = _image()
    mild = apply_corruption(source, CorruptionSpec("translation", 0.05), seed=7)
    strong = apply_corruption(source, CorruptionSpec("translation", 0.20), seed=7)

    source_x = np.argwhere(np.any(source < 250, axis=-1))[:, 1].mean()
    mild_x = np.argwhere(np.any(mild < 250, axis=-1))[:, 1].mean()
    strong_x = np.argwhere(np.any(strong < 250, axis=-1))[:, 1].mean()
    assert source_x < mild_x < strong_x


def test_translation_direction_moves_pixels_left_or_right() -> None:
    source = _image()

    left = apply_corruption(source, CorruptionSpec("translation", 0.20, -1))
    right = apply_corruption(source, CorruptionSpec("translation", 0.20, 1))

    source_x = np.argwhere(np.any(source < 250, axis=-1))[:, 1].mean()
    left_x = np.argwhere(np.any(left < 250, axis=-1))[:, 1].mean()
    right_x = np.argwhere(np.any(right < 250, axis=-1))[:, 1].mean()
    assert left_x < source_x < right_x


def test_exposure_uses_reciprocal_signed_factors() -> None:
    source = np.full((2, 2, 3), 100, dtype=np.uint8)

    darker = apply_corruption(source, CorruptionSpec("exposure", 0.5, -1))
    brighter = apply_corruption(source, CorruptionSpec("exposure", 0.5, 1))

    assert darker[0, 0, 0] == 70
    assert brighter[0, 0, 0] == 141


def test_identity_swap_uses_donor_foreground_and_mask() -> None:
    source = _image()
    source_mask = np.any(source < 250, axis=-1)
    donor = np.full_like(source, [20, 220, 50])
    donor_mask = np.zeros(source.shape[:2], dtype=bool)
    donor_mask[3:12, 5:18] = True

    changed, changed_mask = apply_corruption_with_mask(
        source,
        source_mask,
        CorruptionSpec("identity_swap", 1.0),
        donor=donor,
        donor_foreground_mask=donor_mask,
    )

    assert changed_mask.dtype == bool
    np.testing.assert_array_equal(changed_mask, donor_mask)
    np.testing.assert_array_equal(changed[donor_mask], donor[donor_mask])
    assert np.all(changed[~donor_mask] == 255)


def test_thin_plate_warp_is_seeded_and_changes_geometry() -> None:
    source = _image()
    spec = CorruptionSpec("thin_plate_warp", 0.25)

    first = apply_corruption(source, spec, seed=123)
    second = apply_corruption(source, spec, seed=123)
    different = apply_corruption(source, spec, seed=456)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, source)
    assert not np.array_equal(first, different)


def test_part_transfer_changes_only_the_declared_mask() -> None:
    source = _image()
    donor = np.zeros_like(source)
    donor[...] = [0, 255, 0]
    part_mask = np.zeros(source.shape[:2], dtype=bool)
    part_mask[12:20, 12:20] = True

    result = apply_corruption(
        source,
        CorruptionSpec("part_transfer", 1.0),
        donor=donor,
        part_mask=part_mask,
    )

    np.testing.assert_array_equal(result[~part_mask], source[~part_mask])
    np.testing.assert_array_equal(result[part_mask], donor[part_mask])


def test_part_transfer_normalizes_uint8_alpha_masks() -> None:
    source = _image()
    donor = np.zeros_like(source)
    donor[...] = [0, 255, 0]
    part_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    part_mask[12:20, 12:20] = 255

    result = apply_corruption(
        source,
        CorruptionSpec("part_transfer", 1.0),
        donor=donor,
        part_mask=part_mask,
    )

    np.testing.assert_array_equal(result[part_mask > 0], donor[part_mask > 0])


def test_corruption_validates_strength_and_part_inputs() -> None:
    with pytest.raises(ValueError, match="strength"):
        CorruptionSpec("blur", 1.1)
    with pytest.raises(ValueError, match="donor"):
        apply_corruption(_image(), CorruptionSpec("part_transfer", 0.5))


def test_spatial_corruption_moves_foreground_mask_with_image() -> None:
    source = _image()
    mask = np.any(source < 250, axis=-1)

    changed, changed_mask = apply_corruption_with_mask(
        source,
        mask,
        CorruptionSpec("translation", 0.2),
        seed=3,
    )

    image_foreground = np.any(changed < 250, axis=-1)
    np.testing.assert_array_equal(changed_mask, image_foreground)
    assert changed_mask.dtype == bool
