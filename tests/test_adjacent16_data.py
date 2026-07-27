"""Tests for discovery and validation of dense adjacent-camera inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import build_adjacent16_manifest, pair_baseline
from tests.helpers_adjacent16 import write_dense_source


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


@pytest.fixture
def dense_contract(tmp_path: Path):
    contract = load_adjacent16_contract(CONFIG)
    root = tmp_path / "reconstruction"
    stress = tmp_path / "stress"
    write_dense_source(root, contract)
    return replace(contract, reconstruction_root=root, stress_root=stress)


def test_manifest_validates_all_sources_and_retains_frozen_adjacent_views(dense_contract) -> None:
    """Catch a builder that numerically sorts filenames or omits unselected validation."""

    manifest = build_adjacent16_manifest(dense_contract, require_stress=False)

    assert len(manifest.shoes) == 4
    assert sum(len(shoe.views) for shoe in manifest.shoes) == 64
    for shoe in manifest.shoes:
        assert tuple(view.view_id for view in shoe.views) == (
            "1", "2", "3", "4", "14", "15", "16", "17", "27", "28", "29", "30", "40", "41", "42", "0"
        )
        assert [view.orbit_position for view in shoe.views[:4]] == [9, 10, 11, 12]
        assert all(isinstance(view.view_id, str) for view in shoe.views)
        assert all(view.source_path.name == f"{int(view.view_id)}.jpg" for view in shoe.views)
        assert all(view.source_mask_path.name == f"{int(view.view_id) + 1}.jpg" for view in shoe.views)
        assert all(view.processed_image_correlation >= 0.99 for view in shoe.views)
        assert all(np.isfinite(view.camera_azimuth_deg) for view in shoe.views)
        for offset in range(0, 16, 4):
            assert pair_baseline(shoe.views[offset], shoe.views[offset + 1]) == pytest.approx((1, 7.5))
        steps, angle = pair_baseline(shoe.views[0], shoe.views[3])
        assert (steps, angle) == pytest.approx((3, 22.5))


@pytest.mark.parametrize("kind", ("missing-image", "extra-mask", "missing-camera"))
def test_manifest_rejects_incomplete_or_extra_dense_sources(dense_contract, kind: str) -> None:
    """Catch accepting any 48-view source that has missing or extra records."""

    display = next(iter(dense_contract.raw["shoes"].values()))
    assert isinstance(display, str)
    group = dense_contract.reconstruction_root / "sample_result" / display / "upper"
    if kind == "missing-image":
        (group / "glb_masked_white_bg/masked_inputs/47.jpg").unlink()
    elif kind == "extra-mask":
        (group / "mask/49.jpg").touch()
    else:
        (group / "glb_masked_white_bg/outputs/extrinsics.npz").unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="48|image|mask|camera"):
        build_adjacent16_manifest(dense_contract, require_stress=False)


def test_manifest_requires_complete_strict_stress_audits(dense_contract) -> None:
    """Catch allowing a stress view to bypass its deformation audit metadata."""

    with pytest.raises(FileNotFoundError, match="stress"):
        build_adjacent16_manifest(dense_contract, require_stress=True)
