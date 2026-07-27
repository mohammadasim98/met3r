"""Tests for discovery and validation of dense adjacent-camera inputs."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import build_adjacent16_manifest, pair_baseline
from met3r.deformation_audit import DeformationQuality
from met3r.upper_data import DeformationQuality as LegacyDeformationQuality
from tests.helpers_adjacent16 import write_dense_source, write_valid_stress


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


@pytest.fixture
def dense_contract(tmp_path: Path):
    contract = load_adjacent16_contract(CONFIG)
    root = tmp_path / "reconstruction"
    stress = tmp_path / "stress"
    write_dense_source(root, contract)
    return replace(contract, reconstruction_root=root, stress_root=stress)


@pytest.fixture
def irregular_dense_contract(tmp_path: Path):
    contract = load_adjacent16_contract(CONFIG)
    root = tmp_path / "reconstruction"
    write_dense_source(root, contract, orbit_gaps_deg=(7.0, 8.0) * 24)
    return replace(contract, reconstruction_root=root, stress_root=tmp_path / "stress")


def test_deformation_audit_is_shared_without_changing_upper_legacy_import() -> None:
    """Catch reintroducing adjacent16's dependency on upper-data audit internals."""

    assert LegacyDeformationQuality is DeformationQuality


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


def test_manifest_rejects_present_stress_view_with_incomplete_audit(dense_contract) -> None:
    """Catch accepting a present selected stress view with missing audit metadata."""

    shoe_id = next(iter(dense_contract.raw["shoes"]))
    assert isinstance(shoe_id, str)
    write_valid_stress(dense_contract.stress_root, shoe_id, "1")
    (dense_contract.stress_root / shoe_id / "1/deformation/metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match="stress deformation metrics"):
        build_adjacent16_manifest(dense_contract, require_stress=True)


def test_pair_baselines_preserve_irregular_measured_camera_angles(
    irregular_dense_contract,
) -> None:
    """Catch replacing measured pair angles with a nominal 7.5 degrees per step."""

    manifest = build_adjacent16_manifest(irregular_dense_contract, require_stress=False)

    for shoe in manifest.shoes:
        pairs = [
            pair_baseline(first, second)
            for quartet in range(4)
            for first, second in combinations(shoe.views[quartet * 4 : quartet * 4 + 4], 2)
        ]
        assert {steps for steps, _ in pairs} == {1, 2, 3}
        assert pair_baseline(shoe.views[0], shoe.views[1]) == pytest.approx((1, 8.0))
        assert pair_baseline(shoe.views[0], shoe.views[3]) == pytest.approx((3, 23.0))
        assert pair_baseline(shoe.views[0], shoe.views[1])[1] != pytest.approx(7.5)
