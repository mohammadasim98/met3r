"""Exact quartet-aware enumeration for the frozen adjacent16 protocol."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import build_adjacent16_manifest, pair_baseline
from met3r.adjacent16_design import build_adjacent16_design
from tests.helpers_adjacent16 import write_dense_source


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


@pytest.fixture(scope="module")
def dense_manifest(tmp_path_factory: pytest.TempPathFactory):
    """A real dense manifest with deliberately non-nominal measured angles."""

    contract = load_adjacent16_contract(CONFIG)
    root = tmp_path_factory.mktemp("adjacent16-design") / "reconstruction"
    write_dense_source(root, contract, orbit_gaps_deg=(7.0, 8.0) * 24)
    contract = replace(contract, reconstruction_root=root)
    return build_adjacent16_manifest(contract, require_stress=False), contract


def test_design_expands_the_exact_frozen_matrix(dense_manifest) -> None:
    """Catch dropping a quartet, state placement, pair, or feature variant."""

    manifest, contract = dense_manifest
    design = build_adjacent16_design(manifest, contract)

    assert len(design.controlled_states) == 848
    assert len(design.stress_states) == 16
    assert len(design.controlled_conditions) == 2_592
    assert len(design.stress_conditions) == 96
    assert len(design.feature_variants) == 12
    assert design.row_counts == {
        "controlled": 31_104,
        "geometry_calibration": 6_528,
        "stress": 1_152,
    }


def test_clean_and_changed_states_attach_only_their_quartet_pairs(dense_manifest) -> None:
    """Catch clean-pair undercoverage or changed-view pairs from another quartet."""

    manifest, contract = dense_manifest
    design = build_adjacent16_design(manifest, contract)
    clean = [state for state in design.controlled_states if state.corruption == "clean"]
    changed = [state for state in design.controlled_states if state.corruption != "clean"]

    assert len(clean) == 16
    assert len(changed) == 832
    assert all(
        len([row for row in design.controlled_conditions if row.image_state_id == state.state_id]) == 6
        for state in clean
    )
    assert all(
        len([row for row in design.controlled_conditions if row.image_state_id == state.state_id]) == 3
        for state in changed
    )

    for shoe in manifest.shoes:
        for quartet in contract.quartets:
            rows = [
                row
                for row in design.stress_conditions
                if row.shoe_id == shoe.shoe_id and row.quartet_id == quartet.quartet_id
            ]
            assert [row.baseline_steps for row in rows].count(1) == 3
            assert [row.baseline_steps for row in rows].count(2) == 2
            assert [row.baseline_steps for row in rows].count(3) == 1


def test_pair_conditions_preserve_measured_baselines_and_quartet_ids(dense_manifest) -> None:
    """Catch nominal 7.5-degree substitution or quartet-free pair collisions."""

    manifest, contract = dense_manifest
    design = build_adjacent16_design(manifest, contract)
    by_shoe_view = {
        (shoe.shoe_id, view.view_id): view for shoe in manifest.shoes for view in shoe.views
    }

    for row in (*design.controlled_conditions, *design.stress_conditions):
        first = by_shoe_view[row.shoe_id, row.first_view]
        second = by_shoe_view[row.shoe_id, row.second_view]
        assert row.pair_id == f"{row.quartet_id}__{row.first_view}__{row.second_view}"
        assert (row.baseline_steps, row.angular_separation_deg) == pytest.approx(
            pair_baseline(first, second)
        )
    assert any(row.angular_separation_deg == pytest.approx(8.0) for row in design.stress_conditions)
    assert len({row.pair_id for row in design.stress_conditions}) == 24


def test_identity_donors_and_transformation_values_are_state_deterministic(dense_manifest) -> None:
    """Catch donor-view remapping or pair-specific randomized transformations."""

    manifest, contract = dense_manifest
    first = build_adjacent16_design(manifest, contract)
    second = build_adjacent16_design(manifest, contract)

    assert first.controlled_states == second.controlled_states
    shoes = [shoe.shoe_id for shoe in manifest.shoes]
    for state in first.controlled_states:
        if state.corruption == "identity_swap":
            assert state.donor_shoe_id == shoes[(shoes.index(state.shoe_id) + 1) % len(shoes)]
            assert state.donor_view == state.changed_view
        else:
            assert state.donor_shoe_id is None
            assert state.donor_view is None

    changed = [state for state in first.controlled_states if state.changed_view is not None]
    signatures = {
        (state.shoe_id, state.quartet_id, state.changed_view, state.corruption, state.severity):
        (state.direction, state.seed)
        for state in changed
    }
    assert len(signatures) == len(changed)
    assert all(state.quartet_id in state.state_id for state in first.controlled_states)
