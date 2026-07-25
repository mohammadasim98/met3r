from __future__ import annotations

import json
from pathlib import Path

import pytest

from met3r.upper_data import build_upper_manifest
from met3r.upper_design import build_upper_design, feature_variants


ROOT = Path("/root/dev/ByteSize/Reconstruction")


@pytest.fixture(scope="session")
def real_upper_manifest():
    return build_upper_manifest(ROOT)


@pytest.fixture(scope="session")
def frozen_config() -> dict[str, object]:
    return json.loads(
        (Path(__file__).resolve().parents[1] / "configs/upper_dino_fidelity.json").read_text()
    )


def test_full_design_counts_and_clean_dedup(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")

    assert len(design.changed_states) == 336
    assert len(design.pair_conditions) == 1032
    assert len(design.stress_states) == 4
    assert len(design.stress_pair_conditions) == 24
    assert len(design.feature_variants) == 93
    assert len({row.design_id for row in design.pair_conditions}) == 1032
    assert sum(row.corruption == "clean" for row in design.pair_conditions) == 24


def test_feature_variants_adapt_the_public_legacy_enumeration(frozen_config) -> None:
    variants = feature_variants(frozen_config)

    assert len(variants) == 93
    assert variants[0].key == "user_37_grid_track__dino_vitb16__592__native__37"


def test_pilot_design_is_exact(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="pilot")

    assert len(design.pair_conditions) == 44
    assert {row.pair_id for row in design.pair_conditions} == {
        "right__front",
        "left__right",
    }
    assert {row.backbone for row in design.feature_variants} == {"dinov3_vitl16_pixal3d"}
    assert len(design.feature_variants) == 30


def test_direction_is_balanced_and_pair_independent(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    states = [
        state
        for state in design.changed_states
        if state.corruption == "translation" and state.severity == 0.12
    ]

    assert [state.direction for state in sorted(states, key=lambda item: item.direction)].count(-1) == 8
    assert [state.direction for state in sorted(states, key=lambda item: item.direction)].count(1) == 8
    for state in states:
        rows = [row for row in design.pair_conditions if row.image_state_id == state.state_id]
        assert len(rows) == 3
        assert len({row.image_state_id for row in rows}) == 1


def test_identity_swap_uses_same_view_cyclic_donor(real_upper_manifest, frozen_config) -> None:
    design = build_upper_design(real_upper_manifest, frozen_config, scope="full")
    state = next(
        item
        for item in design.changed_states
        if item.shoe_id == "air_huarache_run_ultra_111449"
        and item.changed_view == "left"
        and item.corruption == "identity_swap"
    )

    assert state.donor_shoe_id == "air_max_90_nrg_dc6083_500"
    assert state.donor_view == "left"
