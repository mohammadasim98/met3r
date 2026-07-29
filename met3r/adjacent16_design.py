"""Immutable quartet-aware state and pair enumeration for adjacent16."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .adjacent16_contract import Adjacent16Contract
from .adjacent16_data import Adjacent16Manifest, AdjacentViewRecord, pair_baseline
from .upper_design import FeatureVariant


@dataclass(frozen=True)
class AdjacentImageState:
    state_id: str
    track: str
    shoe_id: str
    quartet_id: str
    view_ids: tuple[str, str, str, str]
    changed_view: str | None
    corruption: str
    severity: float
    direction: int
    seed: int
    donor_shoe_id: str | None = None
    donor_view: str | None = None


@dataclass(frozen=True)
class AdjacentPairCondition:
    design_id: str
    track: str
    shoe_id: str
    quartet_id: str
    pair_id: str
    first_view: str
    second_view: str
    baseline_steps: int
    angular_separation_deg: float
    changed_view: str | None
    image_state_id: str
    corruption: str
    severity: float


@dataclass(frozen=True)
class Adjacent16Design:
    controlled_states: tuple[AdjacentImageState, ...]
    stress_states: tuple[AdjacentImageState, ...]
    controlled_conditions: tuple[AdjacentPairCondition, ...]
    stress_conditions: tuple[AdjacentPairCondition, ...]
    feature_variants: tuple[FeatureVariant, ...]
    row_counts: Mapping[str, int]


def _state_seed(
    shoe_id: str,
    quartet_id: str,
    view_id: str,
    corruption: str,
    severity: float,
) -> int:
    payload = (
        f"42|{shoe_id}|{quartet_id}|{view_id}|{corruption}|{severity:.6f}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _changed_state_id(
    shoe_id: str,
    quartet_id: str,
    view_id: str,
    corruption: str,
    severity: float,
    direction: int,
) -> str:
    return (
        f"controlled__{shoe_id}__{quartet_id}__view_{view_id}__{corruption}__"
        f"{severity:.6f}__{direction:+d}"
    )


def _views_by_quartet(
    manifest: Adjacent16Manifest,
    contract: Adjacent16Contract,
) -> Mapping[tuple[str, str], tuple[AdjacentViewRecord, ...]]:
    expected = {quartet.quartet_id: quartet.view_ids for quartet in contract.quartets}
    records: dict[tuple[str, str], tuple[AdjacentViewRecord, ...]] = {}
    for shoe in manifest.shoes:
        grouped: dict[str, list[AdjacentViewRecord]] = {}
        for view in shoe.views:
            grouped.setdefault(view.quartet_id, []).append(view)
        if set(grouped) != set(expected):
            raise ValueError("manifest quartets differ from the frozen contract")
        for quartet in contract.quartets:
            views = tuple(grouped[quartet.quartet_id])
            if tuple(view.view_id for view in views) != quartet.view_ids:
                raise ValueError("manifest quartet views differ from the frozen contract")
            records[shoe.shoe_id, quartet.quartet_id] = views
    return MappingProxyType(records)


def _pair_rows(
    state: AdjacentImageState,
    views: Sequence[AdjacentViewRecord],
    *,
    start: int,
    prefix: str,
) -> tuple[tuple[AdjacentPairCondition, ...], int]:
    rows: list[AdjacentPairCondition] = []
    for first, second in itertools.combinations(views, 2):
        if state.changed_view is not None and state.changed_view not in {
            first.view_id,
            second.view_id,
        }:
            continue
        steps, angle = pair_baseline(first, second)
        rows.append(
            AdjacentPairCondition(
                design_id=f"{prefix}{start + len(rows) + 1:08d}",
                track=state.track,
                shoe_id=state.shoe_id,
                quartet_id=state.quartet_id,
                pair_id=(
                    f"{state.shoe_id}__{state.quartet_id}__"
                    f"{first.view_id}__{second.view_id}"
                ),
                first_view=first.view_id,
                second_view=second.view_id,
                baseline_steps=steps,
                angular_separation_deg=angle,
                changed_view=state.changed_view,
                image_state_id=state.state_id,
                corruption=state.corruption,
                severity=state.severity,
            )
        )
    return tuple(rows), start + len(rows)


def _corruptions(contract: Adjacent16Contract) -> Mapping[str, Sequence[object]]:
    corruptions = contract.raw.get("corruptions")
    if not isinstance(corruptions, Mapping):
        raise ValueError("adjacent16 contract requires corruptions")
    return corruptions


def _row_counts(
    contract: Adjacent16Contract,
    controlled_conditions: Sequence[AdjacentPairCondition],
    stress_conditions: Sequence[AdjacentPairCondition],
) -> Mapping[str, int]:
    geometry = contract.raw.get("geometry_calibration")
    if not isinstance(geometry, Mapping) or any(
        not isinstance(values, Sequence) for values in geometry.values()
    ):
        raise ValueError("adjacent16 contract requires geometry calibration magnitudes")
    counts = {
        "controlled": len(controlled_conditions) * len(contract.feature_variants),
        "geometry_calibration": (
            len(stress_conditions)
            * 2
            * sum(len(values) for values in geometry.values())
            * len(contract.geometry_calibration_variant_keys)
        ),
        "stress": len(stress_conditions) * len(contract.feature_variants),
    }
    expected = {name: contract.expected_rows[name] for name in counts}
    if counts != expected:
        raise ValueError("adjacent16 design row counts differ from the frozen contract")
    return MappingProxyType(counts)


def build_adjacent16_design(
    manifest: Adjacent16Manifest,
    contract: Adjacent16Contract,
) -> Adjacent16Design:
    """Expand frozen quartet states and their measured-angle pair conditions."""

    if len(manifest.shoes) != len(contract.raw["shoes"]):
        raise ValueError("manifest shoes differ from the frozen contract")
    by_quartet = _views_by_quartet(manifest, contract)
    donor_shoes = {
        shoe.shoe_id: manifest.shoes[(index + 1) % len(manifest.shoes)].shoe_id
        for index, shoe in enumerate(manifest.shoes)
    }
    controlled_states: list[AdjacentImageState] = []
    controlled_conditions: list[AdjacentPairCondition] = []
    condition_count = 0
    for shoe in manifest.shoes:
        for quartet in contract.quartets:
            views = by_quartet[shoe.shoe_id, quartet.quartet_id]
            view_ids = quartet.view_ids
            clean = AdjacentImageState(
                state_id=f"controlled__{shoe.shoe_id}__{quartet.quartet_id}__clean",
                track="controlled",
                shoe_id=shoe.shoe_id,
                quartet_id=quartet.quartet_id,
                view_ids=view_ids,
                changed_view=None,
                corruption="clean",
                severity=0.0,
                direction=1,
                seed=0,
            )
            controlled_states.append(clean)
            rows, condition_count = _pair_rows(
                clean, views, start=condition_count, prefix="C"
            )
            controlled_conditions.extend(rows)
            for corruption, severities in _corruptions(contract).items():
                for raw_severity in severities:
                    severity = float(raw_severity)
                    for view_id in view_ids:
                        seed = _state_seed(
                            shoe.shoe_id, quartet.quartet_id, view_id, corruption, severity
                        )
                        direction = 1 if seed % 2 else -1
                        donor_shoe_id = (
                            donor_shoes[shoe.shoe_id]
                            if corruption == "identity_swap"
                            else None
                        )
                        state = AdjacentImageState(
                            state_id=_changed_state_id(
                                shoe.shoe_id,
                                quartet.quartet_id,
                                view_id,
                                corruption,
                                severity,
                                direction,
                            ),
                            track="controlled",
                            shoe_id=shoe.shoe_id,
                            quartet_id=quartet.quartet_id,
                            view_ids=view_ids,
                            changed_view=view_id,
                            corruption=corruption,
                            severity=severity,
                            direction=direction,
                            seed=seed,
                            donor_shoe_id=donor_shoe_id,
                            donor_view=view_id if donor_shoe_id is not None else None,
                        )
                        controlled_states.append(state)
                        rows, condition_count = _pair_rows(
                            state, views, start=condition_count, prefix="C"
                        )
                        controlled_conditions.extend(rows)

    stress_states: list[AdjacentImageState] = []
    stress_conditions: list[AdjacentPairCondition] = []
    condition_count = 0
    for shoe in manifest.shoes:
        for quartet in contract.quartets:
            state = AdjacentImageState(
                state_id=f"stress__{shoe.shoe_id}__{quartet.quartet_id}__deformed",
                track="stress",
                shoe_id=shoe.shoe_id,
                quartet_id=quartet.quartet_id,
                view_ids=quartet.view_ids,
                changed_view=None,
                corruption="deformed",
                severity=1.0,
                direction=1,
                seed=_state_seed(
                    shoe.shoe_id, quartet.quartet_id, "all", "deformed", 1.0
                ),
            )
            stress_states.append(state)
            rows, condition_count = _pair_rows(
                state,
                by_quartet[shoe.shoe_id, quartet.quartet_id],
                start=condition_count,
                prefix="S",
            )
            stress_conditions.extend(rows)

    counts = _row_counts(contract, controlled_conditions, stress_conditions)
    return Adjacent16Design(
        controlled_states=tuple(controlled_states),
        stress_states=tuple(stress_states),
        controlled_conditions=tuple(controlled_conditions),
        stress_conditions=tuple(stress_conditions),
        feature_variants=contract.feature_variants,
        row_counts=counts,
    )
