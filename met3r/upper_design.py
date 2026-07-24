"""Deterministic state and pair expansion for the four-view upper experiment."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .design import feature_variants as _design_feature_variants
from .upper_data import UpperManifest


@dataclass(frozen=True)
class FeatureVariant:
    feature_track: str
    backbone: str
    input_size: int
    feature_mode: str
    resolution: int

    @property
    def key(self) -> str:
        return (
            f"{self.feature_track}__{self.backbone}__{self.input_size}__"
            f"{self.feature_mode}__{self.resolution}"
        )


@dataclass(frozen=True)
class ImageState:
    state_id: str
    track: str
    shoe_id: str
    changed_view: str | None
    corruption: str
    severity: float
    direction: int
    seed: int
    donor_shoe_id: str | None = None
    donor_view: str | None = None


@dataclass(frozen=True)
class PairCondition:
    design_id: str
    track: str
    shoe_id: str
    pair_id: str
    first_view: str
    second_view: str
    changed_view: str | None
    image_state_id: str
    corruption: str
    severity: float


@dataclass(frozen=True)
class UpperDesign:
    image_states: tuple[ImageState, ...]
    pair_conditions: tuple[PairCondition, ...]
    stress_states: tuple[ImageState, ...]
    stress_pair_conditions: tuple[PairCondition, ...]
    feature_variants: tuple[FeatureVariant, ...]

    @property
    def changed_states(self) -> tuple[ImageState, ...]:
        return tuple(state for state in self.image_states if state.corruption != "clean")


@dataclass(frozen=True)
class MaterializedState:
    state_id: str
    state_dir: Path
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    masks: np.ndarray
    metadata_path: Path


def _state_seed(shoe_id: str, view_id: str, corruption: str, severity: float) -> int:
    payload = f"42|{shoe_id}|{view_id}|{corruption}|{severity:.6f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def feature_variants(config: Mapping[str, object]) -> tuple[FeatureVariant, ...]:
    """Adapt the generic design records to immutable upper-experiment variants."""

    return tuple(FeatureVariant(**item) for item in _design_feature_variants(config))


def _pairs(views: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(itertools.combinations(views, 2))


def _state_id(
    shoe_id: str,
    changed_view: str | None,
    corruption: str,
    severity: float,
    direction: int,
) -> str:
    if changed_view is None:
        return f"controlled__{shoe_id}__clean"
    return (
        f"controlled__{shoe_id}__{changed_view}__{corruption}__"
        f"{severity:.6f}__{direction:+d}"
    )


def _required_corruptions(config: Mapping[str, object]) -> Mapping[str, Sequence[object]]:
    corruptions = config.get("corruptions")
    if not isinstance(corruptions, Mapping):
        raise ValueError("config requires corruptions")
    for name, severities in corruptions.items():
        if not isinstance(name, str) or not isinstance(severities, Sequence):
            raise ValueError("corruptions must map names to severity lists")
        if name in {"clean", "part_transfer"}:
            raise ValueError(f"upper design does not support corruption {name!r}")
        if any(not isinstance(value, (int, float)) for value in severities):
            raise ValueError("corruption severities must be numeric")
    return corruptions


def _pilot_config(config: Mapping[str, object]) -> Mapping[str, object]:
    pilot = config.get("pilot")
    if not isinstance(pilot, Mapping):
        raise ValueError("config requires a pilot object")
    return pilot


def _controlled_states(
    manifest: UpperManifest,
    config: Mapping[str, object],
    *,
    scope: Literal["pilot", "full"],
) -> tuple[ImageState, ...]:
    full_corruptions = _required_corruptions(config)
    shoes = tuple(manifest.shoes)
    views_by_shoe = {shoe.shoe_id: tuple(view.view_id for view in shoe.views) for shoe in shoes}
    if not shoes or any(len(views) != 4 for views in views_by_shoe.values()):
        raise ValueError("upper manifest requires four views per shoe")

    chosen_shoe_ids = {shoe.shoe_id for shoe in shoes}
    chosen_pairs: set[str] | None = None
    corruptions: Mapping[str, Sequence[object]] = full_corruptions
    if scope == "pilot":
        pilot = _pilot_config(config)
        raw_shoes = pilot.get("shoes")
        raw_pairs = pilot.get("pairs")
        raw_corruptions = pilot.get("corruptions")
        if (
            not isinstance(raw_shoes, Sequence)
            or not isinstance(raw_pairs, Sequence)
            or not isinstance(raw_corruptions, Mapping)
            or not all(isinstance(value, str) for value in raw_shoes)
            or not all(isinstance(value, str) for value in raw_pairs)
        ):
            raise ValueError("pilot requires shoe, pair, and corruption selections")
        chosen_shoe_ids = set(raw_shoes)
        if not chosen_shoe_ids <= set(views_by_shoe):
            raise ValueError("pilot contains an unknown shoe")
        chosen_pairs = set(raw_pairs)
        corruptions = raw_corruptions
        if not set(corruptions) <= set(full_corruptions):
            raise ValueError("pilot contains an unknown corruption")
        for name, severities in corruptions.items():
            if not isinstance(name, str) or not isinstance(severities, Sequence):
                raise ValueError("pilot corruptions must map names to severity lists")
            if any(not isinstance(value, (int, float)) for value in severities):
                raise ValueError("pilot corruption severities must be numeric")
    elif scope != "full":
        raise ValueError("scope must be 'pilot' or 'full'")

    all_view_states = tuple(
        (shoe.shoe_id, view.view_id)
        for shoe in sorted(shoes, key=lambda item: item.shoe_id)
        for view in sorted(shoe.views, key=lambda item: item.view_id)
    )
    donors = {
        shoe.shoe_id: shoes[(index + 1) % len(shoes)].shoe_id
        for index, shoe in enumerate(shoes)
    }
    states: list[ImageState] = []
    for shoe in shoes:
        if shoe.shoe_id not in chosen_shoe_ids:
            continue
        states.append(
            ImageState(
                state_id=_state_id(shoe.shoe_id, None, "clean", 0.0, 1),
                track="controlled",
                shoe_id=shoe.shoe_id,
                changed_view=None,
                corruption="clean",
                severity=0.0,
                direction=1,
                seed=0,
            )
        )

    active_views = {
        (shoe_id, view_id)
        for shoe_id in chosen_shoe_ids
        for view_id in views_by_shoe[shoe_id]
        if chosen_pairs is None
        or any(view_id in pair_id.split("__") for pair_id in chosen_pairs)
    }
    for corruption, raw_severities in corruptions.items():
        if not isinstance(corruption, str):
            raise ValueError("corruption names must be strings")
        for raw_severity in raw_severities:
            severity = float(raw_severity)
            for index, (shoe_id, view_id) in enumerate(all_view_states):
                if (shoe_id, view_id) not in active_views:
                    continue
                direction = -1 if index % 2 == 0 else 1
                donor_shoe_id = donors[shoe_id] if corruption == "identity_swap" else None
                states.append(
                    ImageState(
                        state_id=_state_id(
                            shoe_id, view_id, corruption, severity, direction
                        ),
                        track="controlled",
                        shoe_id=shoe_id,
                        changed_view=view_id,
                        corruption=corruption,
                        severity=severity,
                        direction=direction,
                        seed=_state_seed(shoe_id, view_id, corruption, severity),
                        donor_shoe_id=donor_shoe_id,
                        donor_view=view_id if donor_shoe_id is not None else None,
                    )
                )
    return tuple(states)


def _pair_conditions(
    states: Sequence[ImageState],
    manifest: UpperManifest,
    *,
    selected_pairs: set[str] | None,
    prefix: str,
) -> tuple[PairCondition, ...]:
    views_by_shoe = {shoe.shoe_id: tuple(view.view_id for view in shoe.views) for shoe in manifest.shoes}
    rows: list[PairCondition] = []
    for state in states:
        for first_view, second_view in _pairs(views_by_shoe[state.shoe_id]):
            pair_id = f"{first_view}__{second_view}"
            if selected_pairs is not None and pair_id not in selected_pairs:
                continue
            if state.changed_view is not None and state.changed_view not in {
                first_view,
                second_view,
            }:
                continue
            rows.append(
                PairCondition(
                    design_id=f"{prefix}{len(rows) + 1:08d}",
                    track=state.track,
                    shoe_id=state.shoe_id,
                    pair_id=pair_id,
                    first_view=first_view,
                    second_view=second_view,
                    changed_view=state.changed_view,
                    image_state_id=state.state_id,
                    corruption=state.corruption,
                    severity=state.severity,
                )
            )
    return tuple(rows)


def _stress_design(manifest: UpperManifest) -> tuple[tuple[ImageState, ...], tuple[PairCondition, ...]]:
    states = tuple(
        ImageState(
            state_id=f"stress__{shoe.shoe_id}__deformed",
            track="stress",
            shoe_id=shoe.shoe_id,
            changed_view=None,
            corruption="deformed",
            severity=1.0,
            direction=1,
            seed=_state_seed(shoe.shoe_id, "all", "deformed", 1.0),
        )
        for shoe in manifest.shoes
    )
    return states, _pair_conditions(states, manifest, selected_pairs=None, prefix="S")


def build_upper_design(
    manifest: UpperManifest,
    config: Mapping[str, object],
    scope: Literal["pilot", "full"],
) -> UpperDesign:
    """Expand immutable, reusable image states and their pair-condition cells."""

    variants = feature_variants(config)
    states = _controlled_states(manifest, config, scope=scope)
    selected_pairs: set[str] | None = None
    if scope == "pilot":
        pilot = _pilot_config(config)
        raw_pairs = pilot["pairs"]
        assert isinstance(raw_pairs, Sequence)
        selected_pairs = {str(value) for value in raw_pairs}
        raw_backbones = pilot.get("backbones")
        if not isinstance(raw_backbones, Sequence) or not all(
            isinstance(value, str) for value in raw_backbones
        ):
            raise ValueError("pilot requires backbone selections")
        variants = tuple(item for item in variants if item.backbone in set(raw_backbones))
    pairs = _pair_conditions(states, manifest, selected_pairs=selected_pairs, prefix="D")
    stress_states: tuple[ImageState, ...] = ()
    stress_pairs: tuple[PairCondition, ...] = ()
    if scope == "full":
        stress_states, stress_pairs = _stress_design(manifest)
    return UpperDesign(states, pairs, stress_states, stress_pairs, variants)
