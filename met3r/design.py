"""Deterministic expansion of the known-camera experiment design matrix."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence


def feature_variants(
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    tracks = config.get("feature_tracks")
    modes = config.get("upsampling_modes")
    if not isinstance(tracks, Mapping) or not isinstance(modes, Sequence):
        raise ValueError("config requires feature_tracks and upsampling_modes")
    variants: list[dict[str, object]] = []
    for track_name, raw_track in tracks.items():
        if not isinstance(raw_track, Mapping):
            raise ValueError(f"feature track {track_name!r} must be an object")
        common_grids = raw_track.get("common_evaluation_grids")
        for backbone, raw_spec in raw_track.items():
            if backbone == "common_evaluation_grids":
                continue
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"backbone {backbone!r} must be an object")
            input_size = raw_spec.get("input_size")
            native_grid = raw_spec.get("native_grid")
            if not isinstance(input_size, int) or not isinstance(native_grid, int):
                raise ValueError(
                    f"backbone {backbone!r} requires integer input_size/native_grid"
                )
            variants.append(
                {
                    "feature_track": track_name,
                    "backbone": backbone,
                    "input_size": input_size,
                    "feature_mode": "native",
                    "resolution": native_grid,
                }
            )
            raw_targets = (
                raw_spec.get("naf_targets")
                if raw_spec.get("naf_targets") is not None
                else common_grids
            )
            if raw_targets is None:
                continue
            if not isinstance(raw_targets, Sequence):
                raise ValueError(f"feature targets for {backbone!r} must be a list")
            for mode in modes:
                if not isinstance(mode, str):
                    raise ValueError("upsampling modes must be strings")
                for resolution in raw_targets:
                    if not isinstance(resolution, int):
                        raise ValueError("feature resolutions must be integers")
                    if mode.startswith("naf_") and resolution < native_grid:
                        continue
                    variants.append(
                        {
                            "feature_track": track_name,
                            "backbone": backbone,
                            "input_size": input_size,
                            "feature_mode": mode,
                            "resolution": resolution,
                        }
                    )
    return variants


def build_design_rows(
    config: Mapping[str, object],
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    """Expand every shoe/pair × feature × corruption cell into one audit row."""

    corruptions = config.get("corruptions")
    shoes = manifest.get("shoes")
    execution = config.get("execution", {})
    if not isinstance(corruptions, Mapping):
        raise ValueError("config requires a corruptions object")
    if not isinstance(shoes, Sequence):
        raise ValueError("manifest requires a shoes list")
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object when provided")
    corruption_seed = execution.get("corruption_seed", 42)
    if not isinstance(corruption_seed, int):
        raise ValueError("execution.corruption_seed must be an integer")
    variants = feature_variants(config)
    cells: list[dict[str, object]] = []
    for raw_shoe in shoes:
        if not isinstance(raw_shoe, Mapping):
            raise ValueError("each shoe manifest entry must be an object")
        shoe_id = raw_shoe.get("shoe_id")
        views = raw_shoe.get("views")
        if not isinstance(shoe_id, str) or not isinstance(views, Sequence):
            raise ValueError("each shoe requires shoe_id and views")
        for (first_index, first), (second_index, second) in itertools.combinations(
            enumerate(views),
            2,
        ):
            if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                raise ValueError("each view manifest entry must be an object")
            first_view = first.get("view_id")
            second_view = second.get("view_id")
            first_group = first.get("camera_group")
            second_group = second.get("camera_group")
            if not all(
                isinstance(value, str)
                for value in (first_view, second_view, first_group, second_group)
            ):
                raise ValueError("each view requires view_id and camera_group")
            same_group = first_group == second_group
            pair = {
                "shoe_id": shoe_id,
                "pair_id": f"{first_view}__{second_view}",
                "first_view": first_view,
                "second_view": second_view,
                "first_index": first_index,
                "second_index": second_index,
                "changed_view_index": second_index,
                "pair_scope": "intra_group" if same_group else "cross_group",
                "camera_group": first_group if same_group else "cross_group",
                "analysis_tier": "primary" if same_group else "sensitivity_only",
            }
            for variant in variants:
                for corruption, severities in corruptions.items():
                    if not isinstance(corruption, str) or not isinstance(
                        severities, Sequence
                    ):
                        raise ValueError("corruptions must map names to severity lists")
                    for severity in severities:
                        if not isinstance(severity, (int, float)):
                            raise ValueError("corruption severities must be numeric")
                        cells.append(
                            {
                                **pair,
                                **variant,
                                "corruption": corruption,
                                "severity": float(severity),
                                "corruption_seed": corruption_seed,
                                "design_role": "audit_only",
                                "job_state": "planned",
                            }
                        )
    return [
        {"design_id": f"D{index:08d}", **row}
        for index, row in enumerate(cells, start=1)
    ]
