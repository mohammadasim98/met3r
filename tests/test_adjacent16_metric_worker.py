from __future__ import annotations

import json
from pathlib import Path

from met3r.upper_metric_worker import (
    _StateRecord,
    _clean_state_for_shoe,
    _load_state_index,
    _validate_adjacent_pair_contract,
)


def test_state_index_and_clean_lookup_keep_quartets_separate(tmp_path: Path) -> None:
    """Catch a clean cache lookup collapsing distinct ordered four-view quartets."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    records = [
        {
            "state_id": "clean-q0", "track": "controlled", "shoe_id": "shoe",
            "quartet_id": "q0", "view_ids": ["1", "2", "3", "4"],
            "changed_view": None, "corruption": "clean", "severity": 0.0,
        },
        {
            "state_id": "clean-q1", "track": "controlled", "shoe_id": "shoe",
            "quartet_id": "q1", "view_ids": ["14", "15", "16", "17"],
            "changed_view": None, "corruption": "clean", "severity": 0.0,
        },
    ]
    (manifests / "image_states.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    states = _load_state_index(tmp_path)

    assert states["clean-q1"].view_ids == ("14", "15", "16", "17")
    assert _clean_state_for_shoe(states, "shoe", "q1").state_id == "clean-q1"


def test_adjacent_pair_contract_requires_quartet_and_measured_baseline() -> None:
    """Catch null/default baseline metadata being emitted for adjacent rows."""

    state = _StateRecord(
        "q3-clean", "controlled", "shoe", "q3", ("40", "41", "42", "0"),
        None, "clean", 0.0, Path("/states/q3-clean"),
    )
    pair = {
        "quartet_id": "q3", "first_view": "42", "second_view": "0",
        "baseline_steps": 1, "angular_separation_deg": 7.5,
    }

    _validate_adjacent_pair_contract(pair, state)

    for field, invalid in (("quartet_id", "q0"), ("baseline_steps", None), ("angular_separation_deg", float("nan"))):
        invalid_pair = {**pair, field: invalid}
        try:
            _validate_adjacent_pair_contract(invalid_pair, state)
        except ValueError:
            continue
        raise AssertionError(f"invalid adjacent {field} was accepted")


def test_adjacent_contract_rejects_same_quartet_with_reordered_views() -> None:
    """Catch pairing clean geometry order with a reordered changed-state tensor."""

    clean = _StateRecord(
        "clean", "controlled", "shoe", "q3", ("40", "41", "42", "0"),
        None, "clean", 0.0, Path("/states/clean"),
    )
    changed = _StateRecord(
        "changed", "controlled", "shoe", "q3", ("0", "40", "41", "42"),
        "0", "jpeg", 0.5, Path("/states/changed"),
    )

    try:
        _validate_adjacent_pair_contract({}, changed, clean_state=clean)
    except ValueError as error:
        assert "view order" in str(error)
    else:
        raise AssertionError("reordered state was accepted")
