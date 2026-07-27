from __future__ import annotations

import json
from pathlib import Path

from met3r.upper_metric_worker import _clean_state_for_shoe, _load_state_index


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
