from __future__ import annotations

from pathlib import Path

from met3r.provenance import repository_state, runtime_environment


def test_runtime_environment_records_relevant_versions() -> None:
    environment = runtime_environment()

    assert environment["python"]
    assert "torch" in environment["packages"]
    assert "transformers" in environment["packages"]


def test_repository_state_records_commit_and_dirty_flag() -> None:
    state = repository_state(Path(__file__).parents[1])

    assert len(state["revision"]) == 40
    assert isinstance(state["dirty"], bool)
    if state["dirty"]:
        assert len(state["dirty_fingerprint"]) == 64
    else:
        assert state["dirty_fingerprint"] is None
