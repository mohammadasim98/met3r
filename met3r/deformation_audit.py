"""Strict, protocol-neutral parsing for deformation audit artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeformationQuality:
    status: str
    selected_attempt_id: str
    semantic_gate: bool
    preservation_gate: bool
    camera_preserved: bool
    outsole_preserved: bool
    fit_relation: str
    task_status: str


def _required_nonempty_string(
    payload: Mapping[str, object], key: str, label: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_bool(payload: Mapping[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _required_mapping(
    payload: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _selected_attempt(payload: Mapping[str, object]) -> Mapping[str, object]:
    selected = _required_nonempty_string(
        payload, "selected_attempt_id", "selected attempt ID"
    )
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("deformation attempts must be a list")
    matches = [
        item for item in attempts
        if isinstance(item, Mapping) and item.get("attempt_id") == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"selected attempt {selected!r} is not unique")
    return matches[0]


def _quality(
    deformation: Mapping[str, object], task: Mapping[str, object]
) -> DeformationQuality:
    status = _required_nonempty_string(deformation, "status", "deformation status")
    selected_attempt_id = _required_nonempty_string(
        deformation, "selected_attempt_id", "selected attempt ID"
    )
    selected = _selected_attempt(deformation)
    semantic = _required_mapping(selected, "semantic", "selected semantic audit")
    preservation = _required_mapping(selected, "preservation", "selected preservation audit")
    comparison = _required_mapping(selected, "comparison", "selected comparison audit")
    fit_relation = _required_nonempty_string(comparison, "relation", "fit relation")
    if fit_relation not in {"improved", "equivalent", "worsened"}:
        raise ValueError(f"fit relation is invalid: {fit_relation!r}")
    return DeformationQuality(
        status=status,
        selected_attempt_id=selected_attempt_id,
        semantic_gate=_required_bool(semantic, "passes_gate", "semantic gate"),
        preservation_gate=_required_bool(
            preservation, "passes_gate", "preservation gate"
        ),
        camera_preserved=_required_bool(
            semantic, "camera_preserved", "camera preservation"
        ),
        outsole_preserved=_required_bool(
            semantic, "outsole_preserved", "outsole preservation"
        ),
        fit_relation=fit_relation,
        task_status=_required_nonempty_string(task, "status", "task status"),
    )


def _strict_json_equal(left: object, right: object) -> bool:
    """Compare parsed JSON values without Python's bool/int coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            return False
        return all(_strict_json_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _strict_json_equal(first, second) for first, second in zip(left, right)
        )
    return left == right


def _validate_metrics(
    metrics: Mapping[str, object],
    deformation: Mapping[str, object],
    selected: Mapping[str, object],
) -> None:
    """Verify that metrics.json documents the selected deformation attempt."""

    selected_metrics = _required_mapping(selected, "metrics", "selected metrics")
    selected_comparison = _required_mapping(
        selected, "comparison", "selected comparison"
    )
    selected_preservation = _required_mapping(
        selected, "preservation", "selected preservation"
    )
    status = _required_nonempty_string(deformation, "status", "deformation status")
    if not _strict_json_equal(
        _required_mapping(metrics, "after", "metrics after"), selected_metrics
    ):
        raise ValueError("metrics after does not match the selected attempt")
    if not _strict_json_equal(
        _required_mapping(metrics, "comparison", "metrics comparison"), selected_comparison
    ):
        raise ValueError("metrics comparison does not match the selected attempt")
    if not _strict_json_equal(
        _required_mapping(metrics, "preservation", "metrics preservation"), selected_preservation
    ):
        raise ValueError("metrics preservation does not match the selected attempt")
    if _required_nonempty_string(metrics, "verdict", "metrics verdict") != status:
        raise ValueError("metrics verdict does not match deformation status")


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload
