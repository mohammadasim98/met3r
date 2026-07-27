from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from met3r.adjacent16_stress import (
    StressTask,
    camera_pose_from_da3,
    mask_bbox_xyxy,
    run_stress_tasks,
)


def _png(path: Path, mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (12, 12), color=255 if mode == "L" else (240, 240, 240)).save(path)
    return path


def _task(root: Path, number: int = 0) -> StressTask:
    source = _png(root / f"source-{number}.png")
    mask = _png(root / f"mask-{number}.png", "L")
    render = _png(root / f"render-{number}.png")
    return StressTask(
        shoe_id="shoe", quartet_id="q0", view_id=str(number), source_path=source,
        mask_path=mask, last_render_path=render,
        output_dir=root / "inputs/stress/shoe/q0" / f"view_{number}" / "deformation",
        physical_gpu=number % 4,
    )


def test_camera_pose_uses_da3_w2c_and_world_to_last_rotation() -> None:
    rotation = np.eye(3)
    # W2C camera whose object-to-camera direction is positive X and whose
    # image right/up axes are positive Y/Z respectively.
    w2c = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    intrinsics = np.array([[250.0, 0.0, 252.0], [0.0, 250.0, 252.0], [0.0, 0.0, 1.0]])

    pose = camera_pose_from_da3(w2c, intrinsics, rotation, stage_size=504)

    assert pose.azimuth_deg == 270.0
    assert pose.elevation_deg == 0.0
    assert pose.roll_deg == 0.0
    assert np.isclose(pose.horizontal_fov_deg, np.degrees(2 * np.arctan(504 / 500)))


def test_mask_bbox_is_one_based_source_mask_bbox(tmp_path: Path) -> None:
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    path = tmp_path / "mask/2.jpg"
    path.parent.mkdir()
    Image.fromarray(mask).save(path)

    assert mask_bbox_xyxy(path) == (3, 2, 8, 6)


def test_tasks_use_one_subprocess_per_view_round_robin_and_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((list(command), kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        output = Path(command[command.index("--output-dir") + 1])
        _png(output / "deformed.png")
        _png(output / "deformed_mask.png", "L")
        selected = {
            "attempt_id": "attempt_01",
            "semantic": {"camera_preserved": True, "background_preserved": True},
            "preservation": {"passes_gate": True},
        }
        (output / "deformation.json").write_text(json.dumps({"selected_attempt_id": "attempt_01", "attempts": [selected]}))
        (output / "metrics.json").write_text(json.dumps({"selected": "attempt_01"}))
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    tasks = tuple(_task(tmp_path, number) for number in range(5))

    result = run_stress_tasks(tasks, max_parallel=4, max_outer_attempts=1)

    assert result["accepted"] == 5
    # Completion order is intentionally unconstrained by the four-worker pool;
    # each task's fixed physical assignment is the round-robin sequence.
    assert [task.physical_gpu for task in tasks] == [0, 1, 2, 3, 0]
    assert sorted(gpu for _, gpu in calls) == ["0", "0", "1", "2", "3"]
    assert len(calls) == 5
    for task in tasks:
        assert (task.output_dir / "source.png").is_file()
        assert (task.output_dir / "last_render.png").is_file()
        assert (task.output_dir / "source_mask.png").is_file()
        assert (task.output_dir / "task.json").is_file()
        assert not list(task.output_dir.parent.glob(".staging-*"))


def test_reuse_is_invalidated_by_input_prompt_or_implementation_hash(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path)
    calls: list[int] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        output = Path(command[command.index("--output-dir") + 1])
        _png(output / "deformed.png")
        _png(output / "deformed_mask.png", "L")
        selected = {"attempt_id": "a", "semantic": {"camera_preserved": True, "background_preserved": True}, "preservation": {"passes_gate": True}}
        (output / "deformation.json").write_text(json.dumps({"selected_attempt_id": "a", "attempts": [selected]}))
        (output / "metrics.json").write_text("{}")
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    run_stress_tasks((task,), max_outer_attempts=1)
    run_stress_tasks((task,), max_outer_attempts=1)
    monkeypatch.setattr("met3r.adjacent16_stress._implementation_hash", lambda: "changed")
    run_stress_tasks((task,), max_outer_attempts=1)

    assert len(calls) == 2


def test_rejected_attempts_are_archived_and_never_accepted_by_score(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path)

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        output = Path(command[command.index("--output-dir") + 1])
        _png(output / "deformed.png")
        _png(output / "deformed_mask.png", "L")
        selected = {"attempt_id": "a", "semantic": {"camera_preserved": False, "background_preserved": True}, "preservation": {"passes_gate": True}, "metrics": {"fit_score": 999.0}}
        (output / "deformation.json").write_text(json.dumps({"selected_attempt_id": "a", "attempts": [selected]}))
        (output / "metrics.json").write_text("{}")
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    result = run_stress_tasks((task,), max_outer_attempts=1)

    assert result["accepted"] == 0
    assert not task.output_dir.exists()
    archived = list((task.output_dir.parent / "rejected_attempts").rglob("deformation.json"))
    assert len(archived) == 1
