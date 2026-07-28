from __future__ import annotations

import json
import multiprocessing
import os
import sys
import threading
import time
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from met3r.adjacent16_stress import (
    StressTask,
    camera_pose_from_da3,
    mask_bbox_xyxy,
    run_stress_tasks,
)


@pytest.fixture(autouse=True)
def _stable_bytesize_fingerprint(monkeypatch):
    """Avoid hashing the intentionally huge dirty ByteSize checkout in unit fakes."""

    monkeypatch.setattr("met3r.adjacent16_stress._bytesize_fingerprint", lambda: {"state": "fixture"})


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


def test_real_cardinal_camera_parity_for_all_four_shoes() -> None:
    from met3r.adjacent16_stress import _validate_cardinal_cameras

    root = Path("/root/dev/ByteSize/Reconstruction")
    camera_ids = {
        "air_huarache_run_ultra_111449": "AIR HUARACHE RUN ULTRA(111449)",
        "air_max_90_nrg_dc6083_500": "AIR MAX 90 NRG(DC6083-500)",
        "air_max_90_surplus_cq7743_300": "AIR MAX 90 SURPLUS(CQ7743-300)",
        "air_tuned_max_cv6984_001": "AIR TUNED MAX(CV6984-001)",
    }
    for shoe_id, camera_id in camera_ids.items():
        camera_root = root / "six_view_experiment_result_precomputed/multiview" / shoe_id / "precomputed_icp"
        rotation = np.asarray(json.loads((camera_root / "multiview_camera.json").read_text())["world_to_last_rotation"])
        _validate_cardinal_cameras(root, camera_root, camera_id, rotation)


def test_reference_renderer_uses_selected_pose_bbox_and_atomic_output(tmp_path: Path, monkeypatch) -> None:
    from met3r.adjacent16_contract import load_adjacent16_contract
    from met3r.adjacent16_data import Adjacent16Manifest, AdjacentShoeRecord, AdjacentViewRecord
    from met3r.adjacent16_stress import render_stress_references

    config = Path(__file__).resolve().parents[1] / "configs/adjacent16_core.json"
    contract = replace(load_adjacent16_contract(config), reconstruction_root=tmp_path, stress_root=tmp_path / "stress")
    source, mask = _png(tmp_path / "source.png"), _png(tmp_path / "mask/2.jpg", "L")
    view = AdjacentViewRecord("shoe", "camera", "q0", "1", 0, source, mask, None, None, None, None, None, 1.0, 0.0, "a", "b", None, None, None)
    manifest = Adjacent16Manifest(tmp_path, contract.stress_root, (AdjacentShoeRecord("shoe", "camera", (view,)),))
    precomputed = tmp_path / "six_view_experiment_result_precomputed/multiview/shoe/precomputed_icp"
    precomputed.mkdir(parents=True)
    mesh = _png(tmp_path / "mesh.png")
    (precomputed / "multiview_camera.json").write_text(json.dumps({"world_to_last_rotation": np.eye(3).tolist(), "last_mesh": str(mesh)}))
    monkeypatch.setattr("met3r.adjacent16_stress._validate_cardinal_cameras", lambda *args: None)
    monkeypatch.setattr("met3r.adjacent16_stress._selected_camera_arrays", lambda *args: (np.broadcast_to(np.eye(3), (48, 3, 3)).copy(), np.broadcast_to(np.c_[np.eye(3), np.zeros(3)], (48, 3, 4)).copy()))
    seen = {}
    def fake_render(mesh_path, pose, output, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        _png(output)
    monkeypatch.setattr("met3r.adjacent16_stress._render_last", fake_render)

    outputs = render_stress_references(contract, manifest)

    assert outputs == (contract.stress_root / "shoe/q0/view_1/deformation/last_render.png",)
    assert seen["target_bbox"] == (0, 0, 12, 12)


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


def test_stress_child_launch_does_not_require_fork(
    tmp_path: Path, monkeypatch
) -> None:
    """Native renderer threads make fork unsafe; the child must use spawn."""

    task = _task(tmp_path)
    child = tmp_path / "accepted_child.py"
    child.write_text(
        "\n".join(
            (
                "import json, os, sam3, shutil, sys",
                "from pathlib import Path",
                "assert os.environ['SHOE_DEFORM_CODEX_SANDBOX'] == 'danger-full-access'",
                "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])",
                "shutil.copy2(output / 'source.png', output / 'deformed.png')",
                "shutil.copy2(output / 'source_mask.png', output / 'deformed_mask.png')",
                "selected = {'attempt_id': 'a', 'semantic': {'camera_preserved': True, "
                "'background_preserved': True}, 'preservation': {'passes_gate': True}}",
                "(output / 'deformation.json').write_text(json.dumps("
                "{'selected_attempt_id': 'a', 'attempts': [selected]}))",
                "(output / 'metrics.json').write_text('{}')",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "met3r.adjacent16_stress._deformation_command",
        lambda inputs, staging: [
            sys.executable,
            str(child),
            "--output-dir",
            str(staging),
        ],
    )

    def unsafe_fork(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("fork is unsafe after native renderer threads start")

    monkeypatch.setattr(subprocess, "_fork_exec", unsafe_fork)

    result = run_stress_tasks((task,), max_outer_attempts=1)

    assert result["accepted"] == 1
    assert (task.output_dir / "deformed.png").is_file()


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


def test_reuse_requires_unchanged_inputs_prompt_implementation_bytesize_and_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path)
    calls: list[int] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        output = Path(command[command.index("--output-dir") + 1])
        _accepted_artifacts(output)
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    run_stress_tasks((task,), max_outer_attempts=1)
    run_stress_tasks((task,), max_outer_attempts=1)
    task.source_path.write_bytes(b"changed-source")
    run_stress_tasks((task,), max_outer_attempts=1)
    task.last_render_path.write_bytes(b"changed-render")
    run_stress_tasks((task,), max_outer_attempts=1)
    monkeypatch.setattr("met3r.adjacent16_stress._PROMPT_VERSION", "changed-prompt")
    run_stress_tasks((task,), max_outer_attempts=1)
    monkeypatch.setattr("met3r.adjacent16_stress._implementation_hash", lambda: "changed-implementation")
    run_stress_tasks((task,), max_outer_attempts=1)
    monkeypatch.setattr("met3r.adjacent16_stress._bytesize_fingerprint", lambda: {"state": "changed"})
    run_stress_tasks((task,), max_outer_attempts=1)
    (task.output_dir / "deformed.png").write_bytes(b"tampered-output")
    run_stress_tasks((task,), max_outer_attempts=1)

    assert len(calls) == 7


def test_command_uses_immutable_staged_inputs(tmp_path: Path, monkeypatch) -> None:
    task = _task(tmp_path)
    command: list[str] = []

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        command.extend(args)
        output = Path(args[args.index("--output-dir") + 1])
        source = Path(args[args.index("--image") + 1])
        last_render = Path(args[args.index("--last-render") + 1])
        assert source.name == "source.png"
        assert last_render.name == "last_render.png"
        assert source.parent == last_render.parent
        assert not source.is_relative_to(output)
        assert not last_render.is_relative_to(output)
        _accepted_artifacts(output)
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    assert run_stress_tasks((task,), max_outer_attempts=1)["accepted"] == 1
    assert command


def test_failed_replacement_preserves_canonical_and_archives_rejected_record(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path)
    task.output_dir.mkdir(parents=True)
    (task.output_dir / "sentinel.txt").write_text("old canonical")

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]))
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    monkeypatch.setattr("met3r.adjacent16_stress._promote", lambda *args: (_ for _ in ()).throw(OSError("promotion failed")))
    result = run_stress_tasks((task,), max_outer_attempts=1)

    assert result["accepted"] == 0
    assert (task.output_dir / "sentinel.txt").read_text() == "old canonical"
    record = json.loads(next((task.output_dir.parent / "rejected_attempts").rglob("task.json")).read_text())
    assert record["status"] == "rejected"
    assert record["reason"] == "promotion failed"


def test_claim_and_gpu_semaphore_prevent_same_output_and_gpu_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    first = _task(tmp_path, 0)
    second = _task(tmp_path, 4)
    second = StressTask(**{**first.__dict__, "view_id": "second", "physical_gpu": 0})
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.08)
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]))
        with lock:
            active -= 1
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    # Same-output requests are serialized across independent invocations.
    threads = [threading.Thread(target=run_stress_tasks, args=((first,),), kwargs={"max_outer_attempts": 1}) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert maximum == 1


def test_gpu_queues_allow_at_most_one_subprocess_per_physical_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    active: dict[str, int] = {}
    maximum: dict[str, int] = {}
    lock = threading.Lock()

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            maximum[gpu] = max(maximum.get(gpu, 0), active[gpu])
        time.sleep(0.04)
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]))
        with lock:
            active[gpu] -= 1
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    tasks = tuple(_task(tmp_path, index) for index in range(8))
    assert run_stress_tasks(tasks, max_parallel=4, max_outer_attempts=1)["accepted"] == 8
    assert maximum == {"0": 1, "1": 1, "2": 1, "3": 1}


def test_machine_gpu_claim_serializes_independent_invocations(tmp_path: Path, monkeypatch) -> None:
    first = _task(tmp_path / "first", 0)
    second = _task(tmp_path / "second", 0)
    active = 0
    maximum = 0
    state_lock = threading.Lock()

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.08)
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]))
        with state_lock:
            active -= 1
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    threads = [threading.Thread(target=run_stress_tasks, args=((task,),), kwargs={"max_outer_attempts": 1}) for task in (first, second)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert maximum == 1


def test_machine_gpu_claim_serializes_distinct_processes(tmp_path: Path) -> None:
    lock_root = tmp_path / "gpu-locks"
    queue: multiprocessing.Queue[tuple[float, float]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(target=_hold_gpu_claim, args=(lock_root, queue))
        for _ in range(2)
    ]
    for process in processes: process.start()
    intervals = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    first, second = sorted(intervals)
    assert first[1] <= second[0]


def test_bytesize_fingerprint_uses_git_toplevel_and_all_repo_changes(tmp_path: Path, monkeypatch) -> None:
    import met3r.adjacent16_stress as stress
    original = stress._bytesize_fingerprint_impl
    repository = tmp_path / "repo"
    reconstruction = repository / "Reconstruction"
    (reconstruction / "last_view_match").mkdir(parents=True)
    (reconstruction / "shoe_deform").mkdir()
    (reconstruction / "last_view_match/render.py").write_text("render = 1\n")
    (reconstruction / "shoe_deform/__init__.py").write_text("shoe = 1\n")
    tracked = reconstruction / "tracked.txt"
    tracked.write_text("base\n")
    for args in (("init",), ("config", "user.email", "test@example.invalid"), ("config", "user.name", "Test"), ("add", "."), ("commit", "-m", "base")):
        subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True)
    monkeypatch.setattr(stress, "_BYTESIZE_RECONSTRUCTION", reconstruction)

    baseline = original()
    tracked.write_text("unstaged\n")
    unstaged = original()
    subprocess.run(["git", "add", "Reconstruction/tracked.txt"], cwd=repository, check=True)
    staged = original()
    parent_untracked = repository / "parent-untracked.txt"
    parent_untracked.write_text("one\n")
    untracked_one = original()
    parent_untracked.write_text("two\n")
    untracked_two = original()

    assert baseline["repository"] == str(repository.resolve())
    assert baseline != unstaged != staged
    assert untracked_one != untracked_two
    assert untracked_two["untracked_file_sha256"]["parent-untracked.txt"]


def test_exchange_commits_old_accepted_directory_without_post_exchange_move(
    tmp_path: Path, monkeypatch
) -> None:
    import met3r.adjacent16_stress as stress

    output = tmp_path / "deformation"
    staging = tmp_path / ".staging"
    output.mkdir()
    staging.mkdir()
    (output / "task.json").write_text(json.dumps({"status": "accepted", "old": True}))
    (staging / "task.json").write_text(json.dumps({"status": "accepted", "new": True}))
    real_exchange = stress._rename_exchange
    exchanged = False
    real_replace = stress.os.replace

    def mark_exchange(left, right):  # type: ignore[no-untyped-def]
        nonlocal exchanged
        real_exchange(left, right)
        exchanged = True

    def fail_after_exchange(left, right):  # type: ignore[no-untyped-def]
        if exchanged:
            raise AssertionError("post-exchange move must not occur")
        real_replace(left, right)

    monkeypatch.setattr(stress, "_rename_exchange", mark_exchange)
    monkeypatch.setattr(stress.os, "replace", fail_after_exchange)
    stress._promote(staging, output, _task(tmp_path), "fingerprint", "run")

    assert json.loads((output / "task.json").read_text())["new"] is True
    archived = output.parent / ".promotion_slots/fingerprint/slot_run/task.json"
    assert json.loads(archived.read_text())["old"] is True


def test_exchange_and_recovery_failure_marks_neutral_slot_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    import met3r.adjacent16_stress as stress

    task = _task(tmp_path)
    task.output_dir.mkdir(parents=True)
    (task.output_dir / "task.json").write_text(json.dumps({"status": "accepted", "old": True}))

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]))
        class Result: returncode = 0
        return Result()

    real_replace = stress.os.replace
    def fail_recovery(left, right):  # type: ignore[no-untyped-def]
        if Path(left).parents[1].name == ".promotion_slots" and Path(right).name.startswith(".staging-"):
            raise OSError("recovery move failed")
        real_replace(left, right)

    monkeypatch.setattr(stress, "_rename_exchange", lambda *args: (_ for _ in ()).throw(OSError("exchange failed")))
    monkeypatch.setattr(stress.os, "replace", fail_recovery)
    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    result = run_stress_tasks((task,), max_outer_attempts=1)

    assert result["accepted"] == 0
    assert json.loads((task.output_dir / "task.json").read_text())["old"] is True
    records = list((task.output_dir.parent / ".promotion_slots").rglob("task.json"))
    assert len(records) == 1
    candidate = json.loads(records[0].read_text())
    assert candidate["status"] == "rejected"
    assert candidate["acceptance"] == {"camera_preserved": False, "background_preserved": False, "crop_preservation_gate": False}


@pytest.mark.parametrize("field", ("camera", "background", "crop"))
def test_each_frozen_acceptance_gate_can_reject(tmp_path: Path, monkeypatch, field: str) -> None:
    task = _task(tmp_path)

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        values = {"camera": True, "background": True, "crop": True}
        values[field] = False
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]), **values)
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    assert run_stress_tasks((task,), max_outer_attempts=1)["accepted"] == 0


def test_unrelated_semantic_outsole_and_fit_fields_do_not_change_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path)

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        _accepted_artifacts(Path(args[args.index("--output-dir") + 1]), outsole=False, fit="worsened")
        class Result: returncode = 0
        return Result()

    monkeypatch.setattr("met3r.adjacent16_stress.subprocess.run", fake_run)
    assert run_stress_tasks((task,), max_outer_attempts=1)["accepted"] == 1


def _accepted_artifacts(output: Path, *, camera: bool = True, background: bool = True,
                        crop: bool = True, outsole: bool = True, fit: str = "worsened") -> None:
    _png(output / "deformed.png")
    _png(output / "deformed_mask.png", "L")
    selected = {
        "attempt_id": "attempt", "semantic": {
            "camera_preserved": camera, "background_preserved": background,
            "outsole_preserved": outsole, "passes_gate": False,
        }, "preservation": {"passes_gate": crop},
        "comparison": {"relation": fit}, "metrics": {"fit_score": 999.0},
    }
    (output / "deformation.json").write_text(json.dumps({"selected_attempt_id": "attempt", "attempts": [selected]}))
    (output / "metrics.json").write_text("{}")


def _hold_gpu_claim(lock_root: Path, queue: multiprocessing.Queue) -> None:
    from met3r.adjacent16_stress import _machine_gpu_claim

    with _machine_gpu_claim(0, lock_root=lock_root):
        started = time.monotonic()
        time.sleep(0.12)
        queue.put((started, time.monotonic()))
