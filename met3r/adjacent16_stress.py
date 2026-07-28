"""Independent, audited stress deformations for adjacent16 selected views.

The ByteSize renderer and deformation program remain authoritative.  This
module only supplies DA3 camera conversion, immutable inputs, and an atomic
orchestration boundary around those programs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adjacent16_contract import Adjacent16Contract
from .adjacent16_data import Adjacent16Manifest, AdjacentViewRecord


_BYTESIZE_RECONSTRUCTION = Path("/root/dev/ByteSize/Reconstruction")
_CAMERA_LABEL = "fixed/precomputed DA3 cameras (not calibrated GT)"
_PROMPT_VERSION = "shoe_deform_generation_prompt_v1"
_REQUIRED_FILES = (
    "source.png", "last_render.png", "source_mask.png", "deformed.png",
    "deformed_mask.png", "deformation.json", "metrics.json", "task.json",
)
_NON_TASK_FILES = tuple(name for name in _REQUIRED_FILES if name != "task.json")
_CLAIM_GUARD = threading.Lock()
_CLAIMS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class CameraPose:
    """The public pose shape consumed by ``last_view_match.render``."""

    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    horizontal_fov_deg: float


@dataclass(frozen=True)
class StressTask:
    shoe_id: str
    quartet_id: str
    view_id: str
    source_path: Path
    mask_path: Path
    last_render_path: Path
    output_dir: Path
    physical_gpu: int


def camera_pose_from_da3(
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    world_to_last_rotation: np.ndarray,
    *,
    stage_size: int = 504,
) -> CameraPose:
    """Convert one DA3 W2C camera into ByteSize's canonical-last pose.

    ``world_to_last_rotation`` maps a DA3-world vector into the saved shoe-last
    coordinate system.  The W2C camera's negative world Z axis points from the
    object to the camera, which is exactly ``CameraPose.direction``.
    """

    w2c = _w2c_rotation(extrinsic_w2c)
    last_rotation = _rotation(world_to_last_rotation, "world_to_last_rotation")
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0:
        raise ValueError("intrinsic must be a finite 3x3 matrix with positive fx")
    if not isinstance(stage_size, int) or stage_size <= 0:
        raise ValueError("stage_size must be a positive integer")

    direction = _unit(last_rotation @ (-w2c.T @ np.array([0.0, 0.0, 1.0])))
    right = _unit(last_rotation @ (w2c.T @ np.array([1.0, 0.0, 0.0])))
    elevation = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
    azimuth = math.degrees(math.atan2(-direction[0], direction[1])) % 360.0
    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)
    # This is the zero-roll basis in last_view_match.camera.camera_basis().
    # Reconstructing it here prevents a sign-flipped roll convention.
    base_up = _unit(np.array([
        math.sin(elevation_rad) * math.sin(azimuth_rad),
        -math.sin(elevation_rad) * math.cos(azimuth_rad),
        math.cos(elevation_rad),
    ]))
    base_right = _unit(np.cross(-direction, base_up))
    roll = math.degrees(math.atan2(float(np.dot(right, base_up)), float(np.dot(right, base_right))))
    fov = math.degrees(2.0 * math.atan(stage_size / (2.0 * float(matrix[0, 0]))))
    return CameraPose(azimuth, elevation, _normalise_roll(roll), fov)


def mask_bbox_xyxy(mask_path: str | Path) -> tuple[int, int, int, int]:
    """Return the exclusive xyxy bbox from the selected view's one-based mask."""

    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 127
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"source mask is empty: {mask_path}")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def render_stress_references(
    contract: Adjacent16Contract,
    manifest: Adjacent16Manifest,
) -> tuple[Path, ...]:
    """Render the saved last in every selected fixed/precomputed DA3 view."""

    if contract.raw.get("camera_label") != _CAMERA_LABEL:
        raise ValueError("adjacent16 stress rendering requires the frozen camera label")
    outputs: list[Path] = []
    for shoe in manifest.shoes:
        camera_root = (
            contract.reconstruction_root / "six_view_experiment_result_precomputed"
            / "multiview" / shoe.shoe_id / "precomputed_icp"
        )
        multi = _json_object(camera_root / "multiview_camera.json")
        last_rotation = np.asarray(multi.get("world_to_last_rotation"), dtype=np.float64)
        last_mesh = Path(_required_string(multi, "last_mesh")).resolve()
        if not last_mesh.is_file():
            raise FileNotFoundError(f"saved last mesh does not exist: {last_mesh}")
        _validate_cardinal_cameras(
            contract.reconstruction_root, camera_root, shoe.camera_dataset_id, last_rotation
        )
        intrinsics, extrinsics = _selected_camera_arrays(contract.reconstruction_root, shoe.camera_dataset_id)
        for view in shoe.views:
            index = int(view.view_id)
            pose = camera_pose_from_da3(extrinsics[index], intrinsics[index], last_rotation, stage_size=int(contract.raw["stage_size"]))
            output = _stress_output(contract, view) / "last_render.png"
            _render_reference_atomic(last_mesh, pose, view.source_mask_path, output)
            outputs.append(output)
    return tuple(outputs)


def build_stress_tasks(
    contract: Adjacent16Contract,
    manifest: Adjacent16Manifest,
    references: Sequence[Path],
) -> tuple[StressTask, ...]:
    """Bind selected views to already-rendered references and physical GPUs."""

    views = [view for shoe in manifest.shoes for view in shoe.views]
    if len(views) != len(references):
        raise ValueError("stress references must correspond one-to-one with selected views")
    return tuple(
        StressTask(
            shoe_id=view.shoe_id, quartet_id=view.quartet_id, view_id=view.view_id,
            source_path=view.source_path, mask_path=view.source_mask_path,
            last_render_path=Path(reference), output_dir=_stress_output(contract, view),
            physical_gpu=contract.physical_gpus[index % len(contract.physical_gpus)],
        )
        for index, (view, reference) in enumerate(zip(views, references))
    )


def run_stress_tasks(
    tasks: Sequence[StressTask],
    *,
    max_parallel: int = 4,
    max_outer_attempts: int = 1,
) -> Mapping[str, object]:
    """Run one independent ``shoe_deform`` subprocess per view atomically."""

    if not 1 <= max_parallel <= 4:
        raise ValueError("max_parallel must be between one and four")
    if max_outer_attempts < 1:
        raise ValueError("max_outer_attempts must be positive")
    _validate_tasks(tasks)
    results: dict[str, Mapping[str, object]] = {}
    gpu_claims = {gpu: threading.BoundedSemaphore(1) for gpu in range(4)}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        pending = {
            executor.submit(_run_one_task, task, max_outer_attempts, gpu_claims[task.physical_gpu]): _task_key(task)
            for task in tasks
        }
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    accepted = sum(bool(result["accepted"]) for result in results.values())
    return {"accepted": accepted, "rejected": len(results) - accepted, "tasks": results}


def _run_one_task(
    task: StressTask, max_outer_attempts: int, gpu_claim: threading.BoundedSemaphore
) -> Mapping[str, object]:
    with _output_claim(task.output_dir):
        fingerprint = _task_fingerprint(task)
        if _is_reusable(task, fingerprint):
            return {"accepted": True, "reused": True, "fingerprint": fingerprint}
        run_id = uuid.uuid4().hex
        for outer_attempt in range(1, max_outer_attempts + 1):
            task.output_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=task.output_dir.parent))
            record: dict[str, object] | None = None
            try:
                _copy_inputs(task, staging)
                input_staging = Path(
                    tempfile.mkdtemp(prefix=".inputs-", dir=task.output_dir.parent)
                )
                try:
                    _copy_inputs(task, input_staging)
                    command = _deformation_command(input_staging, staging)
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(task.physical_gpu)
                    environment["SHOE_DEFORM_CODEX_SANDBOX"] = (
                        "danger-full-access"
                    )
                    existing_pythonpath = environment.get("PYTHONPATH")
                    environment["PYTHONPATH"] = os.pathsep.join(
                        part for part in (
                            str(_BYTESIZE_RECONSTRUCTION),
                            str(_BYTESIZE_RECONSTRUCTION / "third_party" / "sam3"),
                            existing_pythonpath,
                        )
                        if part
                    )
                    with gpu_claim:
                        with _machine_gpu_claim(task.physical_gpu):
                            completed = subprocess.run(
                                command, env=environment, close_fds=False,
                                text=True, capture_output=True, check=False,
                            )
                finally:
                    shutil.rmtree(input_staging)
                shutil.copy2(task.mask_path, staging / "source_mask.png")
                generation_valid, acceptance, reliability = _generation_assessment(
                    staging, completed.returncode
                )
                record = _task_record(
                    task, fingerprint, command, outer_attempt, acceptance,
                    completed.returncode, run_id, generation_valid=generation_valid,
                    reliability=reliability,
                )
                if not generation_valid:
                    _write_record(
                        staging,
                        {
                            **record,
                            "status": "rejected",
                            "reason": "generation artifacts are incomplete or malformed",
                        },
                    )
                    _archive_rejected(staging, task, outer_attempt, fingerprint, run_id)
                    continue
                output_hashes = _verified_output_hashes(staging)
                _write_record(staging, {**record, "status": "accepted", "output_hashes": output_hashes})
                _promote(staging, task.output_dir, task, fingerprint, run_id)
                return {
                    "accepted": True,
                    "reused": False,
                    "fingerprint": fingerprint,
                    "strict_compliant": reliability["strict_compliant"],
                    "alignment_class": reliability["alignment_class"],
                }
            except Exception as error:
                if staging.exists():
                    rejected = {**(record or _task_record(task, fingerprint, [], outer_attempt, _false_acceptance(), None, run_id)),
                                "status": "rejected", "reason": str(error), "acceptance": _false_acceptance()}
                    _write_record(staging, rejected)
                    _archive_rejected(staging, task, outer_attempt, fingerprint, run_id)
                if outer_attempt == max_outer_attempts:
                    return {"accepted": False, "error": str(error), "fingerprint": fingerprint}
        return {"accepted": False, "fingerprint": fingerprint}


def _generation_assessment(
    staging: Path, returncode: int
) -> tuple[bool, dict[str, bool], dict[str, object]]:
    acceptance = {"camera_preserved": False, "background_preserved": False, "crop_preservation_gate": False}
    reliability = _false_reliability()
    if (
        returncode != 0
        or not (staging / "deformation.json").is_file()
        or any(not (staging / name).is_file() for name in ("deformed.png", "deformed_mask.png"))
    ):
        return False, acceptance, reliability
    try:
        for name in ("deformed.png", "deformed_mask.png"):
            with Image.open(staging / name) as image:
                image.verify()
    except (OSError, ValueError):
        return False, acceptance, reliability
    deformation = _json_object(staging / "deformation.json")
    selected_id = _required_string(deformation, "selected_attempt_id")
    attempts = deformation.get("attempts")
    if not isinstance(attempts, list):
        return False, acceptance, reliability
    selected = [item for item in attempts if isinstance(item, Mapping) and item.get("attempt_id") == selected_id]
    if len(selected) != 1:
        return False, acceptance, reliability
    semantic = selected[0].get("semantic")
    preservation = selected[0].get("preservation")
    if not isinstance(semantic, Mapping) or not isinstance(preservation, Mapping):
        return False, acceptance, reliability
    acceptance = {
        "camera_preserved": semantic.get("camera_preserved") is True,
        "background_preserved": semantic.get("background_preserved") is True,
        "crop_preservation_gate": preservation.get("passes_gate") is True,
    }
    try:
        center_shift = _finite_float(
            preservation, "center_shift_normalized"
        )
        bottom_shift = _finite_float(
            preservation, "bottom_shift_normalized"
        )
        width_ratio = _finite_float(preservation, "width_ratio")
    except ValueError:
        return False, acceptance, reliability
    moderate = (
        center_shift <= 0.03
        and bottom_shift <= 0.03
        and 0.90 <= width_ratio <= 1.10
    )
    strict_compliant = all(acceptance.values())
    reliability = {
        **acceptance,
        "center_shift_normalized": center_shift,
        "bottom_shift_normalized": bottom_shift,
        "width_ratio": width_ratio,
        "strict_compliant": strict_compliant,
        "moderate_alignment": moderate,
        "alignment_class": (
            "strict"
            if strict_compliant
            else "moderate"
            if moderate
            else "high_misalignment"
        ),
    }
    return True, acceptance, reliability


def _finite_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _copy_inputs(task: StressTask, staging: Path) -> None:
    for source, name in ((task.source_path, "source.png"), (task.mask_path, "source_mask.png"), (task.last_render_path, "last_render.png")):
        if not source.is_file():
            raise FileNotFoundError(f"stress input does not exist: {source}")
        shutil.copy2(source, staging / name)


def _deformation_command(inputs: Path, output: Path) -> list[str]:
    return [
        sys.executable, "-m", "shoe_deform", "--image", str(inputs / "source.png"),
        "--last-render", str(inputs / "last_render.png"), "--output-dir", str(output),
        "--max-attempts", "2",
    ]


def _promote(staging: Path, output_dir: Path, task: StressTask, fingerprint: str, run_id: str) -> None:
    """Atomically install staging; never delete an accepted predecessor."""

    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    slot = output_dir.parent / ".promotion_slots" / fingerprint / f"slot_{run_id}"
    slot.parent.mkdir(parents=True, exist_ok=True)
    # Move the candidate into its final immutable archive slot *before* the
    # one commit operation.  After exchange, that same slot already contains
    # the old accepted directory; there is no fallible post-commit move.
    os.replace(staging, slot)
    try:
        _rename_exchange(slot, output_dir)
    except Exception as exchange_error:
        # Exchange has not committed: old canonical is still canonical.  Put
        # the candidate back where the caller's rejected archive path expects.
        try:
            os.replace(slot, staging)
        except Exception as recovery_error:
            _mark_neutral_recovery_rejected(slot, exchange_error, recovery_error, run_id)
            raise RuntimeError(
                "promotion exchange and candidate recovery both failed; "
                f"neutral recovery slot retained at {slot}"
            ) from recovery_error
        raise


def _mark_neutral_recovery_rejected(
    slot: Path, exchange_error: Exception, recovery_error: Exception, run_id: str
) -> None:
    """Fail closed in a neutral slot when neither promotion nor recovery moved."""

    reason = {
        "promotion_exchange_error": str(exchange_error),
        "recovery_move_error": str(recovery_error),
    }
    try:
        record = dict(_json_object(slot / "task.json"))
        record.update({"status": "rejected", "acceptance": _false_acceptance(), "reason": reason})
        _write_record(slot, record)
    except Exception as audit_error:
        marker = slot.parent / f"recovery_journal_{run_id}.json"
        marker.write_text(
            json.dumps({"slot": str(slot), "reason": reason, "audit_rewrite_error": str(audit_error)}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            "promotion recovery failed and neutral audit rewrite failed; "
            f"durable recovery journal written to {marker}"
        ) from audit_error


def _archive_rejected(
    staging: Path, task: StressTask, outer_attempt: int, fingerprint: str, run_id: str
) -> None:
    archive = task.output_dir.parent / "rejected_attempts" / fingerprint / f"run_{run_id}" / f"outer_{outer_attempt:02d}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, archive)


def _is_reusable(task: StressTask, fingerprint: str) -> bool:
    output_dir = task.output_dir
    task_path = output_dir / "task.json"
    if not task_path.is_file() or any(not (output_dir / name).is_file() for name in _REQUIRED_FILES):
        return False
    try:
        record = _json_object(task_path)
    except (OSError, ValueError):
        return False
    try:
        expected_inputs = _input_hashes(task)
        expected_outputs = _verified_output_hashes(output_dir)
    except (OSError, ValueError):
        return False
    return (
        record.get("status") == "accepted"
        and record.get("generation_valid") is True
        and record.get("fingerprint") == fingerprint
        and record.get("input_hashes") == expected_inputs
        and record.get("output_hashes") == expected_outputs
    )


def _task_fingerprint(task: StressTask) -> str:
    payload = {
        "source_sha256": _sha256(task.source_path), "mask_sha256": _sha256(task.mask_path),
        "last_render_sha256": _sha256(task.last_render_path), "prompt_version": _PROMPT_VERSION,
        "implementation_hash": _implementation_hash(), "bytesize_repository_state": _bytesize_fingerprint(),
        "camera_label": _CAMERA_LABEL,
    }
    return _hash_json(payload)


def _input_hashes(task: StressTask) -> Mapping[str, str]:
    return {
        "source.png": _sha256(task.source_path), "source_mask.png": _sha256(task.mask_path),
        "last_render.png": _sha256(task.last_render_path),
    }


def _false_acceptance() -> dict[str, bool]:
    return {"camera_preserved": False, "background_preserved": False, "crop_preservation_gate": False}


def _false_reliability() -> dict[str, object]:
    return {
        **_false_acceptance(),
        "center_shift_normalized": None,
        "bottom_shift_normalized": None,
        "width_ratio": None,
        "strict_compliant": False,
        "moderate_alignment": False,
        "alignment_class": "invalid",
    }


def _task_record(
    task: StressTask, fingerprint: str, command: Sequence[str], outer_attempt: int,
    acceptance: Mapping[str, bool], returncode: int | None, run_id: str,
    *,
    generation_valid: bool = False,
    reliability: Mapping[str, object] | None = None,
) -> dict[str, object]:
    reliability_record = dict(reliability or _false_reliability())
    return {
        "task_key": _task_key(task), "run_id": run_id, "input_hashes": _input_hashes(task),
        "command": list(command), "prompt_version": _PROMPT_VERSION,
        "outer_attempt": outer_attempt, "fingerprint": fingerprint,
        "implementation_hash": _implementation_hash(),
        "bytesize_repository_state": _bytesize_fingerprint(),
        "camera_label": _CAMERA_LABEL,
        "acceptance": dict(acceptance),
        "generation_valid": generation_valid,
        "strict_compliant": reliability_record["strict_compliant"],
        "reliability": reliability_record,
        "returncode": returncode,
    }


def _write_record(directory: Path, record: Mapping[str, object]) -> None:
    (directory / "task.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _implementation_hash() -> str:
    return _sha256(Path(__file__))


def _bytesize_fingerprint() -> Mapping[str, object]:
    return _bytesize_fingerprint_impl()


def _bytesize_fingerprint_impl() -> Mapping[str, object]:
    def git(*args: str) -> str:
        process = subprocess.Popen(
            ["git", "-C", str(_BYTESIZE_RECONSTRUCTION), *args],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"cannot fingerprint ByteSize: {stderr.strip()}")
        return stdout
    top_level = Path(git("rev-parse", "--show-toplevel").strip()).resolve()

    def top_git(*args: str) -> str:
        process = subprocess.Popen(
            ["git", "-C", str(top_level), *args],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"cannot fingerprint ByteSize: {stderr.strip()}")
        return stdout

    untracked = top_git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    untracked_hashes = {
        name: _sha256(top_level / name)
        for name in untracked if name and (top_level / name).is_file()
    }
    status = top_git("status", "--porcelain=v1")
    return {
        "repository": str(top_level), "reconstruction": str(_BYTESIZE_RECONSTRUCTION.resolve()),
        "head": top_git("rev-parse", "HEAD").strip(),
        "dirty": bool(status), "status_porcelain_v1": status.splitlines(),
        "unstaged_binary_diff_sha256": hashlib.sha256(top_git("diff", "--no-ext-diff", "--binary").encode()).hexdigest(),
        "staged_binary_diff_sha256": hashlib.sha256(top_git("diff", "--cached", "--no-ext-diff", "--binary").encode()).hexdigest(),
        "untracked_file_sha256": untracked_hashes,
        "last_view_match_tree_sha256": _tree_hash(_BYTESIZE_RECONSTRUCTION / "last_view_match"),
        "shoe_deform_tree_sha256": _tree_hash(_BYTESIZE_RECONSTRUCTION / "shoe_deform"),
    }


def _render_reference_atomic(mesh: Path, pose: CameraPose, mask: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as directory:
        candidate = Path(directory) / output.name
        _render_last(mesh, pose, candidate, canvas_size=_image_size(mask), target_bbox=mask_bbox_xyxy(mask))
        os.replace(candidate, output)


def _render_last(mesh: Path, pose: CameraPose, output: Path, **kwargs: Any) -> Any:
    source_root = str(_BYTESIZE_RECONSTRUCTION)
    inserted = source_root not in sys.path
    if inserted:
        sys.path.insert(0, source_root)
    try:
        return importlib.import_module("last_view_match.render").render_last(mesh, pose, output, **kwargs)
    finally:
        if inserted:
            sys.path.remove(source_root)


def _validate_cardinal_cameras(
    reconstruction_root: Path,
    camera_root: Path,
    camera_dataset_id: str,
    last_rotation: np.ndarray,
) -> None:
    intrinsics, extrinsics = _selected_camera_arrays(reconstruction_root, camera_dataset_id)
    for side in ("left", "back", "right", "front"):
        record = _json_object(camera_root / side / "camera.json")
        reference = record.get("camera_reference")
        camera = record.get("camera")
        if not isinstance(reference, Mapping) or reference.get("group") != "upper" or not isinstance(camera, Mapping):
            raise ValueError(f"invalid cardinal camera record: {camera_root / side / 'camera.json'}")
        index = reference.get("index")
        if not isinstance(index, int) or not 0 <= index < len(extrinsics):
            raise ValueError(f"invalid cardinal camera index for {side}")
        derived = camera_pose_from_da3(extrinsics[index], intrinsics[index], last_rotation)
        for field in ("azimuth_deg", "elevation_deg", "roll_deg", "horizontal_fov_deg"):
            expected = camera.get(field)
            if not isinstance(expected, (int, float)) or not math.isclose(getattr(derived, field), float(expected), abs_tol=1e-4):
                raise ValueError(f"DA3-to-last pose does not agree with saved cardinal {side} camera")


def _selected_camera_arrays(root: Path, camera_dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    outputs = root / "sample_result" / camera_dataset_id / "upper" / "glb_masked_white_bg" / "outputs"
    with np.load(outputs / "intrinsics.npz") as artifact:
        intrinsics = np.asarray(artifact["intrinsics"], dtype=np.float64)
    with np.load(outputs / "extrinsics.npz") as artifact:
        extrinsics = np.asarray(artifact["extrinsics"], dtype=np.float64)
    if intrinsics.shape != (48, 3, 3) or extrinsics.shape not in {(48, 3, 4), (48, 4, 4)}:
        raise ValueError("selected DA3 camera arrays do not have the required 48-view shapes")
    return intrinsics, extrinsics


def _stress_output(contract: Adjacent16Contract, view: AdjacentViewRecord) -> Path:
    return contract.stress_root / view.shoe_id / view.quartet_id / f"view_{view.view_id}" / "deformation"


def _validate_tasks(tasks: Sequence[StressTask]) -> None:
    if len({_task_key(task) for task in tasks}) != len(tasks):
        raise ValueError("stress task identity must be unique")
    if any(task.physical_gpu not in {0, 1, 2, 3} for task in tasks):
        raise ValueError("stress tasks must use physical GPUs 0, 1, 2, and 3")
    expected = tuple(index % 4 for index in range(len(tasks)))
    actual = tuple(task.physical_gpu for task in tasks)
    if actual != expected:
        raise ValueError("stress task GPU assignment must round-robin 0, 1, 2, 3")


def _task_key(task: StressTask) -> str:
    return f"{task.shoe_id}/{task.quartet_id}/view_{task.view_id}"


def _w2c_rotation(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape not in {(3, 4), (4, 4)} or not np.isfinite(matrix).all():
        raise ValueError("extrinsic must be a finite 3x4 or 4x4 W2C matrix")
    return _rotation(matrix[:3, :3], "extrinsic rotation")


def _rotation(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{label} must be a proper finite 3x3 rotation")
    return matrix


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 0 or not math.isfinite(length):
        raise ValueError("camera vector is invalid")
    return vector / length


def _normalise_roll(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON record must be an object: {path}")
    return payload


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_output_hashes(directory: Path) -> Mapping[str, str]:
    """Verify and hash exactly the non-task accepted artifacts."""

    missing = [name for name in _NON_TASK_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"accepted deformation is missing required artifacts: {missing}")
    return {name: _sha256(directory / name) for name in _NON_TASK_FILES}


@contextmanager
def _output_claim(output_dir: Path):
    """Cross-thread and cross-process lock for one canonical deformation path."""

    import fcntl  # Linux is the supported production platform for this run.

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    key = str(output_dir.resolve())
    with _CLAIM_GUARD:
        thread_claim = _CLAIMS.setdefault(key, threading.Lock())
    with thread_claim:
        claim_path = output_dir.parent / f".{output_dir.name}.claim.lock"
        with claim_path.open("a+") as claim:
            fcntl.flock(claim.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(claim.fileno(), fcntl.LOCK_UN)


@contextmanager
def _machine_gpu_claim(physical_gpu: int, *, lock_root: Path | None = None):
    """Machine-global physical-GPU exclusion held only for shoe_deform."""

    import fcntl

    root = Path("/tmp") if lock_root is None else Path(lock_root)
    root.mkdir(parents=True, exist_ok=True)
    claim_path = root / f"met3r-adjacent16-physical-gpu-{physical_gpu}.lock"
    with claim_path.open("a+") as claim:
        fcntl.flock(claim.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(claim.fileno(), fcntl.LOCK_UN)


def _rename_exchange(first: Path, second: Path) -> None:
    """Use Linux renameat2(RENAME_EXCHANGE), never a delete-then-replace swap."""

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_EXCHANGE) is required to replace an accepted deformation")
    result = renameat2(-100, os.fsencode(first), -100, os.fsencode(second), 0x2)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"renameat2 exchange failed: {first} <-> {second}")


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
