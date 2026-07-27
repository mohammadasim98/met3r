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
    max_outer_attempts: int = 3,
) -> Mapping[str, object]:
    """Run one independent ``shoe_deform`` subprocess per view atomically."""

    if not 1 <= max_parallel <= 4:
        raise ValueError("max_parallel must be between one and four")
    if max_outer_attempts < 1:
        raise ValueError("max_outer_attempts must be positive")
    _validate_tasks(tasks)
    results: dict[str, Mapping[str, object]] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        pending = {executor.submit(_run_one_task, task, max_outer_attempts): _task_key(task) for task in tasks}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    accepted = sum(bool(result["accepted"]) for result in results.values())
    return {"accepted": accepted, "rejected": len(results) - accepted, "tasks": results}


def _run_one_task(task: StressTask, max_outer_attempts: int) -> Mapping[str, object]:
    fingerprint = _task_fingerprint(task)
    if _is_reusable(task.output_dir, fingerprint):
        return {"accepted": True, "reused": True, "fingerprint": fingerprint}
    for outer_attempt in range(1, max_outer_attempts + 1):
        task.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=task.output_dir.parent))
        try:
            _copy_inputs(task, staging)
            command = _deformation_command(task, staging)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(task.physical_gpu)
            completed = subprocess.run(
                command, cwd=_BYTESIZE_RECONSTRUCTION, env=environment,
                text=True, capture_output=True, check=False,
            )
            # The frozen source mask is the one used to recover the target
            # bbox.  Do not replace it with shoe_deform's independently
            # segmented convenience mask.
            shutil.copy2(task.mask_path, staging / "source_mask.png")
            accepted, acceptance = _accepted(staging, completed.returncode)
            task_record = {
                "status": "accepted" if accepted else "rejected",
                "task_key": _task_key(task), "source_sha256": _sha256(task.source_path),
                "source_mask_sha256": _sha256(task.mask_path), "last_render_sha256": _sha256(task.last_render_path),
                "output_hashes": _output_hashes(staging), "command": command,
                "prompt_version": _PROMPT_VERSION, "outer_attempt": outer_attempt,
                "fingerprint": fingerprint, "implementation_hash": _implementation_hash(),
                "bytesize_repository_state": _bytesize_fingerprint(),
                "camera_label": _CAMERA_LABEL, "acceptance": acceptance,
                "returncode": completed.returncode,
            }
            (staging / "task.json").write_text(json.dumps(task_record, indent=2, sort_keys=True), encoding="utf-8")
            if accepted:
                _finalise_accepted(staging, task.output_dir)
                return {"accepted": True, "reused": False, "fingerprint": fingerprint}
            _archive_rejected(staging, task, outer_attempt, fingerprint)
        except Exception as error:
            if staging.exists():
                _archive_rejected(staging, task, outer_attempt, fingerprint)
            if outer_attempt == max_outer_attempts:
                return {"accepted": False, "error": str(error), "fingerprint": fingerprint}
        # staging was renamed to its archive on rejection.
    return {"accepted": False, "fingerprint": fingerprint}


def _accepted(staging: Path, returncode: int) -> tuple[bool, dict[str, bool]]:
    acceptance = {"camera_preserved": False, "background_preserved": False, "crop_preservation_gate": False}
    if returncode != 0 or not (staging / "deformation.json").is_file():
        return False, acceptance
    deformation = _json_object(staging / "deformation.json")
    selected_id = _required_string(deformation, "selected_attempt_id")
    attempts = deformation.get("attempts")
    if not isinstance(attempts, list):
        return False, acceptance
    selected = [item for item in attempts if isinstance(item, Mapping) and item.get("attempt_id") == selected_id]
    if len(selected) != 1:
        return False, acceptance
    semantic = selected[0].get("semantic")
    preservation = selected[0].get("preservation")
    if not isinstance(semantic, Mapping) or not isinstance(preservation, Mapping):
        return False, acceptance
    acceptance = {
        "camera_preserved": semantic.get("camera_preserved") is True,
        "background_preserved": semantic.get("background_preserved") is True,
        "crop_preservation_gate": preservation.get("passes_gate") is True,
    }
    return all(acceptance.values()), acceptance


def _copy_inputs(task: StressTask, staging: Path) -> None:
    for source, name in ((task.source_path, "source.png"), (task.mask_path, "source_mask.png"), (task.last_render_path, "last_render.png")):
        if not source.is_file():
            raise FileNotFoundError(f"stress input does not exist: {source}")
        shutil.copy2(source, staging / name)


def _deformation_command(task: StressTask, staging: Path) -> list[str]:
    return [sys.executable, "-m", "shoe_deform", "--image", str(task.source_path), "--last-render", str(task.last_render_path), "--output-dir", str(staging), "--max-attempts", "2"]


def _finalise_accepted(staging: Path, output_dir: Path) -> None:
    # Preserve the source-mask provenance instead of the segmenter-derived
    # mask that shoe_deform writes internally.
    required = set(_REQUIRED_FILES) - {"task.json"}
    missing = [name for name in required if not (staging / name).is_file()]
    if missing:
        raise ValueError(f"accepted deformation is missing required artifacts: {missing}")
    parent = output_dir.parent
    backup = parent / f".superseded-{output_dir.name}-{os.urandom(6).hex()}"
    if output_dir.exists():
        os.replace(output_dir, backup)
    os.replace(staging, output_dir)
    if backup.exists():
        shutil.rmtree(backup)


def _archive_rejected(staging: Path, task: StressTask, outer_attempt: int, fingerprint: str) -> None:
    archive = task.output_dir.parent / "rejected_attempts" / fingerprint / f"outer_{outer_attempt:02d}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        shutil.rmtree(archive)
    os.replace(staging, archive)


def _is_reusable(output_dir: Path, fingerprint: str) -> bool:
    task_path = output_dir / "task.json"
    if not task_path.is_file() or any(not (output_dir / name).is_file() for name in _REQUIRED_FILES):
        return False
    try:
        record = _json_object(task_path)
    except (OSError, ValueError):
        return False
    return record.get("status") == "accepted" and record.get("fingerprint") == fingerprint


def _task_fingerprint(task: StressTask) -> str:
    payload = {
        "source_sha256": _sha256(task.source_path), "mask_sha256": _sha256(task.mask_path),
        "last_render_sha256": _sha256(task.last_render_path), "prompt_version": _PROMPT_VERSION,
        "implementation_hash": _implementation_hash(), "bytesize_repository_state": _bytesize_fingerprint(),
        "camera_label": _CAMERA_LABEL,
    }
    return _hash_json(payload)


def _implementation_hash() -> str:
    return _sha256(Path(__file__))


def _bytesize_fingerprint() -> Mapping[str, object]:
    def git(*args: str) -> str:
        process = subprocess.Popen(
            ["git", "-C", str(_BYTESIZE_RECONSTRUCTION), *args],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"cannot fingerprint ByteSize: {stderr.strip()}")
        return stdout
    status = git("status", "--porcelain=v1")
    return {
        "repository": str(_BYTESIZE_RECONSTRUCTION), "head": git("rev-parse", "HEAD").strip(),
        "dirty": bool(status), "status_porcelain_v1": status.splitlines(),
        "diff_sha256": hashlib.sha256(git("diff", "--no-ext-diff", "--binary").encode()).hexdigest(),
        "last_view_match_render_sha256": _sha256(_BYTESIZE_RECONSTRUCTION / "last_view_match" / "render.py"),
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


def _output_hashes(directory: Path) -> Mapping[str, str]:
    """Hash materialized artifacts that exist before the task record itself."""

    return {
        path.name: _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "task.json"
    }


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
