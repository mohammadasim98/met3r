"""Atomic four-view state staging for the frozen adjacent16 protocol."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .adjacent16_data import Adjacent16Manifest, AdjacentViewRecord
from .adjacent16_design import AdjacentImageState
from .corruptions import CorruptionSpec, apply_corruption_with_mask
from .upper_design import MaterializedState


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=128)
def _cached_stage(
    rgb_path: str,
    rgb_mtime_ns: int,
    mask_path: str,
    mask_mtime_ns: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    del rgb_mtime_ns, mask_mtime_ns
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127
    return np.where(mask[..., None], rgb, 255).astype(np.uint8), mask


def _stage(rgb_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _cached_stage(
        str(rgb_path.resolve()), rgb_path.stat().st_mtime_ns,
        str(mask_path.resolve()), mask_path.stat().st_mtime_ns, size,
    )
    return image.copy(), mask.copy()


def _records(state: AdjacentImageState, manifest: Adjacent16Manifest) -> dict[str, AdjacentViewRecord]:
    shoe = next((shoe for shoe in manifest.shoes if shoe.shoe_id == state.shoe_id), None)
    if shoe is None:
        raise ValueError(f"state references unknown shoe: {state.shoe_id}")
    records = {view.view_id: view for view in shoe.views if view.quartet_id == state.quartet_id}
    if tuple(records) != state.view_ids or len(records) != 4:
        raise ValueError("state view IDs do not exactly match its quartet")
    return records


def _result(destination: Path, state: AdjacentImageState) -> MaterializedState:
    image_paths = tuple(destination / "images" / f"{view_id}.png" for view_id in state.view_ids)
    mask_paths = tuple(destination / "masks" / f"{view_id}.png" for view_id in state.view_ids)
    masks = np.stack([np.asarray(Image.open(path).convert("L")) > 127 for path in mask_paths])
    return MaterializedState(state.state_id, destination, image_paths, mask_paths, masks, destination / "state.json")


def _expected_metadata(
    state: AdjacentImageState,
    records: dict[str, AdjacentViewRecord],
    stage_size: int,
) -> dict[str, object]:
    sources: dict[str, dict[str, str]] = {}
    for view_id in state.view_ids:
        record = records[view_id]
        rgb = record.stress_path if state.track == "stress" else record.source_path
        mask = record.stress_mask_path if state.track == "stress" else record.source_mask_path
        if rgb is None or mask is None:
            raise ValueError(f"stress state requires audited deformation for {view_id}")
        sources[view_id] = {"rgb": _hash(rgb), "mask": _hash(mask)}
    return {
        "state": asdict(state),
        "stage_size": stage_size,
        "source_hashes": sources,
        "view_ids": list(state.view_ids),
    }


def _images_manifest(state: AdjacentImageState) -> dict[str, object]:
    return {
        "views": [
            {
                "view_id": view_id,
                "image": f"images/{view_id}.png",
                "mask": f"masks/{view_id}.png",
            }
            for view_id in state.view_ids
        ]
    }


def _expected_view(
    state: AdjacentImageState,
    records: dict[str, AdjacentViewRecord],
    view_id: str,
    stage_size: int,
    *,
    manifest: Adjacent16Manifest,
) -> tuple[np.ndarray, np.ndarray]:
    record = records[view_id]
    rgb_path = record.stress_path if state.track == "stress" else record.source_path
    mask_path = record.stress_mask_path if state.track == "stress" else record.source_mask_path
    if rgb_path is None or mask_path is None:
        raise ValueError(f"missing audited deformation for {state.state_id}/{view_id}")
    image, mask = _stage(rgb_path, mask_path, stage_size)
    if state.track == "controlled" and view_id == state.changed_view and mask.any():
        donor = donor_mask = None
        if state.corruption == "identity_swap":
            if state.donor_shoe_id is None or state.donor_view is None:
                raise ValueError("identity_swap requires donor shoe and view")
            donor_state = AdjacentImageState(
                state_id="donor", track="controlled", shoe_id=state.donor_shoe_id,
                quartet_id=state.quartet_id, view_ids=state.view_ids, changed_view=None,
                corruption="clean", severity=0.0, direction=1, seed=0,
            )
            donor_record = _records(donor_state, manifest).get(state.donor_view)
            if donor_record is None:
                raise ValueError("identity_swap donor is not in the same quartet")
            donor, donor_mask = _stage(donor_record.source_path, donor_record.source_mask_path, stage_size)
        image, mask = apply_corruption_with_mask(
            image, mask, CorruptionSpec(state.corruption, state.severity, state.direction),
            seed=state.seed, donor=donor, donor_foreground_mask=donor_mask,
        )
    return np.where(mask[..., None], image, 255).astype(np.uint8), mask


def _matches(
    destination: Path,
    state: AdjacentImageState,
    metadata: dict[str, object],
    *,
    records: dict[str, AdjacentViewRecord] | None = None,
    manifest: Adjacent16Manifest | None = None,
) -> bool:
    try:
        canonical_metadata = json.loads(json.dumps(metadata, sort_keys=True))
        if json.loads((destination / "state.json").read_text(encoding="utf-8")) != canonical_metadata:
            return False
        if json.loads((destination / "images.json").read_text(encoding="utf-8")) != _images_manifest(state):
            return False
        expected_files = {"state.json", "images.json"}
        for view_id in state.view_ids:
            expected_files.update({f"images/{view_id}.png", f"masks/{view_id}.png"})
        actual_files = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
        if actual_files != expected_files:
            return False
        for view_id in state.view_ids:
            image_path = destination / "images" / f"{view_id}.png"
            mask_path = destination / "masks" / f"{view_id}.png"
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.convert("RGB").size != (504, 504) or mask.convert("L").size != (504, 504):
                    return False
                if records is not None and manifest is not None:
                    expected_image, expected_mask = _expected_view(state, records, view_id, 504, manifest=manifest)
                    if not np.array_equal(np.asarray(image.convert("RGB")), expected_image):
                        return False
                    if not np.array_equal(np.asarray(mask.convert("L")) > 127, expected_mask):
                        return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _replace(temporary: Path, destination: Path) -> None:
    backup: Path | None = None
    try:
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup.", dir=destination.parent
                )
            )
            backup.rmdir()
            destination.rename(backup)
        temporary.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_adjacent_state(
    state: AdjacentImageState,
    manifest: Adjacent16Manifest,
    run_root: Path,
    *,
    stage_size: int = 504,
) -> MaterializedState:
    """Atomically stage exactly one four-view quartet state."""

    if stage_size != 504:
        raise ValueError("stage_size must be exactly 504")
    if state.track not in {"controlled", "stress"}:
        raise ValueError("state track must be controlled or stress")
    if state.track == "stress" and (state.changed_view is not None or state.corruption != "deformed"):
        raise ValueError("stress states must contain four unmodified deformations")
    if state.track == "controlled" and state.changed_view not in {*state.view_ids, None}:
        raise ValueError("controlled state changed view is outside its quartet")
    records = _records(state, manifest)
    destination = Path(run_root) / "states" / state.state_id
    metadata = _expected_metadata(state, records, stage_size)
    if destination.is_dir() and _matches(destination, state, metadata, records=records, manifest=manifest):
        return _result(destination, state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{state.state_id}.staging.", dir=destination.parent))
    try:
        images = temporary / "images"
        masks = temporary / "masks"
        images.mkdir()
        masks.mkdir()
        for view_id in state.view_ids:
            image, mask = _expected_view(state, records, view_id, stage_size, manifest=manifest)
            Image.fromarray(image).save(images / f"{view_id}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks / f"{view_id}.png")
        (temporary / "images.json").write_text(
            json.dumps(_images_manifest(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "state.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not _matches(temporary, state, metadata):
            raise RuntimeError("staged state failed canonical validation")
        _replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _result(destination, state)
