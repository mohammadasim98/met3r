"""Atomic staging for upper-view image states at the camera-artifact size."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .corruptions import CorruptionSpec, apply_corruption_with_mask
from .upper_data import UpperManifest, UpperViewRecord
from .upper_design import ImageState, MaterializedState


def _stage(rgb_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(rgb_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        )
    with Image.open(mask_path) as image:
        mask = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        ) > 127
    return np.where(mask[..., None], rgb, 255).astype(np.uint8), mask


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _view_index(manifest: UpperManifest) -> dict[tuple[str, str], UpperViewRecord]:
    return {
        (shoe.shoe_id, view.view_id): view
        for shoe in manifest.shoes
        for view in shoe.views
    }


def _state_result(state_dir: Path, state: ImageState, view_ids: tuple[str, ...]) -> MaterializedState:
    image_paths = tuple(state_dir / "images" / f"{view_id}.png" for view_id in view_ids)
    mask_paths = tuple(state_dir / "masks" / f"{view_id}.png" for view_id in view_ids)
    masks = np.stack(
        [np.asarray(Image.open(path).convert("L")) > 127 for path in mask_paths]
    )
    return MaterializedState(
        state_id=state.state_id,
        state_dir=state_dir,
        image_paths=image_paths,
        mask_paths=mask_paths,
        masks=masks,
        metadata_path=state_dir / "state.json",
    )


def _transform_metadata(state: ImageState, stage_size: int) -> dict[str, object]:
    transform: dict[str, object] = {
        "kind": state.corruption,
        "strength": state.severity,
        "direction": state.direction,
    }
    if state.corruption == "translation":
        transform["x_offset_px"] = state.direction * round(stage_size * state.severity)
    elif state.corruption == "exposure":
        transform["factor"] = 2.0 ** (state.direction * state.severity)
    return transform


def materialize_image_state(
    state: ImageState,
    manifest: UpperManifest,
    output_dir: str | Path,
    stage_size: int = 504,
) -> MaterializedState:
    """Stage one four-view controlled or real-deformation state atomically."""

    if stage_size <= 0:
        raise ValueError("stage_size must be positive")
    shoe = next((item for item in manifest.shoes if item.shoe_id == state.shoe_id), None)
    if shoe is None:
        raise ValueError(f"state references an unknown shoe: {state.shoe_id}")
    view_ids = tuple(view.view_id for view in shoe.views)
    if state.changed_view is not None and state.changed_view not in view_ids:
        raise ValueError(f"state references an unknown view: {state.changed_view}")
    states_root = Path(output_dir) / "states"
    destination = states_root / state.state_id
    if destination.is_dir():
        return _state_result(destination, state, view_ids)
    states_root.mkdir(parents=True, exist_ok=True)
    records = _view_index(manifest)
    temporary = Path(tempfile.mkdtemp(prefix=f".{state.state_id}.", dir=states_root))
    try:
        images_dir = temporary / "images"
        masks_dir = temporary / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        masks: list[np.ndarray] = []
        sources: dict[str, dict[str, str]] = {}
        for view_id in view_ids:
            record = records[(state.shoe_id, view_id)]
            rgb_path = record.deformed_path if state.track == "stress" else record.source_path
            mask_path = (
                record.deformed_mask_path if state.track == "stress" else record.source_mask_path
            )
            image, mask = _stage(rgb_path, mask_path, stage_size)
            sources[view_id] = {"rgb": _hash(rgb_path), "mask": _hash(mask_path)}
            if view_id == state.changed_view:
                spec = CorruptionSpec(state.corruption, state.severity, state.direction)
                donor = donor_mask = None
                if state.corruption == "identity_swap":
                    if state.donor_shoe_id is None or state.donor_view is None:
                        raise ValueError("identity_swap state requires a donor shoe and view")
                    donor_record = records.get((state.donor_shoe_id, state.donor_view))
                    if donor_record is None:
                        raise ValueError("identity_swap state references an unknown donor")
                    donor, donor_mask = _stage(
                        donor_record.source_path, donor_record.source_mask_path, stage_size
                    )
                    sources[view_id]["donor_rgb"] = _hash(donor_record.source_path)
                    sources[view_id]["donor_mask"] = _hash(donor_record.source_mask_path)
                image, mask = apply_corruption_with_mask(
                    image,
                    mask,
                    spec,
                    seed=state.seed,
                    donor=donor,
                    donor_foreground_mask=donor_mask,
                )
            Image.fromarray(image).save(images_dir / f"{view_id}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"{view_id}.png")
            masks.append(mask)
        (temporary / "images.json").write_text(
            json.dumps(
                {"views": [{"view_id": view_id, "image": f"images/{view_id}.png", "mask": f"masks/{view_id}.png"} for view_id in view_ids]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (temporary / "state.json").write_text(
            json.dumps(
                {
                    "state": asdict(state),
                    "stage_size": stage_size,
                    "source_hashes": sources,
                    "transform": _transform_metadata(state, stage_size),
                    "seed": state.seed,
                    "donor": {
                        "shoe_id": state.donor_shoe_id,
                        "view_id": state.donor_view,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            temporary.rename(destination)
        except FileExistsError:
            shutil.rmtree(temporary)
        return _state_result(destination, state, view_ids)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
