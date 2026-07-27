from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from met3r.adjacent16_contract import load_adjacent16_contract
from met3r.adjacent16_data import build_adjacent16_manifest
from met3r.adjacent16_design import build_adjacent16_design
from met3r.adjacent16_materialize import materialize_adjacent_state
from tests.helpers_adjacent16 import write_dense_source, write_valid_stress


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


@pytest.fixture()
def manifest_and_design(tmp_path: Path):
    contract = load_adjacent16_contract(CONFIG)
    source = tmp_path / "reconstruction"
    stress = tmp_path / "stress"
    write_dense_source(source, contract)
    for shoe_id in contract.raw["shoes"]:
        for quartet in contract.quartets:
            for view_id in quartet.view_ids:
                write_valid_stress(stress, shoe_id, quartet.quartet_id, view_id)
    contract = replace(contract, reconstruction_root=source, stress_root=stress)
    manifest = build_adjacent16_manifest(contract, require_stress=True)
    return manifest, build_adjacent16_design(manifest, contract)


def test_materializes_exact_quartet_with_white_background_and_only_changed_view(
    tmp_path: Path, manifest_and_design
) -> None:
    manifest, design = manifest_and_design
    state = next(
        state for state in design.controlled_states
        if state.changed_view is not None and state.corruption == "jpeg"
    )

    result = materialize_adjacent_state(state, manifest, tmp_path)

    assert tuple(path.stem for path in result.image_paths) == state.view_ids
    assert tuple(path.stem for path in result.mask_paths) == state.view_ids
    assert len(list((result.state_dir / "images").glob("*.png"))) == 4
    assert len(list((result.state_dir / "masks").glob("*.png"))) == 4
    for image_path, mask_path in zip(result.image_paths, result.mask_paths):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        assert image.shape == (504, 504, 3)
        assert np.all(image[~mask] == 255)
    records = {
        view.view_id: view
        for shoe in manifest.shoes if shoe.shoe_id == state.shoe_id
        for view in shoe.views
    }
    unchanged = next(view_id for view_id in state.view_ids if view_id != state.changed_view)
    with Image.open(records[unchanged].source_path) as source:
        expected = np.asarray(source.convert("RGB").resize((504, 504), Image.Resampling.LANCZOS))
    with Image.open(records[unchanged].source_mask_path) as source_mask:
        mask = np.asarray(source_mask.convert("L").resize((504, 504), Image.Resampling.NEAREST)) > 127
    np.testing.assert_array_equal(
        np.asarray(Image.open(result.state_dir / "images" / f"{unchanged}.png").convert("RGB")),
        np.where(mask[..., None], expected, 255).astype(np.uint8),
    )


def test_stress_materializes_all_four_audited_deformations_without_corruption(
    tmp_path: Path, manifest_and_design
) -> None:
    manifest, design = manifest_and_design
    state = design.stress_states[0]

    result = materialize_adjacent_state(state, manifest, tmp_path)

    records = {
        view.view_id: view
        for shoe in manifest.shoes if shoe.shoe_id == state.shoe_id
        for view in shoe.views
    }
    for view_id in state.view_ids:
        record = records[view_id]
        assert record.stress_path is not None
        with Image.open(record.stress_path) as image:
            expected = np.asarray(image.convert("RGB").resize((504, 504), Image.Resampling.LANCZOS))
        with Image.open(record.stress_mask_path) as mask:
            foreground = np.asarray(mask.convert("L").resize((504, 504), Image.Resampling.NEAREST)) > 127
        actual = np.asarray(Image.open(result.state_dir / "images" / f"{view_id}.png").convert("RGB"))
        np.testing.assert_array_equal(actual, np.where(foreground[..., None], expected, 255).astype(np.uint8))
