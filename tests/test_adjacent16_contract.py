from __future__ import annotations

import json
from pathlib import Path

import pytest

from met3r.adjacent16_contract import load_adjacent16_contract


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "adjacent16_core.json"


def _write_config(tmp_path: Path, mutate) -> Path:
    raw = json.loads(CONFIG.read_text())
    mutate(raw)
    path = tmp_path / "adjacent16_core.json"
    path.write_text(json.dumps(raw))
    return path


def test_loads_the_exact_frozen_adjacent16_contract() -> None:
    """Catch accidental changes to the protocol's externally used shape."""

    contract = load_adjacent16_contract(CONFIG)

    assert contract.experiment_id == "20260727_adjacent16_core_v1"
    assert contract.physical_gpus == (0, 1, 2, 3)
    assert tuple(quartet.quartet_id for quartet in contract.quartets) == (
        "q0",
        "q1",
        "q2",
        "q3",
    )
    assert tuple(quartet.view_ids for quartet in contract.quartets) == (
        ("1", "2", "3", "4"),
        ("14", "15", "16", "17"),
        ("27", "28", "29", "30"),
        ("40", "41", "42", "0"),
    )
    assert len(contract.feature_variants) == 12
    assert contract.expected_rows == {
        "controlled": 31_104,
        "geometry_calibration": 6_528,
        "stress": 1_152,
        "total": 38_784,
    }


def test_rejects_numeric_camera_order(tmp_path: Path) -> None:
    """Catch replacing the frozen azimuth orbit with numeric sorting."""

    path = _write_config(tmp_path, lambda raw: raw.update(camera_order=[str(i) for i in range(48)]))

    with pytest.raises(ValueError, match="camera_order"):
        load_adjacent16_contract(path)


def test_rejects_an_extra_feature_variant(tmp_path: Path) -> None:
    """Catch accidental expansion beyond the 12-variant scientific matrix."""

    def mutate(raw: dict[str, object]) -> None:
        variants = raw["feature_variants"]
        assert isinstance(variants, list)
        variants.append(variants[0])

    with pytest.raises(ValueError, match="feature_variants"):
        load_adjacent16_contract(_write_config(tmp_path, mutate))


def test_rejects_naf_at_37(tmp_path: Path) -> None:
    """Catch the prohibited below-native NAF target grid."""

    def mutate(raw: dict[str, object]) -> None:
        variants = raw["feature_variants"]
        assert isinstance(variants, list)
        variant = variants[3]
        assert isinstance(variant, dict)
        variant["feature_mode"] = "naf_normal_guide"

    with pytest.raises(ValueError, match="feature_variants"):
        load_adjacent16_contract(_write_config(tmp_path, mutate))


@pytest.mark.parametrize("forbidden", ["exposure", "blur"])
def test_rejects_excluded_corruption(tmp_path: Path, forbidden: str) -> None:
    """Catch reintroduction of excluded appearance-only corruptions."""

    def mutate(raw: dict[str, object]) -> None:
        corruptions = raw["corruptions"]
        assert isinstance(corruptions, dict)
        corruptions[forbidden] = [0.5]

    with pytest.raises(ValueError, match="corruptions"):
        load_adjacent16_contract(_write_config(tmp_path, mutate))


@pytest.mark.parametrize(
    "gpus",
    [
        [0, 1, 2],
        [0, 1, 2, 2],
        [-1, 0, 1, 2],
        [0, 1, 2, 4],
    ],
)
def test_rejects_any_gpu_set_other_than_the_frozen_physical_set(
    tmp_path: Path, gpus: list[int]
) -> None:
    """Catch nonphysical, duplicate, incomplete, and alternate GPU assignments."""

    path = _write_config(tmp_path, lambda raw: raw.update(physical_gpus=gpus))

    with pytest.raises(ValueError, match="physical_gpus"):
        load_adjacent16_contract(path)


def test_rejects_nonfrozen_camera_wording(tmp_path: Path) -> None:
    """Catch claims that elevate DA3 estimates to calibrated ground truth."""

    path = _write_config(tmp_path, lambda raw: raw.update(camera_label="calibrated GT"))

    with pytest.raises(ValueError, match="camera_label"):
        load_adjacent16_contract(path)


@pytest.mark.parametrize("row_name", ["geometry_calibration", "stress"])
def test_rejects_row_counts_inconsistent_with_frozen_formulas(
    tmp_path: Path, row_name: str
) -> None:
    """Catch count changes that would silently alter calibration or stress scope."""

    def mutate(raw: dict[str, object]) -> None:
        expected_rows = raw["expected_rows"]
        assert isinstance(expected_rows, dict)
        expected_rows[row_name] = 1

    with pytest.raises(ValueError, match="expected_rows"):
        load_adjacent16_contract(_write_config(tmp_path, mutate))
