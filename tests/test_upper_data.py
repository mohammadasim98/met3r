import json
from pathlib import Path

from met3r.upper_data import EXPECTED_SHOES, UPPER_VIEWS, build_upper_manifest


ROOT = Path("/root/dev/ByteSize/Reconstruction")


def test_real_upper_manifest_has_exactly_sixteen_views() -> None:
    manifest = build_upper_manifest(ROOT)
    assert tuple(shoe.shoe_id for shoe in manifest.shoes) == EXPECTED_SHOES
    assert all(tuple(view.view_id for view in shoe.views) == UPPER_VIEWS for shoe in manifest.shoes)
    assert sum(len(shoe.views) for shoe in manifest.shoes) == 16
    assert all(view.source_image.size == (1080, 1080) for shoe in manifest.shoes for view in shoe.views)
    assert min(view.processed_image_correlation for shoe in manifest.shoes for view in shoe.views) >= 0.99


def test_real_deformation_quality_matches_selected_attempt() -> None:
    manifest = build_upper_manifest(ROOT)
    for shoe in manifest.shoes:
        for view in shoe.views:
            quality = view.quality
            assert quality.status == "low_quality"
            assert quality.selected_attempt_id
            assert quality.fit_relation in {"improved", "equivalent", "worsened"}
            assert isinstance(quality.semantic_gate, bool)
            assert isinstance(quality.preservation_gate, bool)
            assert isinstance(quality.camera_preserved, bool)
            assert isinstance(quality.outsole_preserved, bool)


def test_manifest_json_contains_only_absolute_paths_and_scalars(tmp_path: Path) -> None:
    manifest = build_upper_manifest(ROOT)

    destination = manifest.write_json(tmp_path / "manifest.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert Path(payload["reconstruction_root"]).is_absolute()
    assert Path(payload["shoes"][0]["views"][0]["source_path"]).is_absolute()
    assert isinstance(payload["shoes"][0]["views"][0]["processed_image_correlation"], float)
