from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from met3r.human_rating import (
    RATING_DIMENSIONS,
    validate_completed_ratings,
    write_blinded_rating_packet,
)


def test_rating_dimensions_have_distinct_targets_and_anchors() -> None:
    assert tuple(RATING_DIMENSIONS) == (
        "geometric_alignment",
        "semantic_identity",
        "local_detail_alignment",
        "single_view_plausibility",
    )
    for dimension in RATING_DIMENSIONS.values():
        assert set(dimension.anchors) == {1, 3, 5}
        assert dimension.include
        assert dimension.exclude


def test_blinded_packet_hides_metric_and_condition_and_is_seeded(tmp_path: Path) -> None:
    stimulus_paths = {}
    for condition in ("clean", "warp"):
        for side in ("left", "right"):
            path = tmp_path / condition / f"{side}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new(
                "RGB",
                (3, 3),
                color=(255, 0, 0) if condition == "clean" else (0, 0, 255),
            )
            image.save(path, pnginfo=None)
            stimulus_paths[(condition, side)] = path
    samples = [
        {
            "sample_id": "shoe-a-clean",
            "first_view": str(stimulus_paths[("clean", "left")]),
            "second_view": str(stimulus_paths[("clean", "right")]),
            "condition": "clean",
            "metric_score": 0.1,
        },
        {
            "sample_id": "shoe-a-warp",
            "first_view": str(stimulus_paths[("warp", "left")]),
            "second_view": str(stimulus_paths[("warp", "right")]),
            "condition": "warp",
            "metric_score": 0.8,
        },
    ]

    first = write_blinded_rating_packet(samples, tmp_path / "one", seed=42)
    second = write_blinded_rating_packet(samples, tmp_path / "two", seed=42)

    task_text = first.tasks_path.read_text()
    assert "metric_score" not in task_text
    assert "condition" not in task_text
    assert "clean" not in task_text
    assert task_text == second.tasks_path.read_text()
    assert "metric_score" in first.key_path.read_text()
    assert first.key_path.parent == first.tasks_path.parent.parent
    assert len(list((first.tasks_path.parent / "stimuli").iterdir())) == 4
    with Image.open(first.tasks_path.parent / "stimuli" / "R0001_first.png") as image:
        assert not image.getexif()

    repeated = write_blinded_rating_packet(
        samples,
        tmp_path / "repeated",
        seed=42,
        repeat_fraction=0.5,
    )
    with repeated.tasks_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert "is_hidden_repeat" in repeated.key_path.read_text()


def test_completed_rating_validation_requires_integer_one_to_five(tmp_path: Path) -> None:
    path = tmp_path / "ratings.csv"
    columns = ["task_id", *RATING_DIMENSIONS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "R0001",
                "geometric_alignment": "5",
                "semantic_identity": "4",
                "local_detail_alignment": "3",
                "single_view_plausibility": "6",
            }
        )

    with pytest.raises(ValueError, match="single_view_plausibility"):
        validate_completed_ratings(path)
