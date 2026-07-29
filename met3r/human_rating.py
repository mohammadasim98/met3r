"""Blinded human-rating packets with non-overlapping 1--5 criteria."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image


@dataclass(frozen=True)
class RatingDimension:
    title: str
    include: str
    exclude: str
    anchors: Mapping[int, str]


RATING_DIMENSIONS: dict[str, RatingDimension] = {
    "geometric_alignment": RatingDimension(
        title="Geometric / silhouette alignment",
        include=(
            "Judge whether the same physical surface, silhouette, occlusion boundary, and "
            "part position agree after accounting for the camera change."
        ),
        exclude=(
            "Ignore color, material, logo identity, texture sharpness, and whether either "
            "single image looks photorealistic."
        ),
        anchors={
            1: "Large pose/silhouette contradiction; corresponding regions cannot be matched.",
            3: "Overall pose agrees, but several part boundaries or occlusions are displaced.",
            5: "Silhouette, visible surfaces, and part locations agree to near-pixel precision.",
        },
    ),
    "semantic_identity": RatingDimension(
        title="Semantic shoe identity",
        include=(
            "Judge whether model, part inventory, colorway, material class, logo, laces, "
            "sole pattern, and other identifiable attributes describe the same shoe."
        ),
        exclude=(
            "Ignore small spatial displacement, blur/compression, camera-pose accuracy, and "
            "generic image realism."
        ),
        anchors={
            1: "Different shoe identity or major parts/attributes are invented or missing.",
            3: "Same broad shoe, but one or more salient identity attributes disagree.",
            5: "All visible identity-bearing parts and attributes are consistently preserved.",
        },
    ),
    "local_detail_alignment": RatingDimension(
        title="Local detail / pixel-level alignment",
        include=(
            "Judge whether seams, lace holes, logo edges, texture boundaries, and other "
            "fine details occur at the geometrically corresponding locations."
        ),
        exclude=(
            "Do not reward merely recognizing the same shoe; ignore global silhouette and "
            "single-view photographic quality unless they prevent detail localization."
        ),
        anchors={
            1: "Fine details are hallucinated, missing, or located on non-corresponding surfaces.",
            3: "Major details match semantically but are blurred or locally shifted.",
            5: "Fine structures match sharply at their expected surface locations.",
        },
    ),
    "single_view_plausibility": RatingDimension(
        title="Single-view visual plausibility",
        include=(
            "Judge rendering artifacts, broken anatomy, duplicated structures, unnatural "
            "texture, and whether each image is individually plausible."
        ),
        exclude=(
            "Ignore cross-view agreement and camera correspondence; a plausible but different "
            "shoe can score high only on this dimension."
        ),
        anchors={
            1: "Severe artifacts or physically implausible shoe structure.",
            3: "Generally plausible with visible synthesis artifacts or malformed local regions.",
            5: "Both views are individually natural and artifact-free.",
        },
    ),
}


@dataclass(frozen=True)
class RatingPacket:
    tasks_path: Path
    key_path: Path
    rubric_path: Path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rubric_markdown() -> str:
    lines = [
        "# Blinded multi-view rating rubric",
        "",
        "Do not inspect metric outputs or the hidden condition key while rating. Score each "
        "dimension independently from 1 to 5; use 2 or 4 when quality lies between anchors.",
        "",
    ]
    for name, dimension in RATING_DIMENSIONS.items():
        lines.extend(
            [
                f"## {dimension.title} (`{name}`)",
                "",
                f"Evaluate: {dimension.include}",
                "",
                f"Do not evaluate here: {dimension.exclude}",
                "",
                *[
                    f"- {score}: {description}"
                    for score, description in dimension.anchors.items()
                ],
                "",
            ]
        )
    return "\n".join(lines)


def write_blinded_rating_packet(
    samples: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    *,
    seed: int = 42,
    repeat_fraction: float = 0.0,
) -> RatingPacket:
    """Write a randomized rater CSV and a separately stored unblinding key."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stimuli_dir = destination / "stimuli"
    stimuli_dir.mkdir(exist_ok=True)
    if not 0.0 <= repeat_fraction < 1.0:
        raise ValueError("repeat_fraction must be in [0, 1)")
    rng = random.Random(seed)
    order = [(index, False) for index in range(len(samples))]
    if repeat_fraction > 0 and samples:
        repeat_count = min(len(samples), math.ceil(len(samples) * repeat_fraction))
        order.extend((index, True) for index in rng.sample(range(len(samples)), repeat_count))
    rng.shuffle(order)

    tasks: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    for position, (sample_index, is_repeat) in enumerate(order, start=1):
        sample = dict(samples[sample_index])
        for required in ("sample_id", "first_view", "second_view"):
            if required not in sample:
                raise ValueError(f"rating sample is missing {required!r}")
        task_id = f"R{position:04d}"
        staged_paths = {}
        for field, side in (("first_view", "first"), ("second_view", "second")):
            source_path = Path(str(sample[field])).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"missing rating stimulus: {source_path}")
            suffix = source_path.suffix.lower() or ".png"
            staged_name = f"{task_id}_{side}{suffix}"
            with Image.open(source_path) as image:
                image.convert("RGB").save(stimuli_dir / staged_name)
            staged_paths[field] = f"stimuli/{staged_name}"
        tasks.append(
            {
                "task_id": task_id,
                "first_view": staged_paths["first_view"],
                "second_view": staged_paths["second_view"],
                **{name: "" for name in RATING_DIMENSIONS},
                "notes": "",
            }
        )
        keys.append(
            {
                "task_id": task_id,
                "is_hidden_repeat": is_repeat,
                **{
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in sample.items()
                },
            }
        )

    tasks_path = destination / "rating_tasks.csv"
    key_path = destination.parent / f"{destination.name}_unblinding_key.csv"
    rubric_path = destination / "RUBRIC.md"
    _write_csv(
        tasks_path,
        tasks,
        ["task_id", "first_view", "second_view", *RATING_DIMENSIONS, "notes"],
    )
    key_columns = ["task_id", *sorted({key for row in keys for key in row if key != "task_id"})]
    _write_csv(key_path, keys, key_columns)
    rubric_path.write_text(_rubric_markdown(), encoding="utf-8")
    return RatingPacket(tasks_path=tasks_path, key_path=key_path, rubric_path=rubric_path)


def validate_completed_ratings(path: str | Path) -> list[dict[str, object]]:
    """Parse a completed packet and reject missing/non-integer/out-of-range scores."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row_index, row in enumerate(rows, start=2):
        for name in RATING_DIMENSIONS:
            raw = row.get(name, "")
            try:
                value = int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {row_index} {name} must be an integer from 1 to 5") from error
            if not 1 <= value <= 5:
                raise ValueError(f"row {row_index} {name} must be an integer from 1 to 5")
            row[name] = value
    return rows
