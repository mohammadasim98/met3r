"""Qualitative diagnostics for known-camera correspondence failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .known_camera import PairMetricResult


def save_direction_diagnostics(
    target_image: np.ndarray,
    source_image: np.ndarray,
    result: PairMetricResult,
    output_path: str | Path,
    *,
    batch_index: int = 0,
) -> Path:
    """Save images, feature error, support, depth error, and cycle error together."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    score = result.score_map[batch_index].detach().float().cpu().numpy()
    valid = result.valid_mask[batch_index].detach().cpu().numpy()
    depth_error = result.relative_depth_error[batch_index].detach().float().cpu().numpy()
    cycle_error = result.cycle_error_px[batch_index].detach().float().cpu().numpy()
    masked_score = np.where(valid, score, np.nan)
    masked_depth = np.where(valid, depth_error, np.nan)
    masked_cycle = np.where(valid, cycle_error, np.nan)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = (
        (target_image, "Target view", None),
        (source_image, "Source view", None),
        (masked_score, "DINO dissimilarity (valid only)", "magma"),
        (valid, "Common valid support", "gray"),
        (masked_depth, "Relative depth disagreement", "viridis"),
        (masked_cycle, "Cycle reprojection error (px)", "plasma"),
    )
    for axis, (data, title, color_map) in zip(axes.flat, panels):
        rendered = axis.imshow(data, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
        if color_map not in (None, "gray"):
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"score={float(result.score[batch_index]):.4f}, "
        f"overlap={float(result.overlap_fraction[batch_index]):.3f}, "
        f"abstained={bool(result.abstained[batch_index])}"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
