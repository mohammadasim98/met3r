"""Camera/depth preprocessing shared by all feature resolutions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ResizedCameraGeometry:
    depth: Tensor
    mask: Tensor
    confidence: Tensor | None
    intrinsics: Tensor


def confidence_percentile_mask(
    foreground_mask: Tensor,
    confidence: Tensor,
    *,
    percentile: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Remove the lowest-confidence foreground percentile independently per image."""

    if not 0 <= percentile < 100:
        raise ValueError("percentile must be in [0, 100)")
    if confidence.shape != foreground_mask.shape:
        raise ValueError("confidence and foreground_mask must have matching shapes")
    filtered = torch.zeros_like(foreground_mask, dtype=torch.bool)
    thresholds = []
    quantile = percentile / 100
    for batch_index in range(foreground_mask.shape[0]):
        values = confidence[batch_index][foreground_mask[batch_index]]
        if values.numel() == 0:
            thresholds.append(torch.full((), torch.nan, device=confidence.device))
            continue
        threshold = torch.quantile(values.float(), quantile).to(confidence.dtype)
        thresholds.append(threshold)
        filtered[batch_index] = foreground_mask[batch_index] & (
            confidence[batch_index] >= threshold
        )
    return filtered, torch.stack(thresholds)


def resize_camera_geometry(
    depth: Tensor,
    mask: Tensor,
    confidence: Tensor | None,
    intrinsics: Tensor,
    *,
    output_size: tuple[int, int],
) -> ResizedCameraGeometry:
    """Resize raster geometry and scale pixel-coordinate intrinsics."""

    input_h, input_w = depth.shape[-2:]
    output_h, output_w = output_size
    if output_h <= 0 or output_w <= 0:
        raise ValueError("output_size must be positive")
    resized_depth = F.interpolate(
        depth[:, None],
        size=output_size,
        mode="nearest-exact",
    )[:, 0]
    resized_mask = (
        F.interpolate(
            mask[:, None].to(dtype=torch.float32),
            size=output_size,
            mode="nearest-exact",
        )[:, 0]
        > 0.5
    )
    resized_confidence = None
    if confidence is not None:
        resized_confidence = F.interpolate(
            confidence[:, None],
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    scale_x = output_w / input_w
    scale_y = output_h / input_h
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[:, 0, :] *= scale_x
    scaled_intrinsics[:, 1, :] *= scale_y
    # F.interpolate and patch-token grids use pixel-center coordinates:
    # x_out + 0.5 = scale_x * (x_in + 0.5).
    scaled_intrinsics[:, 0, 2] += 0.5 * scale_x - 0.5
    scaled_intrinsics[:, 1, 2] += 0.5 * scale_y - 0.5
    scaled_intrinsics[:, 2, :] = intrinsics[:, 2, :]
    return ResizedCameraGeometry(
        depth=resized_depth,
        mask=resized_mask,
        confidence=resized_confidence,
        intrinsics=scaled_intrinsics,
    )
