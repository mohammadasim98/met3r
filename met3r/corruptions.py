"""Deterministic controlled corruptions for metric-fidelity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


SUPPORTED_CORRUPTIONS = {
    "clean",
    "translation",
    "thin_plate_warp",
    "exposure",
    "blur",
    "jpeg",
    "part_transfer",
}


@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    strength: float

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_CORRUPTIONS:
            raise ValueError(f"unknown corruption {self.kind!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("corruption strength must be in [0, 1]")


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"image must have shape (H,W,C>=3), got {value.shape}")
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def _translate(image: np.ndarray, strength: float) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    offset_x = int(round(width * strength))
    transform = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _thin_plate_warp(
    image: np.ndarray,
    strength: float,
    seed: int,
    *,
    interpolation: int | None = None,
    border_value: int | tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    import cv2
    from scipy.interpolate import RBFInterpolator

    if strength == 0:
        return image.copy()
    height, width = image.shape[:2]
    xs = np.linspace(0, width - 1, 4)
    ys = np.linspace(0, height - 1, 4)
    source_points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    displacement = rng.normal(
        scale=strength * min(height, width) * 0.10,
        size=source_points.shape,
    )
    boundary = (
        (source_points[:, 0] == 0)
        | (source_points[:, 0] == width - 1)
        | (source_points[:, 1] == 0)
        | (source_points[:, 1] == height - 1)
    )
    displacement[boundary] *= 0.25
    destination_points = source_points + displacement
    inverse_warp = RBFInterpolator(
        destination_points,
        source_points,
        kernel="thin_plate_spline",
        smoothing=1e-4,
    )
    output_grid = np.stack(
        np.meshgrid(np.arange(width), np.arange(height)), axis=-1
    ).reshape(-1, 2)
    source_grid = inverse_warp(output_grid).reshape(height, width, 2).astype(np.float32)
    return cv2.remap(
        image,
        source_grid[..., 0],
        source_grid[..., 1],
        interpolation=cv2.INTER_LINEAR if interpolation is None else interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _jpeg(image: np.ndarray, strength: float) -> np.ndarray:
    quality = max(5, int(round(100 - 95 * strength)))
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"))


def apply_corruption(
    image: np.ndarray,
    spec: CorruptionSpec,
    *,
    seed: int = 0,
    donor: np.ndarray | None = None,
    part_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one controlled corruption while preserving size and uint8 RGB format."""

    source = _as_rgb_uint8(image)
    if spec.kind == "clean" or spec.strength == 0:
        return source.copy()
    if spec.kind == "translation":
        return _translate(source, spec.strength)
    if spec.kind == "thin_plate_warp":
        return _thin_plate_warp(source, spec.strength, seed)
    if spec.kind == "exposure":
        factor = 1.0 + spec.strength
        return np.clip(source.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if spec.kind == "blur":
        radius = 0.25 + 5.75 * spec.strength
        return np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(radius)))
    if spec.kind == "jpeg":
        return _jpeg(source, spec.strength)
    if spec.kind == "part_transfer":
        if donor is None or part_mask is None:
            raise ValueError("part_transfer requires both donor and part_mask")
        donor_rgb = _as_rgb_uint8(donor)
        if donor_rgb.shape != source.shape or np.asarray(part_mask).shape != source.shape[:2]:
            raise ValueError("donor and part_mask shapes must match the source image")
        mask_value = np.asarray(part_mask)
        if np.issubdtype(mask_value.dtype, np.integer):
            if mask_value.min() < 0 or mask_value.max() > 255:
                raise ValueError("integer part_mask values must be in [0, 255]")
            mask_value = mask_value.astype(np.float32)
            if mask_value.max() > 1:
                mask_value /= 255.0
        else:
            mask_value = mask_value.astype(np.float32)
            if (
                not np.isfinite(mask_value).all()
                or mask_value.min() < 0
                or mask_value.max() > 1
            ):
                raise ValueError("floating part_mask values must be finite and in [0, 1]")
        alpha = mask_value[..., None] * spec.strength
        return np.clip(source * (1.0 - alpha) + donor_rgb * alpha, 0, 255).astype(np.uint8)
    raise AssertionError(f"unhandled corruption {spec.kind!r}")


def apply_corruption_with_mask(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    spec: CorruptionSpec,
    *,
    seed: int = 0,
    donor: np.ndarray | None = None,
    part_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a corruption and move the foreground mask for spatial variants."""

    import cv2

    source = _as_rgb_uint8(image)
    mask = np.asarray(foreground_mask, dtype=bool)
    if mask.shape != source.shape[:2]:
        raise ValueError("foreground_mask must match the image spatial shape")
    changed = apply_corruption(
        source,
        spec,
        seed=seed,
        donor=donor,
        part_mask=part_mask,
    )
    if spec.kind == "translation" and spec.strength > 0:
        height, width = mask.shape
        offset_x = int(round(width * spec.strength))
        transform = np.array(
            [[1.0, 0.0, offset_x], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        changed_mask = cv2.warpAffine(
            mask.astype(np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
    elif spec.kind == "thin_plate_warp" and spec.strength > 0:
        changed_mask = _thin_plate_warp(
            mask.astype(np.uint8),
            spec.strength,
            seed,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
        ) > 0
    else:
        changed_mask = mask.copy()
    return changed, changed_mask
