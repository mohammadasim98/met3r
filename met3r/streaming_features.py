"""One-variant-at-a-time feature upsampling for upper-view workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .features import build_naf_guides
from .upper_design import FeatureVariant

_Result = TypeVar("_Result")
_NAF_GUIDES = {
    "naf_normal_guide": "normal",
    "naf_blurred_guide": "blurred",
    "naf_shuffled_guide": "shuffled",
}


def iter_feature_variants(
    native: Tensor,
    images: Tensor,
    variants: Sequence[FeatureVariant],
    naf: nn.Module | None,
) -> Iterator[tuple[FeatureVariant, Tensor]]:
    """Yield native or exactly one generated feature tensor at a time."""

    if native.ndim != 4:
        raise ValueError("native features must be a BCHW tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    if native.shape[0] != images.shape[0]:
        raise ValueError("native features and images must have the same batch size")

    guides: dict[str, Tensor] | None = None
    for variant in variants:
        if variant.feature_mode == "native":
            yield variant, native
            continue
        if variant.resolution <= 0:
            raise ValueError("feature resolution must be positive")
        with torch.inference_mode():
            if variant.feature_mode == "bilinear":
                upsampled = _retry_cuda_oom(
                    lambda: F.interpolate(
                        native,
                        (variant.resolution, variant.resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                try:
                    guide_name = _NAF_GUIDES[variant.feature_mode]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported feature mode {variant.feature_mode!r}"
                    ) from error
                if variant.resolution < max(native.shape[-2:]):
                    raise ValueError(
                        "NAF target resolution must be at least the native grid"
                    )
                if naf is None:
                    raise ValueError("NAF feature modes require a NAF upsampler")
                if guides is None:
                    guides = _retry_cuda_oom(lambda: build_naf_guides(images))
                upsampled = _retry_cuda_oom(
                    lambda: naf(
                        guides[guide_name],
                        native,
                        (variant.resolution, variant.resolution),
                    )
                )
        try:
            yield variant, upsampled
        finally:
            del upsampled


def _retry_cuda_oom(operation: Callable[[], _Result]) -> _Result:
    try:
        return operation()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return operation()
