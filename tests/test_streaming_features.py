from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from met3r.features import MultiScaleFeatureExtractor
from met3r.streaming_features import iter_feature_variants
from met3r.upper_design import FeatureVariant


class FakeBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(images, kernel_size=4, stride=4)


class FakeNAF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.guides: list[torch.Tensor] = []

    def forward(
        self,
        guide: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        self.guides.append(guide)
        return F.interpolate(features, output_size, mode="nearest") + 1


class ParameterizedNAF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 3, kernel_size=1)

    def forward(
        self,
        guide: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del guide, output_size
        return self.projection(features)


def _variants() -> tuple[FeatureVariant, ...]:
    return (
        FeatureVariant("t", "b", 8, "native", 2),
        FeatureVariant("t", "b", 8, "bilinear", 4),
        FeatureVariant("t", "b", 8, "naf_normal_guide", 8),
        FeatureVariant("t", "b", 8, "naf_blurred_guide", 8),
    )


def test_streaming_generator_yields_variants_in_order_with_expected_shapes() -> None:
    native = torch.ones(4, 8, 2, 2)
    images = torch.ones(4, 3, 8, 8)
    variants = _variants()

    results = list(iter_feature_variants(native, images, variants, FakeNAF()))

    assert [variant.key for variant, _ in results] == [
        item.key for item in variants
    ]
    assert [tuple(feature.shape[-2:]) for _, feature in results] == [
        (2, 2),
        (4, 4),
        (8, 8),
        (8, 8),
    ]
    assert results[0][1] is native


def test_streaming_generator_releases_previous_upsampled_variant() -> None:
    native = torch.ones(4, 8, 2, 2)
    images = torch.ones(4, 3, 8, 8)
    iterator = iter_feature_variants(native, images, _variants(), FakeNAF())
    next(iterator)
    _, bilinear = next(iterator)
    reference = weakref.ref(bilinear)
    del bilinear

    next(iterator)
    gc.collect()

    assert reference() is None
    iterator.close()


def test_streaming_parameterized_naf_accepts_native_inference_tensor_without_grad() -> None:
    images = torch.rand(4, 3, 8, 8)
    native = MultiScaleFeatureExtractor(FakeBackbone()).extract_native(images)
    variants = (
        FeatureVariant("t", "b", 8, "naf_normal_guide", 2),
        FeatureVariant("t", "b", 8, "naf_blurred_guide", 2),
    )
    iterator = iter_feature_variants(native, images, variants, ParameterizedNAF())

    _, first = next(iterator)
    assert not first.requires_grad
    reference = weakref.ref(first)
    del first
    _, second = next(iterator)
    gc.collect()

    assert not second.requires_grad
    assert reference() is None
    iterator.close()


def test_naf_guides_are_built_lazily_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import met3r.streaming_features as streaming

    calls = 0
    original = streaming.build_naf_guides

    def counted_guides(images: torch.Tensor) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(images)

    monkeypatch.setattr(streaming, "build_naf_guides", counted_guides)
    native = torch.ones(4, 8, 2, 2)
    images = torch.ones(4, 3, 8, 8)
    iterator = iter_feature_variants(native, images, _variants(), FakeNAF())

    next(iterator)
    next(iterator)
    assert calls == 0
    list(iterator)
    assert calls == 1


def test_normal_streaming_does_not_empty_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_empty_cache() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", counted_empty_cache)

    list(
        iter_feature_variants(
            torch.ones(4, 8, 2, 2),
            torch.ones(4, 3, 8, 8),
            _variants(),
            FakeNAF(),
        )
    )

    assert calls == 0


def test_cuda_oom_is_retried_once_after_emptying_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OOMOnceNAF(FakeNAF):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def forward(
            self,
            guide: torch.Tensor,
            features: torch.Tensor,
            output_size: tuple[int, int],
        ) -> torch.Tensor:
            self.attempts += 1
            if self.attempts == 1:
                raise torch.cuda.OutOfMemoryError("test OOM")
            return super().forward(guide, features, output_size)

    cache_calls = 0

    def counted_empty_cache() -> None:
        nonlocal cache_calls
        cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", counted_empty_cache)
    naf = OOMOnceNAF()
    variants = (FeatureVariant("t", "b", 8, "naf_normal_guide", 8),)

    results = list(
        iter_feature_variants(
            torch.ones(4, 8, 2, 2),
            torch.ones(4, 3, 8, 8),
            variants,
            naf,
        )
    )

    assert len(results) == 1
    assert naf.attempts == 2
    assert cache_calls == 1


def test_unknown_feature_mode_is_rejected() -> None:
    variant = FeatureVariant("t", "b", 8, "unknown", 8)

    with pytest.raises(ValueError, match="unsupported feature mode"):
        list(
            iter_feature_variants(
                torch.ones(4, 8, 2, 2),
                torch.ones(4, 3, 8, 8),
                (variant,),
                FakeNAF(),
            )
        )


def test_naf_target_below_native_grid_is_rejected() -> None:
    variant = FeatureVariant("t", "b", 8, "naf_normal_guide", 1)

    with pytest.raises(
        ValueError,
        match="NAF target resolution must be at least the native grid",
    ):
        list(
            iter_feature_variants(
                torch.ones(4, 8, 2, 2),
                torch.ones(4, 3, 8, 8),
                (variant,),
                FakeNAF(),
            )
        )


def test_extract_native_preserves_existing_extractor_api() -> None:
    images = torch.rand(2, 3, 8, 8)
    extractor = MultiScaleFeatureExtractor(FakeBackbone(), naf_upsampler=FakeNAF())

    native = extractor.extract_native(images)
    outputs = extractor.extract(images, bilinear_scales=(4,))

    assert native.shape == (2, 3, 2, 2)
    torch.testing.assert_close(native, outputs["native"])
