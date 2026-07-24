from __future__ import annotations

import torch
import pytest
from torch import nn
from torch.nn import functional as F

from met3r.features import (
    BACKBONE_SPECS,
    HuggingFacePatchBackbone,
    MultiScaleFeatureExtractor,
    TorchHubDinoPatchBackbone,
    build_naf_guides,
    load_feature_backbone,
    load_naf_upsampler,
    NAF_REPOSITORY,
)


class FakeBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(images, kernel_size=16, stride=16)


class FakeNAF(nn.Module):
    def forward(
        self,
        guide: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del guide
        return F.interpolate(features, output_size, mode="nearest") + 1


def test_multiscale_extractor_returns_native_bilinear_and_naf_features() -> None:
    images = torch.rand(2, 3, 512, 512)
    extractor = MultiScaleFeatureExtractor(FakeBackbone(), naf_upsampler=FakeNAF())

    outputs = extractor.extract(
        images,
        bilinear_scales=(32, 64),
        naf_scales=(32, 64),
        naf_guides={"normal": images},
    )

    assert outputs["native"].shape == (2, 3, 32, 32)
    assert outputs["bilinear_32"].shape == (2, 3, 32, 32)
    assert outputs["bilinear_64"].shape == (2, 3, 64, 64)
    assert outputs["naf_normal_32"].shape == (2, 3, 32, 32)
    assert outputs["naf_normal_64"].shape == (2, 3, 64, 64)
    torch.testing.assert_close(outputs["naf_normal_32"], outputs["native"] + 1)


class FakeHFOutput:
    def __init__(self, tokens: torch.Tensor) -> None:
        self.last_hidden_state = tokens


class FakeHFModel(nn.Module):
    def forward(self, *, pixel_values: torch.Tensor) -> FakeHFOutput:
        batch = pixel_values.shape[0]
        prefix = torch.full((batch, 5, 4), -1.0, device=pixel_values.device)
        patches = torch.arange(32 * 32 * 4, device=pixel_values.device, dtype=torch.float32)
        patches = patches.reshape(1, 32 * 32, 4).expand(batch, -1, -1)
        return FakeHFOutput(torch.cat((prefix, patches), dim=1))


def test_huggingface_backbone_removes_cls_and_register_tokens() -> None:
    adapter = HuggingFacePatchBackbone(
        FakeHFModel(),
        patch_size=16,
        register_tokens=4,
        normalize=False,
    )

    features = adapter(torch.rand(2, 3, 512, 512))

    assert features.shape == (2, 4, 32, 32)
    assert torch.all(features[:, :, 0, 0] == torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_backbone_registry_contains_matched_base_models_and_pixal3d_large() -> None:
    assert BACKBONE_SPECS["dino_vitb16"].patch_size == 16
    assert BACKBONE_SPECS["dinov2_vitb14"].patch_size == 14
    assert BACKBONE_SPECS["dinov3_vitb16"].patch_size == 16
    assert BACKBONE_SPECS["dinov3_vitl16_pixal3d"].revision == (
        "3c276edd87d6f6e569ff0c4400e086807d0f3881"
    )
    assert all(
        spec.revision is not None
        for spec in BACKBONE_SPECS.values()
    )


class FakeDinoV1(nn.Module):
    def get_intermediate_layers(self, images: torch.Tensor, n: int) -> list[torch.Tensor]:
        assert n == 1
        batch = images.shape[0]
        cls = torch.full((batch, 1, 4), -1.0)
        patches = torch.arange(32 * 32 * 4, dtype=torch.float32)
        patches = patches.reshape(1, 32 * 32, 4).expand(batch, -1, -1)
        return [torch.cat((cls, patches), dim=1)]


def test_torch_hub_dino_backbone_removes_cls_token() -> None:
    adapter = TorchHubDinoPatchBackbone(
        FakeDinoV1(),
        patch_size=16,
        normalize=False,
    )

    features = adapter(torch.rand(1, 3, 512, 512))

    assert features.shape == (1, 4, 32, 32)
    assert torch.all(features[0, :, 0, 0] == torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_naf_guides_include_normal_blurred_and_explicitly_shuffled_images() -> None:
    images = torch.zeros(2, 3, 9, 9)
    images[0, :, 4, 4] = 1
    images[1] = 0.5

    guides = build_naf_guides(
        images,
        blur_sigma=2.0,
        shuffle_indices=torch.tensor([1, 0]),
    )

    torch.testing.assert_close(guides["normal"], images)
    torch.testing.assert_close(guides["shuffled"], images.flip(0))
    assert guides["blurred"][0, :, 4, 4].max().item() < 1


def test_naf_shuffled_guide_rejects_single_image_batch() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_naf_guides(torch.zeros(1, 3, 9, 9))


def test_feature_loader_pins_pixal3d_model_revision() -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_hf_loader(model_id: str, *, revision: str | None) -> nn.Module:
        calls.append((model_id, revision))
        return FakeHFModel()

    adapter = load_feature_backbone(
        "dinov3_vitl16_pixal3d",
        hf_loader=fake_hf_loader,
    )

    assert isinstance(adapter, HuggingFacePatchBackbone)
    assert calls == [
        (
            "camenduru/dinov3-vitl16-pretrain-lvd1689m",
            "3c276edd87d6f6e569ff0c4400e086807d0f3881",
        )
    ]


def test_naf_loader_uses_a_pinned_repository_revision() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_hub_loader(
        repository: str,
        entrypoint: str,
        **kwargs: object,
    ) -> nn.Module:
        calls.append((repository, entrypoint, kwargs))
        return FakeNAF()

    model = load_naf_upsampler(device="cpu", hub_loader=fake_hub_loader)

    assert isinstance(model, FakeNAF)
    assert calls[0][0] == NAF_REPOSITORY
    assert ":" in NAF_REPOSITORY
    assert len(NAF_REPOSITORY.rsplit(":", maxsplit=1)[1]) == 40
