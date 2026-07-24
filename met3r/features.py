"""Backbone-independent multi-resolution feature extraction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

NAF_REVISION = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
NAF_REPOSITORY = f"valeoai/NAF:{NAF_REVISION}"


@dataclass(frozen=True)
class BackboneSpec:
    """Pinned identity and token layout for one experiment backbone."""

    family: str
    source: str
    model_id: str
    patch_size: int
    register_tokens: int = 0
    revision: str | None = None

    @property
    def native_grid_512(self) -> int:
        return 512 // self.patch_size


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "dino_vitb16": BackboneSpec(
        family="dino",
        source="torch_hub",
        model_id=(
            "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"
            "::dino_vitb16"
        ),
        patch_size=16,
        revision="7c446df5b9f45747937fb0d72314eb9f7b66930a",
    ),
    "dinov2_vitb14": BackboneSpec(
        family="dinov2",
        source="huggingface",
        model_id="facebook/dinov2-base",
        patch_size=14,
        revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    ),
    "dinov3_vitb16": BackboneSpec(
        family="dinov3",
        source="huggingface",
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch_size=16,
        register_tokens=4,
        revision="5931719e67bbdb9737e363e781fb0c67687896bc",
    ),
    "dinov3_vitl16_pixal3d": BackboneSpec(
        family="dinov3",
        source="huggingface",
        model_id="camenduru/dinov3-vitl16-pretrain-lvd1689m",
        patch_size=16,
        register_tokens=4,
        revision="3c276edd87d6f6e569ff0c4400e086807d0f3881",
    ),
}


class HuggingFacePatchBackbone(nn.Module):
    """Adapt a Hugging Face DINO model to normalized BCHW patch features."""

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: int,
        register_tokens: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.register_tokens = register_tokens
        self.normalize = normalize
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        pixel_values = images
        if self.normalize:
            pixel_values = (pixel_values - self.image_mean) / self.image_std
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        prefix_tokens = 1 + self.register_tokens
        patch_tokens = tokens[:, prefix_tokens:]
        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"backbone returned {patch_tokens.shape[1]} patch tokens; expected {expected}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            images.shape[0],
            patch_tokens.shape[-1],
            grid_h,
            grid_w,
        )


class TorchHubDinoPatchBackbone(nn.Module):
    """Adapt the original torch-hub DINO ViT to BCHW patch features."""

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.normalize = normalize
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        pixel_values = images
        if self.normalize:
            pixel_values = (pixel_values - self.image_mean) / self.image_std
        tokens = self.model.get_intermediate_layers(pixel_values, n=1)[0]
        patch_tokens = tokens[:, 1:]
        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"backbone returned {patch_tokens.shape[1]} patch tokens; expected {expected}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            images.shape[0],
            patch_tokens.shape[-1],
            grid_h,
            grid_w,
        )


class MultiScaleFeatureExtractor(nn.Module):
    """Extract native features and construct named interpolation/NAF variants."""

    def __init__(self, backbone: nn.Module, *, naf_upsampler: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.naf_upsampler = naf_upsampler

    @torch.inference_mode()
    def extract(
        self,
        images: Tensor,
        *,
        bilinear_scales: Sequence[int] = (),
        naf_scales: Sequence[int] = (),
        naf_guides: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        native = self.backbone(images)
        if native.ndim != 4:
            raise ValueError("backbone must return BCHW patch features")

        outputs = {"native": native}
        for scale in bilinear_scales:
            _validate_scale(scale)
            outputs[f"bilinear_{scale}"] = F.interpolate(
                native,
                size=(scale, scale),
                mode="bilinear",
                align_corners=False,
            )

        if naf_scales:
            if self.naf_upsampler is None:
                raise ValueError("naf_scales require a NAF upsampler")
            if not naf_guides:
                raise ValueError("naf_scales require at least one named RGB guide")
            for guide_name, guide in naf_guides.items():
                if guide.shape != images.shape:
                    raise ValueError(f"NAF guide {guide_name!r} must match the input image shape")
                for scale in naf_scales:
                    _validate_scale(scale)
                    outputs[f"naf_{guide_name}_{scale}"] = self.naf_upsampler(
                        guide,
                        native,
                        (scale, scale),
                    )
        return outputs


def build_naf_guides(
    images: Tensor,
    *,
    blur_sigma: float = 4.0,
    shuffle_indices: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build the fixed guide ablations used by the NAF fidelity experiment."""

    if blur_sigma <= 0:
        raise ValueError("blur_sigma must be positive")
    if images.shape[0] < 2:
        raise ValueError("shuffled NAF guide requires at least two images")
    max_kernel = min(images.shape[-2:])
    if max_kernel % 2 == 0:
        max_kernel -= 1
    desired_kernel = int(round(8 * blur_sigma + 1))
    if desired_kernel % 2 == 0:
        desired_kernel += 1
    kernel = max(3, min(desired_kernel, max_kernel))
    if shuffle_indices is None:
        shuffle_indices = torch.roll(
            torch.arange(images.shape[0], device=images.device),
            shifts=1,
        )
    if shuffle_indices.shape != (images.shape[0],):
        raise ValueError("shuffle_indices must contain one index per image")
    return {
        "normal": images,
        "blurred": gaussian_blur(images, [kernel, kernel], [blur_sigma, blur_sigma]),
        "shuffled": images.index_select(0, shuffle_indices.to(images.device)),
    }


def load_feature_backbone(
    name: str,
    *,
    hf_loader: Callable[..., nn.Module] | None = None,
    hub_loader: Callable[..., nn.Module] | None = None,
) -> nn.Module:
    """Load and freeze one pinned experiment backbone."""

    try:
        spec = BACKBONE_SPECS[name]
    except KeyError as error:
        available = ", ".join(sorted(BACKBONE_SPECS))
        raise ValueError(f"unknown backbone {name!r}; choose from {available}") from error

    if spec.source == "huggingface":
        if hf_loader is None:
            from transformers import AutoModel

            hf_loader = AutoModel.from_pretrained
        model = hf_loader(spec.model_id, revision=spec.revision)
        adapter: nn.Module = HuggingFacePatchBackbone(
            model,
            patch_size=spec.patch_size,
            register_tokens=spec.register_tokens,
        )
    elif spec.source == "torch_hub":
        if hub_loader is None:
            hub_loader = torch.hub.load
        repo, entrypoint = spec.model_id.split("::", maxsplit=1)
        model = hub_loader(repo, entrypoint)
        adapter = TorchHubDinoPatchBackbone(
            model,
            patch_size=spec.patch_size,
        )
    else:
        raise ValueError(f"unsupported backbone source {spec.source!r}")

    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


def load_naf_upsampler(
    *,
    device: torch.device | str,
    hub_loader: Callable[..., nn.Module] | None = None,
) -> nn.Module:
    """Load the frozen NAF module with the same entrypoint used by Pixal3D."""

    hub_loader = hub_loader or torch.hub.load
    model = hub_loader(
        NAF_REPOSITORY,
        "naf",
        pretrained=True,
        device=device,
        trust_repo=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _validate_scale(scale: int) -> None:
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("feature scales must be positive integers")
