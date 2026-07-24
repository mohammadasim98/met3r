"""Deterministic fixed-basis PCA projection for feature visualizations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .upper_design import FeatureVariant

_PCA_COMPONENTS = 3
_PCA_SEED = 42
_PCA_FIELDS = {
    "mean",
    "components",
    "lower",
    "upper",
    "fingerprint",
    "seed",
}


@dataclass(frozen=True)
class PCABasis:
    """Small persistent PCA artifact shared by every image for a backbone."""

    mean: np.ndarray
    components: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fingerprint: str

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if self.fingerprint != pca_fingerprint(self):
            raise ValueError("PCA basis fingerprint mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            mean=self.mean,
            components=self.components,
            lower=self.lower,
            upper=self.upper,
            fingerprint=np.asarray(self.fingerprint),
            seed=np.asarray(_PCA_SEED),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PCABasis":
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != _PCA_FIELDS:
                raise ValueError("PCA basis contains unexpected fields")
            if int(np.asarray(artifact["seed"]).item()) != _PCA_SEED:
                raise ValueError("PCA basis seed mismatch")
            basis = cls(
                mean=np.asarray(artifact["mean"]),
                components=np.asarray(artifact["components"]),
                lower=np.asarray(artifact["lower"]),
                upper=np.asarray(artifact["upper"]),
                fingerprint=str(np.asarray(artifact["fingerprint"]).item()),
            )
        if basis.fingerprint != pca_fingerprint(basis):
            raise ValueError("PCA basis fingerprint mismatch")
        _validate_basis(basis)
        return basis


def pca_fingerprint(basis: PCABasis) -> str:
    """Hash all numeric PCA parameters with explicit dtype and shape metadata."""

    digest = hashlib.sha256()
    for name in ("mean", "components", "lower", "upper"):
        array = np.ascontiguousarray(np.asarray(getattr(basis, name)))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_pca_basis(
    batches: Iterable[Tensor],
    max_tokens: int = 200_000,
    seed: int = _PCA_SEED,
) -> PCABasis:
    """Fit one deterministic three-component basis over all supplied BCHW batches."""

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    flattened: list[Tensor] = []
    channels: int | None = None
    for batch in batches:
        if not isinstance(batch, Tensor) or batch.ndim != 4:
            raise ValueError("PCA feature batches must be BCHW tensors")
        if channels is None:
            channels = batch.shape[1]
        elif batch.shape[1] != channels:
            raise ValueError("PCA feature batches must have the same channel count")
        if batch.shape[0] == 0 or batch.shape[-2] == 0 or batch.shape[-1] == 0:
            continue
        flattened.append(
            batch.detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(0, 2, 3, 1)
            .reshape(-1, batch.shape[1])
        )
    if not flattened:
        raise ValueError("PCA fitting requires at least one non-empty feature batch")

    tokens = torch.cat(flattened, dim=0)
    if tokens.shape[0] > max_tokens:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(tokens.shape[0], generator=generator)[:max_tokens]
        tokens = tokens.index_select(0, indices)
    if min(tokens.shape) < _PCA_COMPONENTS:
        raise ValueError("PCA fitting requires at least three tokens and channels")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("PCA feature batches must contain only finite values")

    mean = tokens.mean(dim=0)
    centered = tokens - mean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, components = torch.pca_lowrank(
            centered,
            q=_PCA_COMPONENTS,
            center=False,
        )
    for component_index in range(_PCA_COMPONENTS):
        component = components[:, component_index]
        largest_index = int(component.abs().argmax().item())
        if component[largest_index] < 0:
            components[:, component_index].mul_(-1)

    projected = centered @ components
    lower = torch.quantile(projected, 0.01, dim=0)
    upper = torch.quantile(projected, 0.99, dim=0)
    incomplete = PCABasis(
        mean=mean.numpy(),
        components=components.numpy(),
        lower=lower.numpy(),
        upper=upper.numpy(),
        fingerprint="",
    )
    basis = PCABasis(
        mean=incomplete.mean,
        components=incomplete.components,
        lower=incomplete.lower,
        upper=incomplete.upper,
        fingerprint=pca_fingerprint(incomplete),
    )
    _validate_basis(basis)
    return basis


def project_pca(feature: Tensor, basis: PCABasis) -> np.ndarray:
    """Project BCHW features to fixed-bound uint8 RGB without image normalization."""

    if not isinstance(feature, Tensor) or feature.ndim != 4:
        raise ValueError("feature must be a BCHW tensor")
    _validate_basis(basis)
    if feature.shape[1] != basis.components.shape[0]:
        raise ValueError("feature channel count does not match PCA basis")
    height, width = feature.shape[-2:]
    tokens = (
        feature.detach()
        .to(device="cpu", dtype=torch.float32)
        .permute(0, 2, 3, 1)
        .reshape(-1, feature.shape[1])
    )
    mean = torch.from_numpy(np.asarray(basis.mean)).to(dtype=tokens.dtype)
    components = torch.from_numpy(np.asarray(basis.components)).to(dtype=tokens.dtype)
    lower = torch.from_numpy(np.asarray(basis.lower)).to(dtype=tokens.dtype)
    upper = torch.from_numpy(np.asarray(basis.upper)).to(dtype=tokens.dtype)
    projected = (tokens - mean) @ components
    span = upper - lower
    normalized = torch.where(
        span > 0,
        (projected - lower) / torch.clamp_min(span, torch.finfo(tokens.dtype).eps),
        torch.zeros_like(projected),
    )
    rgb = (
        normalized.clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .reshape(feature.shape[0], height, width, _PCA_COMPONENTS)
    )
    return rgb.numpy()


def render_pca_png(
    feature: Tensor,
    basis: PCABasis,
    variant: FeatureVariant,
    path: str | Path,
    *,
    track_max_resolution: int,
) -> Path | None:
    """Render only the native/lowest and track-highest retained PCA variants."""

    if (
        variant.feature_mode != "native"
        and variant.resolution != track_max_resolution
    ):
        return None
    if feature.shape[0] != 1:
        raise ValueError("render_pca_png requires exactly one feature image")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(project_pca(feature, basis)[0], mode="RGB").save(destination)
    return destination


def _validate_basis(basis: PCABasis) -> None:
    mean = np.asarray(basis.mean)
    components = np.asarray(basis.components)
    lower = np.asarray(basis.lower)
    upper = np.asarray(basis.upper)
    if mean.ndim != 1:
        raise ValueError("PCA mean must be one-dimensional")
    if components.shape != (mean.shape[0], _PCA_COMPONENTS):
        raise ValueError("PCA components must have shape [channels, 3]")
    if lower.shape != (_PCA_COMPONENTS,) or upper.shape != (_PCA_COMPONENTS,):
        raise ValueError("PCA bounds must contain three values")
    if not all(np.isfinite(array).all() for array in (mean, components, lower, upper)):
        raise ValueError("PCA basis must contain only finite values")
