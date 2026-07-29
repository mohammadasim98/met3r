from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from met3r.pca_features import (
    PCABasis,
    fit_pca_basis,
    pca_fingerprint,
    project_pca,
    render_pca_png,
)
from met3r.upper_design import FeatureVariant


def test_fixed_pca_colors_are_shared_across_images(tmp_path: Path) -> None:
    first = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    second = first + 10

    basis = fit_pca_basis([first, second], max_tokens=100, seed=42)
    path = tmp_path / "basis.npz"
    basis.save(path)
    loaded = PCABasis.load(path)

    np.testing.assert_allclose(loaded.components, basis.components)
    assert loaded.fingerprint == basis.fingerprint
    assert not np.array_equal(
        project_pca(first, basis),
        project_pca(first, fit_pca_basis([first], seed=42)),
    )


def test_pca_fit_is_deterministic_subsampled_and_sign_canonicalized() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(2, 5, 9, 11, generator=generator)

    first = fit_pca_basis([features], max_tokens=37, seed=42)
    second = fit_pca_basis([features], max_tokens=37, seed=42)

    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    assert first.fingerprint == second.fingerprint
    largest_loadings = np.abs(first.components).argmax(axis=0)
    assert all(
        first.components[index, component] >= 0
        for component, index in enumerate(largest_loadings)
    )


def test_pca_fit_does_not_call_all_device_manual_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_all_device_seed(seed: int) -> None:
        del seed
        raise AssertionError("fit_pca_basis must seed only the CPU generator")

    monkeypatch.setattr(torch, "manual_seed", reject_all_device_seed)

    fit_pca_basis(
        [torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 4)],
        seed=42,
    )


def test_pca_fit_preserves_cpu_and_initialized_cuda_rng_state() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 4:
        pytest.skip("permitted CUDA device 4 is unavailable")

    cpu_before = torch.random.get_rng_state().clone()
    with torch.cuda.device(4):
        cuda_original = torch.cuda.get_rng_state(4).clone()
        try:
            torch.cuda.manual_seed(17)
            cuda_before = torch.cuda.get_rng_state(4).clone()

            fit_pca_basis(
                [torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 4)],
                seed=42,
            )

            torch.testing.assert_close(torch.random.get_rng_state(), cpu_before)
            torch.testing.assert_close(torch.cuda.get_rng_state(4), cuda_before)
        finally:
            torch.cuda.set_rng_state(cuda_original, 4)


def test_projection_uses_stored_global_bounds_without_per_image_normalization() -> None:
    incomplete = PCABasis(
        mean=np.zeros(3, dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        lower=np.zeros(3, dtype=np.float32),
        upper=np.full(3, 10, dtype=np.float32),
        fingerprint="",
    )
    basis = PCABasis(
        mean=incomplete.mean,
        components=incomplete.components,
        lower=incomplete.lower,
        upper=incomplete.upper,
        fingerprint=pca_fingerprint(incomplete),
    )
    feature = torch.full((1, 3, 2, 2), 5.0)

    rendered = project_pca(feature, basis)

    assert rendered.shape == (1, 2, 2, 3)
    assert rendered.dtype == np.uint8
    assert np.all(rendered == 128)


def test_pca_load_rejects_tampered_parameters(tmp_path: Path) -> None:
    basis = fit_pca_basis(
        [torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 4)],
        seed=42,
    )
    path = basis.save(tmp_path / "basis.npz")
    with np.load(path) as artifact:
        payload = {name: np.asarray(artifact[name]) for name in artifact.files}
    payload["mean"] = payload["mean"] + 1
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        PCABasis.load(path)


def test_pca_save_rejects_malformed_basis_with_matching_fingerprint(
    tmp_path: Path,
) -> None:
    incomplete = PCABasis(
        mean=np.array([np.nan, 0, 0], dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        lower=np.zeros(3, dtype=np.float32),
        upper=np.ones(3, dtype=np.float32),
        fingerprint="",
    )
    malformed = PCABasis(
        mean=incomplete.mean,
        components=incomplete.components,
        lower=incomplete.lower,
        upper=incomplete.upper,
        fingerprint=pca_fingerprint(incomplete),
    )
    destination = tmp_path / "malformed.npz"

    with pytest.raises(ValueError, match="finite"):
        malformed.save(destination)

    assert not destination.exists()


@pytest.mark.parametrize("resolution", (74, 148, 296))
def test_render_pca_png_skips_intermediate_resolutions(
    tmp_path: Path,
    resolution: int,
) -> None:
    basis = fit_pca_basis(
        [torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)],
        seed=42,
    )
    variant = FeatureVariant(
        "user_37_grid_track",
        "dinov3_vitl16_pixal3d",
        592,
        "bilinear",
        resolution,
    )
    destination = tmp_path / f"{resolution}.png"

    result = render_pca_png(
        torch.ones(1, 3, 2, 2),
        basis,
        variant,
        destination,
        track_max_resolution=592,
    )

    assert result is None
    assert not destination.exists()


@pytest.mark.parametrize(
    ("feature_mode", "resolution"),
    (("native", 37), ("bilinear", 592)),
)
def test_render_pca_png_writes_native_and_track_highest(
    tmp_path: Path,
    feature_mode: str,
    resolution: int,
) -> None:
    basis = fit_pca_basis(
        [torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)],
        seed=42,
    )
    variant = FeatureVariant(
        "user_37_grid_track",
        "dinov3_vitl16_pixal3d",
        592,
        feature_mode,
        resolution,
    )
    destination = tmp_path / f"{feature_mode}.png"

    result = render_pca_png(
        torch.ones(1, 3, 2, 2),
        basis,
        variant,
        destination,
        track_max_resolution=592,
    )

    assert result == destination
    assert destination.is_file()


def test_pca_fit_rejects_invalid_or_incompatible_batches() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_pca_basis([])
    with pytest.raises(ValueError, match="BCHW"):
        fit_pca_basis([torch.ones(3, 2, 2)])
    with pytest.raises(ValueError, match="same channel"):
        fit_pca_basis(
            [
                torch.ones(1, 3, 2, 2),
                torch.ones(1, 4, 2, 2),
            ]
        )
    with pytest.raises(ValueError, match="max_tokens"):
        fit_pca_basis([torch.ones(1, 3, 2, 2)], max_tokens=0)
