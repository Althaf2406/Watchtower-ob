"""Unit tests for Phase 1: Data Preprocessing, Asymmetric Clipping, Refinement, & Resampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.resampling_eolearn import MultiResolutionResampler, resample_band_array
from src.data.sen2fire_dataset import Sen2FireMultispectralDataset
from src.data.wsts_plus_dataset import WSTSPlusDataset


def test_asymmetric_radiometric_clipping() -> None:
    # 12 bands: 0..9 (B1..B10), 10..11 (B11..B12)
    s2 = np.zeros((12, 10, 10), dtype=np.float32)
    s2[:10] = 15000.0   # Should clip to 1.0 (since 15000/10000 = 1.5 > 1.0)
    s2[10:] = 25000.0   # Should clip to 2.0 (since 25000/10000 = 2.5 > 2.0)

    aerosol = np.array([[[-5.0, 10.0]]], dtype=np.float32)  # Should clip to [-3.0, 8.0]

    clipped_s2, clipped_aero = Sen2FireMultispectralDataset.apply_asymmetric_radiometric_clipping(s2, aerosol)

    assert np.allclose(clipped_s2[:10].max(), 1.0)
    assert np.allclose(clipped_s2[10:].max(), 2.0)
    assert np.allclose(clipped_aero.min(), -3.0)
    assert np.allclose(clipped_aero.max(), 8.0)


def test_spectral_indices_computation() -> None:
    ds = Sen2FireMultispectralDataset("data/nonexistent", synthetic_fallback_samples=1)
    s2_clipped = np.full((12, 8, 8), 0.5, dtype=np.float32)
    s2_clipped[2] = 0.2   # B3 Green
    s2_clipped[3] = 0.1   # B4 Red
    s2_clipped[7] = 0.8   # B8 NIR
    s2_clipped[8] = 0.6   # B8A Narrow NIR
    s2_clipped[10] = 0.7  # B11 SWIR1
    s2_clipped[11] = 0.4  # B12 SWIR2

    indices = ds._compute_spectral_indices(s2_clipped)
    assert indices.shape == (4, 8, 8)

    # NBR = (B8 - B12) / (B8 + B12) = (0.8 - 0.4) / (0.8 + 0.4) = 0.4 / 1.2 = 1/3
    expected_nbr = (0.8 - 0.4) / (0.8 + 0.4)
    assert np.allclose(indices[0], expected_nbr, atol=1e-4)

    # AF_SWIR = (B12 - B8A) / (B12 + B8A) = (0.4 - 0.6) / (0.4 + 0.6) = -0.2 / 1.0 = -0.2
    expected_af_swir = (0.4 - 0.6) / (0.4 + 0.6)
    assert np.allclose(indices[1], expected_af_swir, atol=1e-4)

    # NDVI = (B8 - B4) / (B8 + B4) = (0.8 - 0.1) / (0.8 + 0.1) = 0.7 / 0.9 = 7/9
    expected_ndvi = (0.8 - 0.1) / (0.8 + 0.1)
    assert np.allclose(indices[2], expected_ndvi, atol=1e-4)

    # NDWI = (B3 - B8) / (B3 + B8) = (0.2 - 0.8) / (0.2 + 0.8) = -0.6 / 1.0 = -0.6
    expected_ndwi = (0.2 - 0.8) / (0.2 + 0.8)
    assert np.allclose(indices[3], expected_ndwi, atol=1e-4)


def test_refine_multiclass_mask_rules() -> None:
    ds = Sen2FireMultispectralDataset("data/nonexistent", synthetic_fallback_samples=1)
    coarse_label = np.ones((1, 3, 1), dtype=np.uint8)  # 3 pixels, all positive fire initially
    s2 = np.zeros((12, 3, 1), dtype=np.float32)
    aero = np.zeros((1, 3, 1), dtype=np.float32)
    indices = np.zeros((4, 3, 1), dtype=np.float32)

    # Pixel 0: Smoke condition -> Demote to Class 0
    # B2 > 0.25, aerosol > 1.0, NBR >= 0.05
    s2[1, 0, 0] = 0.35      # B2
    aero[0, 0, 0] = 1.8     # aerosol
    indices[0, 0, 0] = 0.10 # NBR

    # Pixel 1: Standard Burn Scar -> Class 1
    s2[1, 1, 0] = 0.05
    aero[0, 1, 0] = 0.2
    indices[0, 1, 0] = -0.40

    # Pixel 2: Active Flame Front -> Class 2
    # B12 > 0.35, B12 / B8A > 1.8, B11 > 0.25
    s2[8, 2, 0] = 0.15      # B8A
    s2[10, 2, 0] = 0.40     # B11
    s2[11, 2, 0] = 0.60     # B12 (0.60 / 0.15 = 4.0 > 1.8)

    refined = ds._refine_multiclass_mask(coarse_label, s2, aero, indices)
    assert refined[0, 0] == 0, "Smoke pixel was not demoted to Class 0"
    assert refined[1, 0] == 1, "Standard burn scar pixel was not assigned Class 1"
    assert refined[2, 0] == 2, "Active thermal spike was not promoted to Class 2"


def test_positive_oversampling_crop() -> None:
    ds = Sen2FireMultispectralDataset(
        "data/nonexistent",
        patch_size=64,
        is_train=True,
        positive_oversample_prob=1.0,  # Always fire-centered
        synthetic_fallback_samples=1,
    )
    features = np.zeros((17, 256, 256), dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.int64)
    raw_aero = np.zeros((256, 256), dtype=np.float32)

    # Place a single isolated fire cluster at (180, 210)
    mask[180, 210] = 2

    crop_feat, crop_mask, _ = ds._crop_patch(features, mask, raw_aero)
    assert crop_feat.shape == (17, 64, 64)
    assert crop_mask.shape == (64, 64)
    assert (crop_mask == 2).any(), "100% positive oversampling failed to capture the fire pixel!"


def test_multiresolution_resampler_modes() -> None:
    resampler = MultiResolutionResampler(target_shape=(64, 64))
    b20 = np.random.uniform(0.1, 0.9, size=(6, 32, 32)).astype(np.float32)
    b60 = np.random.uniform(0.1, 0.9, size=(2, 11, 11)).astype(np.float32)
    lbl = np.random.choice([0, 1, 2], size=(16, 16)).astype(np.int64)

    up_b20 = resampler.resample_20m_swir_rededge(b20)
    up_b60 = resampler.resample_60m_and_aerosol(b60)
    up_lbl = resampler.resample_label(lbl)

    assert up_b20.shape == (6, 64, 64)
    assert up_b60.shape == (2, 64, 64)
    assert up_lbl.shape == (64, 64)
    # Nearest neighbor preserves discrete class integers exactly
    assert set(np.unique(up_lbl)).issubset({0, 1, 2})


def test_wsts_plus_windows_and_subsets() -> None:
    for win in (1, 5):
        for sub, c in [("Veg", 7), ("Multi", 33), ("All", 40)]:
            ds = WSTSPlusDataset(
                root_dir="data/nonexistent",
                split="train",
                temporal_window=win,
                feature_subset=sub,  # type: ignore[arg-type]
                patch_size=64,
                synthetic_fallback_samples=2,
            )
            item = ds[0]
            assert item["sequence"].shape == (win, c, 64, 64)
            assert item["mask"].shape == (64, 64)

