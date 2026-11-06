"""Unit tests for Phase 2: Weight Surgery & Neural Architecture Specifications."""

from __future__ import annotations

import pytest
import torch

from src.models.attention_unet_r50 import SSL4EOAttentionUNetResNet50
from src.models.compact_vit_s16 import SSL4EOCompactViTS16Seg, interpolate_pos_embed_2d
from src.models.weight_surgery import adapt_ssl4eo_first_layer_weights


def test_weight_surgery_landsat_7ch() -> None:
    # 7 Landsat OLI bands: [Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2]
    w_src = torch.ones((32, 7, 7, 7), dtype=torch.float32)
    w_src[:, 3] = 2.0  # Red
    w_src[:, 4] = 6.0  # NIR

    w_target = adapt_ssl4eo_first_layer_weights(w_src, source_sensor="ssl4eo_l", target_in_chans=17)
    assert w_target.shape == (32, 17, 7, 7)

    # Variance-preserving factor: C_pre / C_target = 7 / 17
    scale = 7.0 / 17.0

    # Red-Edge B5 = (0.75 * Red + 0.25 * NIR) * scale = (0.75*2 + 0.25*6)*scale = 3.0 * scale
    assert torch.allclose(w_target[:, 4], torch.full_like(w_target[:, 4], 3.0 * scale), atol=1e-5)
    # Red-Edge B6 = (0.50 * Red + 0.50 * NIR) * scale = (1.0 + 3.0)*scale = 4.0 * scale
    assert torch.allclose(w_target[:, 5], torch.full_like(w_target[:, 5], 4.0 * scale), atol=1e-5)
    # Red-Edge B7 = (0.25 * Red + 0.75 * NIR) * scale = (0.5 + 4.5)*scale = 5.0 * scale
    assert torch.allclose(w_target[:, 6], torch.full_like(w_target[:, 6], 5.0 * scale), atol=1e-5)

    # S5P UV Aerosol B13 (idx 12) = 0.5 * (B1 + B2) * scale = 0.5 * (1 + 1) * scale = 1.0 * scale
    assert torch.allclose(w_target[:, 12], torch.full_like(w_target[:, 12], 1.0 * scale), atol=1e-5)


def test_weight_surgery_sentinel2_13ch() -> None:
    w_src = torch.randn((64, 13, 7, 7), dtype=torch.float32)
    w_target = adapt_ssl4eo_first_layer_weights(w_src, source_sensor="ssl4eo_s12", target_in_chans=17)
    assert w_target.shape == (64, 17, 7, 7)

    scale = 13.0 / 17.0
    # Optical bands B1..B9 (0..9) should match w_src[:, 0:10] * scale
    assert torch.allclose(w_target[:, 0:10], w_src[:, 0:10] * scale, atol=1e-5)
    # SWIR1 (idx 10) corresponds to S2 band 11 (src idx 11)
    assert torch.allclose(w_target[:, 10], w_src[:, 11] * scale, atol=1e-5)
    # SWIR2 (idx 11) corresponds to S2 band 12 (src idx 12)
    assert torch.allclose(w_target[:, 11], w_src[:, 12] * scale, atol=1e-5)


def test_pos_embed_2d_bicubic_interpolation() -> None:
    # ViT-S/16 pos_embed for 224x224 has 1 + 14x14 = 197 tokens
    pos_224 = torch.randn(1, 197, 384)
    # Target 256x256 grid has 1 + 16x16 = 257 tokens
    pos_256 = interpolate_pos_embed_2d(pos_224, target_grid_size=(16, 16), num_prefix_tokens=1)
    assert pos_256.shape == (1, 257, 384)
    # Prefix CLS token remains identical
    assert torch.equal(pos_256[:, :1, :], pos_224[:, :1, :])


def test_attention_unet_lateral_channels_and_forward() -> None:
    model = SSL4EOAttentionUNetResNet50(in_channels=17, num_classes=3, ssl_source="ssl4eo_s12")
    assert model.lateral_channels == (256, 192, 128, 64, 32)

    x = torch.randn(2, 17, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 3, 128, 128)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    assert 25.0 <= param_count <= 28.0, f"Expected ~26.24M parameters, got {param_count:.2f}M"


def test_compact_vit_intermediate_extraction_and_forward() -> None:
    model = SSL4EOCompactViTS16Seg(
        in_channels=17,
        num_classes=3,
        img_size=128,
        patch_size=16,
        ssl_source="ssl4eo_l",
    )
    assert model.extract_blocks == (2, 5, 8, 11)

    x = torch.randn(2, 17, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 3, 128, 128)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    assert 22.0 <= param_count <= 26.0, f"Expected ~22.81M - 24.3M parameters, got {param_count:.2f}M"

